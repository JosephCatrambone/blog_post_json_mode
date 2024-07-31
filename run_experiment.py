# Perform the actual experiment.
import json
import sqlite3
import time

import anthropic
import torch
from guardrails import Guard
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from task import Task, SCHEMAS, NUEXTRACT_SCHEMAS, EXAMPLES, OPENAI_TOOLS, DATA_MODELS, ANTHROPIC_TOOLS


device = "cuda" if torch.cuda.is_available() else "cpu"
active_model_name = None  # We'll run out of memory on our GPU if we load everything at the same time. Check this against the loaded model.
active_model = None
active_tokenizer = None


def load_cached_active_model(model_name: str):
    global active_model_name
    global active_model
    global active_tokenizer
    if model_name != active_model_name:
        if active_model is not None:
            active_model.to("cpu")
        del active_tokenizer
        del active_model
        if device == "cuda":
            torch.cuda.empty_cache()
        # Load model directly
        active_model_name = model_name
        active_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        active_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        active_model.to(device)
        active_model.eval()
    return active_model, active_tokenizer


def load_cached_pipe(model_name):
    global active_model_name
    global active_model
    if model_name != active_model_name:
        del active_model
        active_model_name = model_name
        active_model = pipeline("text-generation", model=model_name, device=device, max_new_tokens=2048, )
    return active_model


# Set up the inference for the numind nuextract case:
# %pip install torch
# %pip install transformers

# Taken directly from the NuMind NuExtract documentation
# https://huggingface.co/numind/NuExtract-tiny
def predict_nuextract(model_name, text, schema, examples=["", "", ""], device="cuda"):
    model, tokenizer = load_cached_active_model(f"numind/{model_name}")
    schema = json.dumps(json.loads(schema), indent=4)
    input_llm = "<|input|>\n### Template:\n" + schema + "\n"
    for i in examples:
        if i != "":
            input_llm += "### Example:\n" + json.dumps(json.loads(i), indent=4) + "\n"

    input_llm += "### Text:\n" + text + "\n<|output|>\n"
    input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=2048).to(device)

    output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)
    return output.split("<|output|>")[1].split("<|end-output|>")[0]


def predict_openai(client, model_name, document, schema, examples, use_json_mode: bool = False, function_calling_tools = None):
    messages = [
        {
            "role": "system",
            "content": f"You will be provided with unstructured data in the form of a document. Your task is to create a JSON object that adheres to the following schema:\n{schema}\n{examples}\nReturn only the result JSON. If data for a given field is not present, provide an empty array ([]) for arrays, an empty string for strings, and null for missing values."
        },
        {
            "role": "user",
            "content": document
        }
    ]
    extra_args = {}
    if use_json_mode:
        extra_args["response_format"] = {"type": "json_object"}
    if function_calling_tools:
        extra_args["tool_choice"] = "required"
        extra_args["tools"] = function_calling_tools
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
        max_tokens=2048,
        top_p=1,
        #response_format={"type": "json_object"}  # Toggle 'json mode'.
        **extra_args
    )
    if function_calling_tools:
        # There will almost certainly be only one function call, but...
        # TODO: Find a nicer join that keeps the consistent prediction style.
        prediction = "\n".join([tool_call.function.arguments for tool_call in response.choices[0].message.tool_calls])
    else:
        prediction = response.choices[0].message.content
    return prediction


def predict_anthropic(client, model_name, document, schema, examples, function_calling_tools = None):
    extra_args = {}
    if function_calling_tools:
        extra_args["tool_choice"] = "any"
        extra_args["tools"] = function_calling_tools
    messages = [
        {
            "role": "user",
            "content": f"You will be provided with unstructured data in the form of a document. Your task is to create a JSON object that adheres to the following schema:\n{schema}\n{examples}\nReturn only the result JSON. If data for a given field is not present, provide an empty array ([]) for arrays, an empty string for strings, and null for missing values.\nInput Document:\n" + document,
        }
    ]
    out = client.messages.create(
        model=model_name,
        messages=messages,
        max_tokens=2048,
        temperature=0.1,
        top_p=1,
        **extra_args,
    )
    if function_calling_tools:
        # Try to grab the last entry, which should be tool use:
        prediction = out.content[-1].input
        # Could do this.  Might be the better way:
        #for c in out.content:
        #    if c.type == 'tool_use':
        #        prediction = c.input
    else:
        prediction = out.content[0].text
    time.sleep(0.1)  # The best rate-limiting.
    return prediction


def predict_hf_with_instruct_pipe(provider: str, model_name: str, document, schema, examples, constrain_decoding=None):
    pipe = load_cached_pipe(f"{provider}/{model_name}")  # "mistralai/mistral-7b-instruct-v0.3"
    g = None
    if constrain_decoding:
        g = Guard.from_pydantic(constrain_decoding, output_formatter='jsonformer')
    messages = [
        {
            "role": "system",
            "content": f"You will be provided with unstructured data in the form of a document. Your task is to create a JSON object that adheres to the following schema:\n{schema}\n{examples}\nReturn only the result JSON. If data for a given field is not present, provide an empty array ([]) for arrays, an empty string for strings, and null for missing values."
        },
        {
            "role": "user",
            "content": document
        }
    ]
    if g:
        output = g(pipe, prompt=messages[0]["content"] + "\nInput Document:\n" + messages[1]["content"]).raw_llm_output
        # output = g(pipe, messages).raw_llm_output
    else:
        output = pipe(messages)
    assert output[0]["generated_text"][-1]["role"] == "assistant"
    return output[0]["generated_text"][-1]["content"]


def predict_hf_with_system(provider: str, model_name: str, document, schema, examples):
    model, tokenizer = load_cached_active_model(f"{provider}/{model_name}")
    messages = [
        {
            "role": "system",
            "content": f"You will be provided with unstructured data in the form of a document. Your task is to create a JSON object that adheres to the following schema:\n{schema}\n{examples}\nReturn only the result JSON. If data for a given field is not present, provide an empty array ([]) for arrays, an empty string for strings, and null for missing values."
        },
        {
            "role": "user",
            "content": document
        }
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=4096)
    text = tokenizer.decode(outputs[0], skip_special_tokens=False)  # We want the 'assistant' special token.
    text = text.split("<|assistant|>")[-1].split("<|end|>")[0]
    return text


def run():
    # Set up models and clients:
    print("Setting up models and clients:")

    anthropic_client = anthropic.Anthropic()
    openai_client = OpenAI()

    print("Getting data ready.")
    db = sqlite3.connect("data.db")
    db.row_factory = sqlite3.Row
    #db.execute("CREATE TABLE IF NOT EXISTS model_outputs (id INTEGER PRIMARY KEY AUTOINCREMENT, model_id INTEGER NOT NULL, task_id INTEGER NOT NULL, document_id INTEGER NOT NULL, num_samples INTEGER NOT NULL, time REAL NOT NULL, guardrails TEXT NOT NULL, raw_output TEXT NOT NULL, exception TEXT NOT NULL);")

    print("Running...")
    #for task in (Task.NER_FLAT, ):
    for task in Task:
        task_str = str(task)[len("Task:"):]
        task_id = db.execute("SELECT id FROM tasks WHERE task = ?;", (task_str.lower(),)).fetchone()["id"]
        for model_provider, model_name in (
                #("numind", "nuextract-tiny"),
                #("numind", "nuextract"),
                #("numind", "nuextract-large"),
                #("openai", "gpt-3.5-turbo"),
                #("openai", "gpt-4-turbo"),
                #("openai", "gpt-4o-mini"),
                #("openai", "gpt-4"),
                #("anthropic", "claude-3-opus-20240229"),
                #("anthropic", "claude-3-5-sonnet-20240620"),
                #("microsoft", "phi-3-mini-4k-instruct"),
                #("meta-llama", "meta-llama-3.1-8b"),
                #("meta-llama", "meta-llama-3.1-8b-instruct"),
                ("mistral", "mistral-7b-instruct-v0.3"),
        ):
            model_id = db.execute("SELECT id FROM models WHERE name = ?;", (model_name,)).fetchone()["id"]
            for num_examples in (0, 1, 3):
                data_cursor = db.execute("SELECT id, text FROM documents;")
                #for datasource in ("wikipedia", "news"):
                #    for doc_idx, doc_text in enumerate(data[datasource]):
                for row in data_cursor:
                    doc_id = row["id"]
                    doc_text = row["text"]
                    print(f"Doc ID: {doc_id} - Task ID: {task_id} - Model: {model_name} - #Ex: {num_examples}")
                    start_time = time.time()  # Will probably be overwritten, but in the event of an exception...
                    prediction = ""
                    ex = ""
                    try:
                        # Prep prompt and schema:
                        schema = SCHEMAS[task]
                        if model_provider == "numind":
                            schema = NUEXTRACT_SCHEMAS[task]
                        examples = EXAMPLES[task]
                        maybe_examples = ""
                        if num_examples > 0:
                            maybe_examples = f"The following are examples of the schema:\n"
                            for x in range(num_examples):
                                maybe_examples += examples[x] + "\n\n"

                        # Run:
                        start_time = time.time()
                        if model_provider == "openai":
                            prediction = predict_openai(openai_client, model_name, doc_text, schema, maybe_examples)
                        elif model_provider == "anthropic":
                            prediction = predict_anthropic(anthropic_client, model_name, doc_text, schema, maybe_examples)
                        elif model_provider == "numind":
                            prediction = predict_nuextract(model_name, doc_text, schema, examples[:num_examples])
                        elif model_provider == "microsoft":
                            prediction = predict_hf_with_system(model_provider, model_name, doc_text, schema, maybe_examples)
                        else:
                            # Pass DATA_MODELS as constrained_decoding value.
                            prediction = predict_hf_with_instruct_pipe(model_provider, model_name, doc_text, schema, maybe_examples)
                    except Exception as e:
                        ex = str(e)
                    end_time = time.time()

                    db.execute("INSERT INTO model_outputs (model_id, document_id, task_id, num_samples, time, guardrails, raw_output, exception) VALUES (?, ?, ?, ?, ?, ?, ?, ?);", (
                        model_id,
                        doc_id,
                        task_id,
                        num_examples,
                        end_time - start_time,
                        "prompt_engineering",  # "json_mode", "function_calling", "constrained_decoding"
                        prediction,
                        ex
                    ))
                    db.commit()

if __name__ == "__main__":
    run()
