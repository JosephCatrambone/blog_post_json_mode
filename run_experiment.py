# Perform the actual experiment.
import json
import sqlite3
import time
from enum import Enum

import anthropic
import pandas as pd
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from task import Task, SCHEMAS, NUEXTRACT_SCHEMAS, EXAMPLES
from dataset import get_data
#%pip install openai
#%pip install anthropic




device = "cuda"


def make_df():
    return pd.DataFrame(columns=["doc_idx", "task", "model", "dataset", "guardrails", "num_shots", "time", "raw_model_output", "exception"])


def append(df, new_data: dict):
    columns = list(df.columns)
    new_row = pd.Series([new_data[k] for k in columns], index=df.columns)
    df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    return df


def append_run(df, doc_idx, task, model, dataset, guardrails, num_shots, time, raw_model_output, exc):
    return append(df, {"doc_idx": doc_idx, "task": task, "model": model, "dataset": dataset, "guardrails": guardrails, "num_shots": num_shots, "time": time, "raw_model_output": raw_model_output, "exception": exc})


# Set up the inference for the numind nuextract case:
# %pip install torch
# %pip install transformers

# Taken directly from the NuMind NuExtract documentation
# https://huggingface.co/numind/NuExtract-tiny
def predict_NuExtract(model, tokenizer, text, schema, examples=["", "", ""], device="cuda"):
    schema = json.dumps(json.loads(schema), indent=4)
    input_llm = "<|input|>\n### Template:\n" + schema + "\n"
    for i in examples:
        if i != "":
            input_llm += "### Example:\n" + json.dumps(json.loads(i), indent=4) + "\n"

    input_llm += "### Text:\n" + text + "\n<|output|>\n"
    input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True,
                          max_length=4000).to(device)

    output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)
    return output.split("<|output|>")[1].split("<|end-output|>")[0]


def predict_openai(client, model_name, document, schema, examples):
    prediction = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": f"You will be provided with unstructured data in the form of a document. Your task is to create a JSON object that adheres to the following schema:\n{schema}\n{examples}\nReturn only the result JSON. If data for a given field is not present, provide an empty array ([]) for arrays, an empty string for strings, and null for missing values."
            },
            {
                "role": "user",
                "content": document
            }
        ],
        temperature=0.1,
        max_tokens=1024,
        top_p=1
    ).choices[0].message.content
    return prediction


def predict_anthropic(client, model_name, document, schema, examples):
    prediction = client.messages.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": f"You will be provided with unstructured data in the form of a document. Your task is to create a JSON object that adheres to the following schema:\n{schema}\n{maybe_examples}\nReturn only the result JSON. If data for a given field is not present, provide an empty array ([]) for arrays, an empty string for strings, and null for missing values.\nInput Document:\n" + doc_text,
            }
        ],
        max_tokens=1024,
        temperature=0.1,
        top_p=1
    ).content[0].text
    time.sleep(1)  # The best rate-limiting.
    return prediction


def run():
    # Set up models and clients:
    print("Setting up models and clients:")
    nuextract_model = AutoModelForCausalLM.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True)
    nuextract_model.to("cuda")
    nuextract_model.eval()
    anthropic_client = anthropic.Anthropic()
    openai_client = OpenAI()

    print("Getting data ready.")
    db = sqlite3.connect("data.db")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE IF NOT EXISTS model_outputs (id INTEGER PRIMARY KEY AUTO INCREMENT, model_id INTEGER NOT NULL, task_id INTEGER NOT NULL, document_id INTEGER NOT NULL, num_samples INTEGER NOT NULL, time REAL NOT NULL, guardrails TEXT NOT NULL, raw_output TEXT NOT NULL, exception TEXT NOT NULL);")

    print("Running...")
    #for task in (Task.NER_FLAT, ):
    for task in Task:
        task_id = db.execute("SELECT id FROM tasks WHERE task = ?;", (str(task)[len("Task:"):],)).fetchone()["id"]
        for model_provider, model_name in (
                #("numind", "nuextract"),
                ("openai", "gpt-3.5-turbo"),
                ("openai", "gpt-4-turbo"),
                ("anthropic", "claude-3-opus-20240229")
        ):
            model_id = db.execute("SELECT id FROM models WHERE name = ?;", (model_name,)).fetchone()["id"]
            for num_examples in (0, 1, 3):
                data_cursor = db.execute("SELECT id, text FROM documents WHERE source = 'golden';")
                #for datasource in ("wikipedia", "news"):
                #    for doc_idx, doc_text in enumerate(data[datasource]):
                for row in data_cursor:
                    doc_id = row["id"]
                    doc_text = row["text"]
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
                            prediction = predict_NuExtract(nuextract_model, tokenizer, doc_text, schema, examples[:num_examples])
                        else:
                            print(f"EXCEPTION: Typo in model_provider: {model_provider}")
                    except Exception as e:
                        ex = str(e)
                    end_time = time.time()

                    db.execute("INSERT INTO model_outputs (model_id, document_id, task_id, num_samples, time, guardrails, raw_output, exception) VALUES (?, ?, ?, ?, ?, ?, ?, ?);", (
                        model_id,
                        doc_id,
                        task_id,
                        num_examples,
                        end_time - start_time,
                        "",
                        prediction,
                        ex
                    ))
