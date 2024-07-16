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
    data = get_data()

    print("Running...")
    results = make_df()
    checkpoint_count = 0
    for task in Task:
        for model_provider, model_name in (
                ("numind", "nuextract"),
                ("openai", "gpt-3.5-turbo"),
                ("openai", "gpt-4-turbo"),
                ("anthropic", "claude-3-opus-20240229")
        ):
            for datasource in ("wikipedia", "news"):
                for num_examples in (0, 1, 3):
                    for doc_idx, doc_text in enumerate(data[datasource]):
                        print(
                            f"Run: {model_name} - {datasource} - {num_examples} - {doc_idx} ",
                            end="")
                        # Check to see if we already ran this data.  If we did, we can skip it.
                        maybe_row = results.loc[results.doc_idx == doc_idx][
                            results.num_shots == num_examples][results.task == task][
                            results.model == model_name][results.dataset == datasource]
                        row_data = maybe_row.to_dict(orient='records')
                        existing_row_idx = None
                        if row_data:
                            existing_row_idx = maybe_row.index[0]
                            # unfilled_results.loc[existing_row_idx, "exception"] = ""
                            if row_data[0]["exception"] == "" and row_data[0]["raw_model_output"] != "":
                                print("- SKIPPED!")
                                continue
                            else:
                                print("- RERUNNING")
                        else:
                            print("...")
                        # We need to run this.
                        start_time = time.time()  # Will probably be overwritten, but in the event of an exception...
                        prediction = ""
                        ex = ""
                        try:
                            schema = SCHEMAS[task]
                            if model_provider == "numind":
                                schema = NUEXTRACT_SCHEMAS[task]
                            examples = EXAMPLES[task]
                            maybe_examples = ""
                            if num_examples > 0:
                                maybe_examples = f"The following are examples of the schema:\n"
                                for x in range(num_examples):
                                    maybe_examples += examples[x] + "\n\n"
                            start_time = time.time()
                            if model_provider == "openai":
                                prediction = openai_client.chat.completions.create(
                                    model=model_name,
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": f"You will be provided with unstructured data in the form of a document. Your task is to create a JSON object that adheres to the following schema:\n{schema}\n{maybe_examples}\nReturn only the result JSON. If data for a given field is not present, provide an empty array ([]) for arrays, an empty string for strings, and null for missing values."
                                        },
                                        {
                                            "role": "user",
                                            "content": doc_text
                                        }
                                    ],
                                    temperature=0.1,
                                    max_tokens=1024,
                                    top_p=1
                                ).choices[0].message.content
                            elif model_provider == "anthropic":
                                prediction = anthropic_client.messages.create(
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
                            elif model_provider == "numind":
                                prediction = predict_NuExtract(nuextract_model, tokenizer, doc_text, schema, examples[:num_examples])
                            else:
                                print(f"EXCEPTION: Typo in model_provider: {model_provider}")
                        except Exception as e:
                            ex = str(e)
                        end_time = time.time()
                        if existing_row_idx is None:
                            results = append_run(
                                results,
                                doc_idx,
                                task,
                                model_name,
                                datasource,
                                "None",
                                num_examples,
                                end_time - start_time,
                                prediction,
                                ex
                            )
                        else:
                            results.loc[existing_row_idx, "doc_idx"] = doc_idx
                            results.loc[existing_row_idx, "task"] = task
                            results.loc[existing_row_idx, "num_shots"] = num_examples
                            results.loc[existing_row_idx, "model"] = model_name
                            results.loc[existing_row_idx, "dataset"] = datasource
                            results.loc[existing_row_idx, "time"] = end_time - start_time
                            results.loc[existing_row_idx, "raw_model_output"] = prediction
                            results.loc[existing_row_idx, "exception"] = ex

                    results.to_json(f'checkpoint_{checkpoint_count}.json')
                    checkpoint_count += 1

    # Generate multiple output formats for our consumption and viewing pleasure.
    results.to_csv("model_outputs.csv")
    results.to_json("model_outputs.json")
    connection = sqlite3.connect("model_outputs.db")
    results.to_db("model_outputs", connection)
