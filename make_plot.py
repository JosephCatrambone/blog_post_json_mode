import sqlite3
import sys

import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def plot_all(db):
    q = """
    SELECT
        models.display_name AS mname, model_outputs.num_samples AS num_samples, AVG(evaluation.valid_json) AS valid_json, AVG(schema_match_minimal) AS minimal_schema_match, AVG(schema_decode_success) AS schema_decode_success, AVG(evaluation.eval_score) AS eval_score
    FROM evaluation
    JOIN model_outputs ON evaluation.model_output_id = model_outputs.id
    JOIN tasks ON tasks.id = model_outputs.task_id
    JOIN models ON models.id = model_outputs.model_id
    JOIN documents ON documents.id = model_outputs.document_id
    WHERE model_outputs.guardrails = "prompt_engineering"
    --WHERE documents.source = "golden" AND tasks.task = "ner_flat" --AND document_id = 30
    GROUP BY models.name, model_outputs.num_samples;
    """
    data = pd.read_sql_query(q, db)
    #penguins = sns.load_dataset("penguins")

    # Draw a nested barplot by species and sex
    for y in ("valid_json", "minimal_schema_match", "schema_decode_success", "eval_score"):
        g = sns.catplot(
            data=data.sort_values(by="mname"),
            kind="bar", x="mname", y=y, hue="num_samples",
            errorbar="sd", palette="dark", alpha=.6, height=5,
            #order=['place the desired order here']
        )
        #g.set(ylim=(0, 1))
        g.figure.set_size_inches(25, 5)
        g.despine(left=True)
        g.set_axis_labels("", y.replace("_", " ").upper())
        g.legend.set_title("Num Examples")
        g.savefig(f"plot_{y}.png")


def compare_function_calls(db):
    for metric in ("valid_json", "schema_match_minimal", "schema_decode_success", "eval_score"):
        q = f"""
        SELECT
            models.display_name AS mname, model_outputs.guardrails AS guardrails, AVG(evaluation.{metric}) AS score
        FROM models
        JOIN model_outputs ON model_outputs.model_id = models.id
        JOIN evaluation ON model_outputs.id = evaluation.model_output_id
        WHERE models.id IN (SELECT id FROM models WHERE (name LIKE "gpt%" or name LIKE "claude%"))
        GROUP BY models.name, model_outputs.guardrails;
        """
        data = pd.read_sql_query(q, db)
        #penguins = sns.load_dataset("penguins")

        # Draw a nested barplot by species and sex
        g = sns.catplot(
            data=data.sort_values(by="mname"),
            kind="bar", x="mname", y="score", hue="guardrails",
            errorbar="sd", palette="dark", alpha=.6, height=5,
            #order=['place the desired order here']
        )
        #g.set(ylim=(0, 1))
        g.figure.set_size_inches(15, 5)
        g.despine(left=True)
        g.set_axis_labels("", metric.replace("_", " ").upper())
        g.legend.set_title("Generation Method")
        g.savefig(f"plot_openai_{metric}.png")


def compare_constrained_decoding(db):
    for metric in ("valid_json", "schema_match_minimal", "schema_decode_success", "eval_score"):
        q = f"""
        SELECT
            models.display_name AS mname, model_outputs.guardrails AS guardrails, AVG(evaluation.{metric}) AS score
        FROM models
        JOIN model_outputs ON model_outputs.model_id = models.id
        JOIN evaluation ON model_outputs.id = evaluation.model_output_id
        WHERE models.id IN (SELECT id FROM models WHERE (name LIKE "meta-llama%" or name LIKE "mistral%"))
        GROUP BY models.name, model_outputs.guardrails;
        """
        data = pd.read_sql_query(q, db)
        #penguins = sns.load_dataset("penguins")

        # Draw a nested barplot by species and sex
        g = sns.catplot(
            data=data.sort_values(by="mname"),
            kind="bar", x="mname", y="score", hue="guardrails",
            errorbar="sd", palette="dark", alpha=.6, height=5,
            #order=['place the desired order here']
        )
        if metric != "eval_score":
            g.set(ylim=(0, 1))
        g.figure.set_size_inches(10, 5)
        g.despine(left=True)
        g.set_axis_labels("", metric.replace("_", " ").upper())
        g.legend.set_title("Generation Method")
        g.savefig(f"plot_cd_{metric}.png")


def compare_latencies(db):
    q = f"""
            SELECT
                models.display_name AS mname, model_outputs.guardrails, model_outputs.time AS latency
            FROM models
            JOIN model_outputs ON model_outputs.model_id = models.id
            JOIN evaluation ON model_outputs.id = evaluation.model_output_id
            WHERE model_outputs.num_samples = 0 AND (models.name LIKE "gpt%" or models.name LIKE "claude%") AND model_outputs.exception = "";
    """
    data = pd.read_sql_query(q, db)
    # penguins = sns.load_dataset("penguins")

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=data.sort_values(by="mname"),
        kind="bar", x="mname", y="latency", hue="guardrails",
        errorbar=("sd", 0.5), palette="dark", alpha=.6, height=5,
        # order=['place the desired order here']
    )
    g.figure.set_size_inches(15, 5)
    g.despine(left=True)
    g.set_axis_labels("", "Latency (seconds)")
    g.legend.set_title("Generation Method")
    g.savefig(f"plot_latency.png")


def main():
    db = sqlite3.connect("data.db")
    db.row_factory = sqlite3.Row
    #plot_all(db)
    #compare_function_calls(db)
    #compare_constrained_decoding(db)
    compare_latencies(db)


if __name__ == "__main__":
    main()
