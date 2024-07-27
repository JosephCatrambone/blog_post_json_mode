import sqlite3
import sys

import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def plot_all(db):
    q = """
    SELECT
        models.name AS mname, model_outputs.num_samples AS num_samples, AVG(evaluation.valid_json) AS valid_json, AVG(schema_match_minimal) AS minimal_schema_match, AVG(schema_decode_success) AS schema_decode_success, AVG(evaluation.eval_score) AS eval_score
    FROM evaluation
    JOIN model_outputs ON evaluation.model_output_id = model_outputs.id
    JOIN tasks ON tasks.id = model_outputs.task_id
    JOIN models ON models.id = model_outputs.model_id
    JOIN documents ON documents.id = model_outputs.document_id
    WHERE model_outputs.guardrails = ""
    --WHERE documents.source = "golden" AND tasks.task = "ner_flat" --AND document_id = 30
    GROUP BY models.name, model_outputs.num_samples;
    """
    data = pd.read_sql_query(q, db)
    #penguins = sns.load_dataset("penguins")

    # Draw a nested barplot by species and sex
    for y in ("valid_json", "minimal_schema_match", "schema_decode_success", "eval_score"):
        g = sns.catplot(
            data=data, kind="bar", x="mname", y=y, hue="num_samples",
            errorbar="sd", palette="dark", alpha=.6, height=5,
        )
        g.figure.set_size_inches(10, 5)
        g.despine(left=True)
        g.set_axis_labels("", y.replace("_", " ").capitalize())
        g.legend.set_title("Num Examples")
        g.savefig(f"plot_{y}.png")


def main():
    db = sqlite3.connect("data.db")
    db.row_factory = sqlite3.Row
    plot_all(db)


if __name__ == "__main__":
    main()
