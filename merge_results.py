import sqlite3
import sys


def merge_results(read_from_db, write_to_db):
    """
    Read a bunch of results from the model_outputs table and, if they don't exist in the 'write_to_db', write them.
    Uses uniqueness of model_id, guardrails, task_id, and document_id fields to determine if an insert should happen.
    This not a smart algorithm.
    """
    if isinstance(read_from_db, str):
        read_from_db = sqlite3.connect(read_from_db)
    if isinstance(write_to_db, str):
        write_to_db = sqlite3.connect(write_to_db)

    read_from_db.row_factory = sqlite3.Row
    write_to_db.row_factory = sqlite3.Row

    # Lazy and stupid?  Relatable!
    cursor = read_from_db.execute("SELECT * FROM model_outputs;")
    for row in cursor:
        #model_id, document_id, task_id, num_samples, time, guardrails, raw_output, exception
        model_id = row['model_id']
        document_id = row['document_id']
        task_id = row['task_id']
        num_samples = row['num_samples']
        guardrails = row['guardrails']
        raw_output = row['raw_output']
        exception = row['exception']
        time = row['time']
        exists = write_to_db.execute(
            "SELECT 1 FROM model_outputs WHERE model_id = ? AND document_id = ? AND task_id = ? AND num_samples = ? AND guardrails = ? LIMIT 1;",
            (model_id, document_id, task_id, num_samples, guardrails)
        )
        if exists.fetchone() is None:
            res = write_to_db.execute(
                "INSERT INTO model_outputs (model_id, document_id, task_id, num_samples, guardrails, time, raw_output, exception) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (model_id, document_id, task_id, num_samples, guardrails, time, raw_output, exception)
            )
            print(f"Inserting: ModelID: {model_id}  DocID: {document_id}  TaskID: {task_id}  NumSamples: {num_samples}  Guard: {guardrails} -- New Row ID: {res.lastrowid}")
    write_to_db.commit()


if __name__ == "__main__":
    merge_results(sys.argv[1], sys.argv[2])
