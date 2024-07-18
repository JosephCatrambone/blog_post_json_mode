
import json
import os
import sqlite3
#%pip install partial-json-parser
from dataclasses import dataclass

import partial_json_parser
from sentence_transformers import SentenceTransformer

from task import Task, DATA_MODELS, NERMultiExtraction, NERNestedExtraction, NestedEventExtraction, NestedThingExtraction, NestedUnitExtraction


model = SentenceTransformer("all-mpnet-base-v2")


@dataclass
class Score:
    well_formed_json: bool = False
    schema_matches_no_extras: bool = False  # Has at least the required fields but may have extra imagined fields. match_exact implies this.
    schema_matches_exact: bool = False  # Has all required fields and no extra imagined fields. decode_success implies this.
    schema_decode_success: bool = False  # All fields are present and none violate constraints
    #content_extracted: bool = False  # Contain the desired data and no hallucinated content
    #no_duplicate_content: bool = False  # Contains only desired valid data. Not multiple copies of it
    #no_hallucinated_content: bool = False  # Contains the desired valid data but has more than 50% other unrelated data (badly hallucinated content)
    eval_score: float = 0.0  # The intersection over union of the data with ground truth.


def run_evaluations():
    db = sqlite3.connect("data.db")
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE "evaluation" (
        "id" INTEGER PRIMARY KEY AUTOINCREMENT,
        "model_output_id"	INTEGER NOT NULL,
        "valid_json"	INTEGER NOT NULL,
        "schema_match_no_extras"	INTEGER NOT NULL,
        "schema_match_exact"	INTEGER NOT NULL,
        "schema_decode_success"	INTEGER NOT NULL,
        "eval_score"	REAL NOT NULL
    );""")
    cursor = db.execute("""
    SELECT 
        model_outputs.id AS model_output_id, 
        tasks.task AS task_name, 
        model_outputs.raw_output AS model_out, 
        ground_truth.json AS gt_text 
    FROM model_outputs 
    JOIN tasks ON model_outputs.task_id = tasks.id 
    JOIN ground_truth ON (
        ground_truth.document_id = model_outputs.document_id AND 
        ground_truth.task_id = model_outputs.task_id
    );
    """)
    for r in cursor:
        task_name = r["task_name"]
        task = Task(task_name)
        output_id = r["model_output_id"]
        raw_model_output = r["model_out"]
        ground_truth = json.loads(r["gt_text"])

        s = Score()

        # Just try to load the thing, falling back to partial loading.
        maybe_data = dict()
        try:
            maybe_data = json.loads(raw_model_output)
            s.well_formed_json = True
        except json.JSONDecodeError:
            s.well_formed_json = False

        # Not well formed, but parsable:
        if not s.well_formed_json:
            try:
                maybe_data = partial_json_parser.loads(raw_model_output)
            except Exception:  # partial_json_parser.JSONDecodeError:
                maybe_data = dict()

        # Diff the keys to see if we have extras:
        # TODO: Use the BFCL list/dict tree comparison.
        if dict_has_all_fields(DATA_MODELS[task].model_fields, maybe_data):
            s.schema_matches_min = True
        if dict_has_no_extra_fields(DATA_MODELS[task].model_fields, maybe_data):
            s.schema_matches_no_extras = True

        # Maybe try to do a parse with the data model we have defined:
        try:
            #DATA_MODELS[task].__pydantic_model__.parse_raw()
            #if isinstance(maybe_data, dict):
            #    parsed_data = DATA_MODELS[task](**maybe_data)
            #elif isinstance(maybe_data, list):
            #    parsed_data = DATA_MODELS[task](maybe_data)
            parsed_data = DATA_MODELS[task](**maybe_data)
            s.schema_matches_exact = True
        except Exception:
            # Missing some key fields.
            parsed_data = None

        # Check all values.
        # The first qualitative measure.
        if task in TASK_TO_EVAL_FN and parsed_data is not None:
            try:
                TASK_TO_EVAL_FN[task](DATA_MODELS[task](**ground_truth), parsed_data, s)
            except Exception as e:
                print(e)
                continue

        db.execute(
            """INSERT INTO evaluation ( 
                model_output_id,
                valid_json,
                schema_match_no_extras,
                schema_match_exact,
                schema_decode_success,
                eval_score,
            ) VALUES (?, ?, ?, ?, ?, ?);""", (
                output_id,
                s.well_formed_json,
                s.schema_matches_no_extras,
                s.schema_matches_exact,
                s.schema_decode_success,
                s.eval_score
            )
        )
    db.commit()
    db.close()


def eval_ner_flat(ground_truth: NERMultiExtraction, parsed_data: NERMultiExtraction, score_ref: Score):
    # TODO: This is perhaps slightly lazy.
    """
    embeddings = model.encode(["San Francisco", "California", "San Francisco, CA", "San Francisco, California", "SF", "Maine", "New England", "Location"])
    similarities = model.similarity(model.encode(["San Francisco", ]), embeddings)
    >>> tensor([[1.0000, 0.6371, 0.8915, 0.9057, 0.7240, 0.4704, 0.6216, 0.6042]])
    """

    # Dump all the elements together because it turns out that NER is really noisy:
    gt_elements = set()
    pred_elements = set()
    for attribute in ('names', 'locations', 'organizations', 'misc'):
        gt_elements |= set(ground_truth.__getattribute__(attribute))
        pred_elements |= set(parsed_data.__getattribute__(attribute))
    gt_elements = list(gt_elements)
    pred_elements = list(pred_elements)

    if len(gt_elements) == 0:
        # When there are no elements, predicting the empty set is a match, otherwise we halve the score for every element.
        score = 1.0 / (1.0 + len(pred_elements)**2)
    else:
        FUZZY_MATCH = False
        BINARIZE = False
        THRESHOLD = 0.7
        gt_entry_embeddings = model.encode(gt_elements)
        model_entry_embeddings = model.encode(pred_elements)
        similarity = model.similarity(gt_entry_embeddings, model_entry_embeddings)
        if FUZZY_MATCH:  # Just sum the max matches.
            score = similarity.max(dim=1).values.sum().item() / float(max(similarity.shape))
        elif BINARIZE:  # If the max match is above a threshold, treat it like a match.
            score = ((similarity.max(dim=1).values > THRESHOLD) * 1.0).sum().item() / float(max(similarity.shape))
        else:  # If the max match is above a threshold, add it raw rather than snap to 1.0.
            score = ((similarity.max(dim=1).values > THRESHOLD) * similarity.max(dim=1).values).sum().item() / float(max(similarity.shape))
        # If the ground truth and model were perfectly aligned we would have a square matrix with ones along the diagonal.

    if score_ref is not None:
        score_ref.eval_score = score

    """
    gt_pile_of_names = set()
    gt_pile_of_names |= set(ground_truth.names)
    gt_pile_of_names |= set(ground_truth.locations)
    gt_pile_of_names |= set(ground_truth.organizations)
    gt_pile_of_names |= set(ground_truth.misc)

    parsed_pile_of_names = set()
    parsed_pile_of_names |= set(parsed_data.names)
    parsed_pile_of_names |= set(parsed_data.locations)
    parsed_pile_of_names |= set(parsed_data.organizations)
    parsed_pile_of_names |= set(parsed_data.misc)

    # IoU evaluation.
    intersection_size = float(len(parsed_pile_of_names.intersection(gt_pile_of_names)))
    union_size = float(len(parsed_pile_of_names.union(gt_pile_of_names)))
    if union_size < 1e-6:
        breakpoint()
        raise Exception("Ground truth empty!?")
    else:
        score_ref.eval_score = intersection_size/union_size
    """

    return score


TASK_TO_EVAL_FN = {
    Task.NER_FLAT: eval_ner_flat,
}


def dict_has_all_fields(pydantic_fields: dict, sample: dict, max_depth: int = 0) -> bool:
    """Returns true if the sample object has at all the required fields for itself and any objects that are children.
    If a child object of the required fields is an object and the sample object has null or an empty array, we do not count this as missing.
    If the sample object has extra fields, we do not count these as problematic."""
    for k in pydantic_fields.keys():
        if k not in sample:
            return False
    """
    if max_depth > 0:
        for k, required_subfields in pydantic_fields.items():
            if sample[k] is not None:
                if isinstance(required_subfields, dict):
                    if not dict_has_all_fields(required_subfields, sample[k], max_depth-1):
                        return False
    """
    return True


def dict_has_no_extra_fields(minimal_fields: dict, sample: dict, max_depth: int = 0) -> bool:
    """Returns true if the given sample has only the fields present in minimal_fields."""
    for k in sample.keys():
        if k not in minimal_fields:
            return False
    """
    if recursive:
        for k, v in minimal_fields.items():
            if isinstance(v, dict):
                if not isinstance(minimal_fields[k], dict):
                    return False
                elif not dict_has_no_extra_fields(minimal_fields[k], v, recursive):
                    return False
    """
    return True


def list_has_all_items_unordered(minimal_entries: list, sample: list):
    return set(minimal_entries).issubset(set(sample))


if __name__ == "__main__":
    run_evaluations()
