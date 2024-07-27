
import json
import os
import sqlite3
#%pip install partial-json-parser
from dataclasses import dataclass

import partial_json_parser
from sentence_transformers import SentenceTransformer

from task import Task, DATA_MODELS, NERMultiExtraction, NERNestedExtraction, \
    NestedEventExtraction, NestedThingExtraction, NestedUnitExtraction, UnitExtraction

model = SentenceTransformer("all-mpnet-base-v2")


@dataclass
class Score:
    well_formed_json: bool = False
    schema_matches_minimal: bool = False  # Has at least the required fields but may have extra imagined fields. decode_success implies this.
    schema_decode_success: bool = False  # All fields are present and none violate constraints
    #content_extracted: bool = False  # Contain the desired data and no hallucinated content
    #no_duplicate_content: bool = False  # Contains only desired valid data. Not multiple copies of it
    #no_hallucinated_content: bool = False  # Contains the desired valid data but has more than 50% other unrelated data (badly hallucinated content)
    eval_score: float = 0.0  # The intersection over union of the data with ground truth.


def run_evaluations():
    db = sqlite3.connect("data.db")
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS "evaluation" (
        "id" INTEGER PRIMARY KEY AUTOINCREMENT,
        "model_output_id"	INTEGER NOT NULL,
        "valid_json"	INTEGER NOT NULL,
        "schema_match_minimal"	INTEGER NOT NULL,
        "schema_decode_success"	INTEGER NOT NULL,
        "eval_score"	REAL NOT NULL
    );""")
    db.execute("DELETE FROM evaluation;")
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
                # Special pleading:
                # Often times we'll finish an object with a partial model.
                # If we can recover by pulling out completely empty dicts, it may help.
            except Exception:  # partial_json_parser.JSONDecodeError:
                maybe_data = dict()

        # Diff the keys to see if we have extras:
        # TODO: Use the BFCL list/dict tree comparison.
        # DATA_MODELS[Task.EVENT_EXTRACTION].__fields__
        if dict_has_all_fields(DATA_MODELS[task].model_fields, maybe_data):
            s.schema_matches_minimal = True

        # Maybe try to do a parse with the data model we have defined:
        try:
            #DATA_MODELS[task].__pydantic_model__.parse_raw()
            parsed_data = DATA_MODELS[task](**maybe_data)
            s.schema_matches_minimal = True
            s.schema_decode_success = True
        except Exception:
            # Missing some key fields.
            parsed_data = None

        # Check all values.
        # The first qualitative measure.
        if task in TASK_TO_EVAL_FN and parsed_data is not None:
            TASK_TO_EVAL_FN[task](DATA_MODELS[task](**ground_truth), parsed_data, s)

        print(f"{output_id} - {task_name} - {s}")
        db.execute(
            """INSERT INTO evaluation ( 
                model_output_id,
                valid_json,
                schema_match_minimal,
                schema_decode_success,
                eval_score
            ) VALUES (?, ?, ?, ?, ?);""", (
                output_id,
                s.well_formed_json,
                s.schema_matches_minimal,
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

    score = fuzzy_diff_sets(gt_elements, pred_elements)

    if score_ref is not None:
        score_ref.eval_score = score

    return score


def eval_ner_nested(ground_truth: NERNestedExtraction, parsed_data: NERNestedExtraction, score_ref: Score):
    gt_entities = set()
    pred_entities = set()
    for ent in ground_truth.named_entities:
        gt_entities.add(ent.text)
    for ent in parsed_data.named_entities:
        pred_entities.add(ent.text)
    score = fuzzy_diff_sets(gt_entities, pred_entities, threshold=0.7)
    if score_ref is not None:
        score_ref.eval_score = score
    return score


def eval_thing_extraction(ground_truth: NestedThingExtraction, parsed_data: NestedThingExtraction, score_ref: Score):
    gt_entities = set()
    pred_entities = set()
    for ent in ground_truth.things:
        gt_entities.add(ent.text)
    for ent in parsed_data.things:
        pred_entities.add(ent.text)
    score = fuzzy_diff_sets(list(gt_entities), list(pred_entities), threshold=0.7)
    if score_ref is not None:
        score_ref.eval_score = score
    return score


def eval_unit_extraction(ground_truth: NestedUnitExtraction, parsed_data: NestedUnitExtraction, score_ref: Score):
    match_score = 0
    gt_item_count = 0
    parsed_data_item_count = 0
    for gt_item in ground_truth.items:
        gt_item_count += 1
        found = False
        quantity_match = False
        unit_match = False
        parsed_data_item_count = 0
        for pred_item in parsed_data.items:
            parsed_data_item_count += 1
            if pred_item.item == gt_item.item:
                found = True
                if gt_item.quantity:
                    if pred_item.quantity:
                        if abs(float(gt_item.quantity) - float(pred_item.quantity)) < 0.1:
                            quantity_match = True
                else:
                    if not pred_item.quantity:
                        quantity_match = True  # Both are missing.
                if gt_item.unit == pred_item.unit:
                    unit_match = True
        if found:
            match_score += 0.5
        if quantity_match:
            match_score += 0.25
        if unit_match:
            match_score += 0.25
    if gt_item_count == 0 and parsed_data_item_count == 0:
        return 1.0
    match_score /= max(gt_item_count, parsed_data_item_count)
    if score_ref is not None:
        score_ref.eval_score = match_score
    return match_score


def fuzzy_diff_sets(gt_elements, pred_elements, threshold: float = 0.0, binarize: bool = False) -> float:
    if len(gt_elements) == 0 or len(pred_elements) == 0:
        if len(gt_elements) == len(pred_elements):
            score = 1.0
        else:
            score = 0.0
        # When there are no elements, predicting the empty set is a match, otherwise we halve the score for every element.
        #score = 1.0 / (1.0 + (len(gt_elements) + len(pred_elements))**2)
    else:
        gt_entry_embeddings = model.encode(list(gt_elements))
        model_entry_embeddings = model.encode(list(pred_elements))
        similarity = model.similarity(gt_entry_embeddings, model_entry_embeddings)
        if binarize:  # If the max match is above a threshold, treat it like a match.
            score = ((similarity.max(dim=1).values > threshold) * 1.0).sum().item() / float(max(similarity.shape))
        else:  # If the max match is above a threshold, add it raw rather than snap to 1.0.
            score = ((similarity.max(dim=1).values > threshold) * similarity.max(dim=1).values).sum().item() / float(max(similarity.shape))
        # If the ground truth and model were perfectly aligned we would have a square matrix with ones along the diagonal.

    score = min(1.0, score)  # Sometimes there's some arithmetic error. Snap to 1-0.

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
    Task.NER_NESTED: eval_ner_nested,
    Task.THING_EXTRACTION: eval_thing_extraction,
    Task.UNIT_EXTRACTION: eval_unit_extraction,
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
