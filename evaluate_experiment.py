
import json
import sqlite3
#%pip install partial-json-parser
from dataclasses import dataclass

import partial_json_parser
from pydantic import ValidationError

from dataset import load_ground_truth
from task import Task, DATA_MODELS, NERMultiExtraction, NERNestedExtraction, NestedEventExtraction, NestedThingExtraction, NestedUnitExtraction


@dataclass
class Score:
    well_formed_json: bool = False
    schema_matches_exact: bool = False  # Has all required fields and no extra imagined fields.
    schema_matches_min: bool = False  # Has at least the required fields but may have extra imagined fields. match_exact implies match min.
    constraints_followed: bool = False  # All fields are present and none violate constraints
    #content_extracted: bool = False  # Contain the desired data and no hallucinated content
    #no_duplicate_content: bool = False  # Contains only desired valid data. Not multiple copies of it
    #no_hallucinated_content: bool = False  # Contains the desired valid data but has more than 50% other unrelated data (badly hallucinated content)
    iou_score: float = 0.0  # The intersection over union of the data with ground truth.


def run_evaluations():
    print("Reading model_outputs.db and saving to evaluation.db")
    input_connection = sqlite3.connect('model_outputs.db')
    input_connection.row_factory = sqlite3.Row
    output_connection = sqlite3.connect("evaluation.db")
    output_connection.row_factory = sqlite3.Row
    output_connection.execute("CREATE TABLE evaluation (task TEXT, guardrails TEXT, model TEXT, num_shots INTEGER, dataset_name TEXT, doc_idx INTEGER, well_formed_json INTEGER, schema_matches_min INTEGER, schema_matches_exact INTEGER, constraints_followed INTEGER, content_iou REAL);")
    cursor = input_connection.cursor()
    cursor.execute("SELECT * FROM model_outputs;")
    ground_truth = load_ground_truth()
    for r in cursor:
        s = Score()
        task = Task(r["task"])

        # Just try to load the thing, falling back to partial loading.
        maybe_data = dict()
        try:
            maybe_data = json.loads(r['raw_model_output'])
            s.well_formed_json = True
        except json.JSONDecodeError:
            s.well_formed_json = False

        # Not well formed, but parsable:
        if not s.well_formed_json:
            try:
                maybe_data = partial_json_parser.loads(r['raw_model_output'])
            except Exception:  # partial_json_parser.JSONDecodeError:
                maybe_data = dict()

        # Diff the keys to see if we have extras:
        # TODO: Use the BFCL list/dict tree comparison.
        parsed_data = None
        if dict_has_all_fields(DATA_MODELS[task].model_fields, maybe_data):
            s.schema_matches_min = True
            if dict_has_no_extra_fields(DATA_MODELS[task].model_fields, maybe_data):
                s.schema_matches_exact = True
        try:
            #DATA_MODELS[task].__pydantic_model__.parse_raw()
            #if isinstance(maybe_data, dict):
            #    parsed_data = DATA_MODELS[task](**maybe_data)
            #elif isinstance(maybe_data, list):
            #    parsed_data = DATA_MODELS[task](maybe_data)
            parsed_data = DATA_MODELS[task](**maybe_data)
            s.schema_matches_exact = True
            s.constraints_followed = True
            s.schema_matches_min = True
        except Exception:
            # Missing some key fields.
            parsed_data = None

        # Check all values.
        # The first qualitative measure.
        if task in TASK_TO_EVAL_FN and parsed_data is not None:
            try:
                TASK_TO_EVAL_FN[task](ground_truth[str(task)][r["dataset"]][str(r["doc_idx"])], parsed_data, s)
            except Exception as e:
                print(e)

        output_connection.execute(
            "INSERT INTO evaluation (task, guardrails, model, num_shots, dataset_name, doc_idx, well_formed_json, schema_matches_min, schema_matches_exact, constraints_followed, content_iou) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
            (r["task"], r["guardrails"], r["model"], r["num_shots"], r["dataset"], r["doc_idx"], s.well_formed_json, s.schema_matches_min, s.schema_matches_exact, s.constraints_followed, s.iou_score)
        )
    output_connection.commit()
    input_connection.close()
    output_connection.close()


def eval_ner_flat(ground_truth: dict, parsed_data: NERMultiExtraction, score_ref: Score):
    # TODO: This is perhaps slightly lazy.  Since NER is a mess of edge cases, we pile together the different lists and then diff them.
    ground_truth = NERMultiExtraction(**ground_truth)
    gt_pile_of_names = set()
    gt_pile_of_names.union(set(ground_truth.names))
    gt_pile_of_names.union(set(ground_truth.locations))
    gt_pile_of_names.union(set(ground_truth.organizations))
    gt_pile_of_names.union(set(ground_truth.misc))

    parsed_pile_of_names = set()
    parsed_pile_of_names.union(set(parsed_data.names))
    parsed_pile_of_names.union(set(parsed_data.locations))
    parsed_pile_of_names.union(set(parsed_data.organizations))
    parsed_pile_of_names.union(set(parsed_data.misc))

    # IoU evaluation.
    intersection_size = float(len(parsed_pile_of_names.intersection(gt_pile_of_names)))
    union_size = float(len(parsed_pile_of_names.union(gt_pile_of_names)))
    if union_size < 1e-6:
        raise Exception("Ground truth empty!?")
    else:
        score_ref.iou_score = intersection_size/union_size


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
