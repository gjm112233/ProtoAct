import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd


# BLEU metric helper functions.
# Function 1: convert parameter values to tokens.
def value_to_tokens(value: Any) -> list[str]:
    """
    Convert a parameter value into a token list that BLEU can consume.
    Example:
    96-well plate with 100uL cell suspension per well 37 C
    ['96-well', 'plate', 'with', '100uL', 'cell', 'suspension', 'per', 'well', '37 C']
    """
    text = str(value).strip()

    # Split the text into three token categories:
    # continuous alphanumeric spans, numeric values with unit suffixes, and single symbols, while ignoring whitespace.
    tokens = re.findall(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*(?:[%℃])?|[^\sA-Za-z0-9]", text)
    return tokens


# Function 2: compute BLEU for one parameter value.
def compute_bleu_for_value(
    gt_value: Any,
    pred_value: Any,
    weights: tuple[float, float, float, float] = (0.5, 0.5, 0, 0),
) -> float:
    """
    Compute BLEU for a single parameter value. By default this uses up to 2-gram BLEU.
    """
    reference_tokens = value_to_tokens(gt_value)
    candidate_tokens = value_to_tokens(pred_value)
    if not reference_tokens or not candidate_tokens:
        return 0.0

    # Use 1-gram BLEU when either side has only one token.
    if len(reference_tokens) == 1 or len(candidate_tokens) == 1:
        weights = (1, 0, 0, 0)

    score = sentence_bleu(
        [reference_tokens],  # reference must be list[list[str]].
        candidate_tokens,  # candidate must be list[str].
        weights=weights,
        smoothing_function=SmoothingFunction().method1
    )
    return score


# Function 3: compute BLEU for all parameters in one matched action pair and its mean.
def compute_bleu_for_matched_action(
    gt_item: dict[str, Any],
    pred_item: dict[str, Any],
    weights: tuple[float, float, float, float] = (0.5, 0.5, 0, 0),
) -> dict[str, Any]:
    """
    Compute BLEU for every parameter in one matched action pair.
    The result records the action name, one BLEU score per parameter, and the mean BLEU for this action.
    """
    gt_action = gt_item.get("action")
    pred_action = pred_item.get("action")

    if gt_action != pred_action:
        raise ValueError(f"Action mismatch; cannot compare: gt={gt_action}, pred={pred_action}")

    # Rebuild the action dictionary.
    # Treat every key except action as a parameter.
    gt_params = {k: v for k, v in gt_item.items() if k != "action"}
    pred_params = {k: v for k, v in pred_item.items() if k != "action"}

    # If parameter names differ, keep only shared parameters so scores are comparable.
    if set(gt_params.keys()) != set(pred_params.keys()):
        # Find the parameter names present on both sides.
        common_keys = set(gt_params.keys()) & set(pred_params.keys())
        gt_params = {k: v for k, v in gt_params.items() if k in common_keys}
        pred_params = {k: v for k, v in pred_params.items() if k in common_keys}

    # Record one score for each parameter.
    param_scores: dict[str, float] = {}

    # Iterate over parameters in a stable order.
    for param_name in sorted(gt_params.keys()):
        # Read the gt and pred values for this parameter.
        gt_value = gt_params[param_name]
        pred_value = pred_params[param_name]

        # Compute BLEU for this parameter pair.
        score = compute_bleu_for_value(gt_value, pred_value, weights=weights)
        param_scores[param_name] = score  # Save this parameter score.

    # Average all parameter BLEU scores for this item.
    mean_bleu = (
        sum(param_scores.values()) / len(param_scores)
        if param_scores else 0.0
    )

    return {
        "action": gt_action,
        "param_scores": param_scores,
        "mean_bleu": mean_bleu
    }


# Function 4: compute BLEU for all matched action pairs.
def compute_bleu_for_all_matched_actions(
    gt_matched: list[dict[str, Any]],
    pred_matched: list[dict[str, Any]],
    weights: tuple[float, float, float, float] = (0.5, 0.5, 0, 0),
) -> dict[str, Any]:
    """
    Compute parameter BLEU for all matched action pairs.
    Returns the per-action details, the global mean over all parameter scores, and the mean over action-level averages.
    """
    if len(gt_matched) != len(pred_matched):
        raise ValueError("gt_matched and pred_matched must have the same length")

    per_action_results: list[dict[str, Any]] = []  # BLEU or SciBERTScore results for each matched action.

    all_param_scores: list[float] = []
    all_action_mean_scores: list[float] = []

    # Iterate through each aligned gt/pred pair.
    for gt_item, pred_item in zip(gt_matched, pred_matched):
        # Compute action-level parameter BLEU details.
        # Compute action-level parameter BLEU details.
        # Compute action-level parameter BLEU details.
        # Compute action-level parameter BLEU details.
        # Compute action-level parameter BLEU details.
        #     }
        result = compute_bleu_for_matched_action(
            gt_item,
            pred_item,
            weights=weights
        )
        per_action_results.append(result)

        # Collect the current item's per-parameter scores.
        param_scores = list(result["param_scores"].values())
        # Append these scores to the global parameter-score list.
        all_param_scores.extend(param_scores)

        # Record the mean score for the current action.
        all_action_mean_scores.append(result["mean_bleu"])

    # Average all parameter BLEU scores for this item.
    global_param_mean_bleu = (
        sum(all_param_scores) / len(all_param_scores)
        if all_param_scores else 0.0
    )

    # Average all parameter BLEU scores for this item.
    action_mean_bleu = (
        sum(all_action_mean_scores) / len(all_action_mean_scores)
        if all_action_mean_scores else 0.0
    )

    return {
        # Evaluation step for this metric calculation.
        # Evaluation step for this metric calculation.
        # Evaluation step for this metric calculation.
        # Evaluation step for this metric calculation.
        #       }
        "per_action_results": per_action_results,
        # Average all parameter scores across all matched items.
        "global_param_mean_bleu": global_param_mean_bleu,
        # Average the per-action mean scores.
        "action_mean_bleu": action_mean_bleu
    }


def match_actions_for_tp(
    gt_subtasks: List[Dict[str, Any]],
    pred_subtasks: List[Dict[str, Any]]
):
    """
    Match action pairs that count as true positives between gt and pred.
    Matching is one-to-one and preserves the first unused pred item with the same action name.
    """

    # Track which pred items have already been used to avoid duplicate matches.
    used = [False] * len(pred_subtasks)

    gt_matched = []
    pred_matched = []
    unmatched_pred = []  # Pred actions that were not matched.

    # Iterate through every gt subtask.
    for gt_item in gt_subtasks:
        gt_action = gt_item.get("action")  # Read the current gt action name.

        # Search pred for the next unused item with the same label.
        for i, pred_item in enumerate(pred_subtasks):
            if used[i]:
                continue

            pred_action = pred_item.get("action")  # Read the current pred action name.

            # When action names match, record this pair.
            if gt_action == pred_action:
                # Append the matched gt item.
                gt_matched.append(gt_item)
                pred_matched.append(pred_item)

                used[i] = True  # Mark this pred item as used.
                break

    for i in range(len(pred_subtasks)):
        if used[i] == False:
            unmatched_pred.append(pred_subtasks[i])
    return gt_matched, pred_matched, unmatched_pred


def compute_f1(gt_counts: Counter, pred_counts: Counter):
    # Take the union of labels that appear in gt or pred.
    all_actions = set(gt_counts.keys()) | set(pred_counts.keys())

    TP = 0
    FP = 0
    FN = 0

    # Take the union of labels that appear in gt or pred.
    for action in all_actions:
        gt = gt_counts.get(action, 0)  # Get how many times this action appears in gt.
        pred = pred_counts.get(action, 0)  # Get how many times this action appears in pred.

        TP += min(gt, pred)  # Number of correctly predicted actions.
        FP += max(pred - gt, 0)  # Number of extra predicted actions (pred > gt).
        FN += max(gt - pred, 0)  # Number of missed actions (gt > pred).

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Load JSON data from disk.
def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Normalize supported JSON layouts into a stage list.
def normalize_stages(data: Any) -> list[dict[str, Any]]:
    """
    Normalize supported JSON layouts into a list of stage dictionaries.
    Supported layouts are: {"stages": [...]}, a top-level stage list, or a single-stage object with monitor and subtasks keys.
    """
    if isinstance(data, dict):  # If data is a dictionary.
        if "stages" in data:  # Check whether the "stages" key exists.
            stages = data["stages"]  # Read the value corresponding to "stages".
            if not isinstance(stages, list):  # Continue checking whether stages is a list.
                # Raise a clear error for invalid input.
                raise ValueError("'stages' must be a list")
            return stages  # For format 1, return the stages value.

        # Treat a single-stage object as a one-item stage list.
        if "monitor" in data and "subtasks" in data:
            return [data]

        raise ValueError(
            "Unexpected JSON structure. Use {'stages': [...]} or a single-stage {'monitor': ..., 'subtasks': ...} structure."
        )

    # A top-level list is already a stage list.
    if isinstance(data, list):
        return data

    raise ValueError("Unexpected JSON structure. Use {'stages': [...]} or a single-stage {'monitor': ..., 'subtasks': ...} structure.")


def canonical_param_string(item: dict[str, Any], exclude_keys: set[str]) -> str:
    """
    Convert all non-key parameters of an action or monitor into a stable comparable string.
    Example: {"action":"set_centrifuge","force":"800g","time":"6min"} becomes "force=800g|time=6min".
    """
    params = {k: v for k, v in item.items() if k not in exclude_keys}
    if not params:
        return ""

    parts = [f"{k}={params[k]}" for k in sorted(params)]
    return "|".join(parts)


# Flatten all monitors and subtasks from stages.
# Count occurrences and parameter combinations for each action or monitor type.
def flatten_and_count(stages: list[dict[str, Any]]) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    Counter,
    Counter,
    dict[str, Counter],
    dict[str, Counter],
    Counter,
    Counter,
]:
    """
    Flatten all monitors and subtasks from every stage.
    Also count each monitor type, each action type, and each stable parameter combination per type/action.
    """
    all_monitors: list[dict[str, Any]] = []  # Flattened list of all monitors.
    all_subtasks: list[dict[str, Any]] = []  # Flattened list of all subtasks.
    # Initialize counters for monitor types and action types.
    gt_monitor_counts: Counter = Counter()  # Count occurrences of each monitor type.
    gt_subtask_counts: Counter = Counter()  # Count occurrences of each action type.

    # Evaluation step for this metric calculation.
    gt_subtask_param_key_counts: Counter = Counter()

    # Evaluation step for this metric calculation.
    gt_monitor_param_key_counts: Counter = Counter()

    # Create a dictionary to count parameter combinations for each monitor/subtask type.
    # Count how many times each distinct parameter combination appears.
    # Example: {"type": "wait_time", "time": "6min"}
    #      {"type": "wait_time", "time": "48h"}
    # Result: {
    #     "wait_time": Counter({
    #         "time=6min": 1,
    #         "time=48h": 1
    #      })
    # }
    gt_monitor_params: dict[str, Counter] = defaultdict(Counter)

    # Same idea for subtasks.
    # Result: {
    #     "set_centrifuge": Counter({
    #         "force=800g|time=6min": 2
    #     }),
    #     "aspirate": Counter({
    #         "amount=1mL|object=PBS": 1,
    #         "amount=2mL|object=trypsin": 1
    #     })
    # }
    gt_subtask_params: dict[str, Counter] = defaultdict(Counter)

    # Iterate through stages with an index for error messages.
    for idx, stage in enumerate(stages):
        # Read monitor and subtasks lists, defaulting to empty lists.
        monitors = stage.get("monitor", [])
        subtasks = stage.get("subtasks", [])

        # Validate that the monitor field is a list.
        if not isinstance(monitors, list):
            raise ValueError(f"Stage {idx} field 'monitor' is not a list")
        if not isinstance(subtasks, list):
            raise ValueError(f"Stage {idx} field 'subtasks' is not a list")

        # Process every monitor dictionary in the current stage.
        for m in monitors:
            if not isinstance(m, dict):
                raise ValueError(f"Stage {idx} contains a non-dict monitor")
            if "type" not in m:
                raise ValueError(f"Stage {idx} monitor is missing the 'type' field")

            # Normalize the monitor type string.
            m_type = str(m["type"]).strip()
            # Add this monitor to the flattened list.
            all_monitors.append(m)
            # Increment the count for this monitor type.
            gt_monitor_counts[m_type] += 1

            # Convert monitor parameters into a stable string.
            # {"action":"set_centrifuge","force":"800g","time":"6min"}
            #     -> "force=800g|time=6min"
            param_str = canonical_param_string(m, exclude_keys={"type"})
            # Increment the parameter-combination count for this monitor type.
            gt_monitor_params[m_type][param_str] += 1

            # Count monitor parameter keys except type.
            for key in m.keys():
                if key != "type":
                    gt_monitor_param_key_counts[key] += 1

        # Process every subtask dictionary in the current stage.
        for s in subtasks:
            if not isinstance(s, dict):
                raise ValueError(f"Stage {idx} contains a non-dict subtask")
            if "action" not in s:
                raise ValueError(f"Stage {idx} subtask is missing the 'action' field")

            # Normalize the action name string.
            action = str(s["action"]).strip()
            # Add this subtask to the flattened list.
            all_subtasks.append(s)
            # Increment the count for this action type.
            gt_subtask_counts[action] += 1

            # Convert subtask parameters into a stable string.
            param_str = canonical_param_string(s, exclude_keys={"action"})
            # Increment the parameter-combination count for this action type.
            gt_subtask_params[action][param_str] += 1

            # Count subtask parameter keys except action.
            for key in s.keys():
                if key != "action":
                    gt_subtask_param_key_counts[key] += 1

    return (
        all_monitors,
        all_subtasks,
        gt_monitor_counts,
        gt_subtask_counts,
        gt_monitor_params,
        gt_subtask_params,
        gt_monitor_param_key_counts,
        gt_subtask_param_key_counts
    )


def main() -> None:
    # cell_names = ["renalcell", "alveolarcell", "gallbladdercell", "mammarycell", "prostatecell",
    #               "vascularcell", "glialcell", "melanocytes", "adipocyte", "musclecell", "neuralcell",
    #               "adipose_stemcell", "dentalpulpcell", "cartilagecell", "osteoblast", "osteoclast", "umbilicalcell", "lymphcell", "pancreascell"]
    cell_names = ["osteoclast", ]
    for cell_name in cell_names:
        gt_data = load_json(f"./test_data/gt_{cell_name}.json")
        gt_stages = normalize_stages(gt_data)  # Normalize gt data and return the stages list.
        # print(gt_stages)

        pred_data = load_json(f"./test_data/qwen3-max_pred_{cell_name}.json")
        pred_stages = normalize_stages(pred_data)

        (
            gt_all_monitors,  # Flattened list of all gt monitors.
            gt_all_subtasks,  # Flattened list of all gt subtasks.
            gt_monitor_counts,  # Occurrence count for each gt monitor type.
            gt_subtask_counts,  # Occurrence count for each gt action.
            gt_monitor_params,  # Parameter-combination counts for each gt monitor type.
            gt_subtask_params,  # Parameter-combination counts for each gt action type.
            gt_monitor_param_key_counts,  # Count each monitor parameter key in gt.
            gt_subtask_param_key_counts  # Count each subtask parameter key in gt.
        ) = flatten_and_count(gt_stages)

        # print(gt_subtask_param_key_counts)

        (
            pred_all_monitors,  # Flattened list of all predicted monitors.
            pred_all_subtasks,  # Flattened list of all predicted subtasks.
            pred_monitor_counts,  # Count occurrences of each predicted monitor type.
            pred_subtask_counts,  # Count occurrences of each predicted action type.
            pred_monitor_params,  # Parameter-combination counts for each predicted monitor type.
            pred_subtask_params,  # Parameter-combination counts for each predicted action type.
            pred_monitor_param_key_counts,  # Count each monitor parameter key in pred.
            pred_subtask_param_key_counts  # Count each subtask parameter key in pred.
        ) = flatten_and_count(pred_stages)

        print("Current cell:", cell_name)
        # print("\n=== monitor_counts ===")
        # print("gt:", dict(gt_monitor_param_key_counts))
        # print("pred:", dict(pred_monitor_param_key_counts))
        #
        # print("\n=== subtask_counts ===")
        # print("gt:", dict(gt_subtask_param_key_counts))
        # print("pred:", dict(pred_subtask_param_key_counts))

        print("=====================================")
        # Optional diagnostic output.
        print("Action-level Precision / Recall / F1")
        result = compute_f1(gt_subtask_param_key_counts, pred_subtask_param_key_counts)

        print("TP(correct action parameters):", result["TP"])
        print("FP(extra predicted action parameters):", result["FP"])  # Number of false positives.
        print("FN(missed action parameters):", result["FN"])  # Number of false negatives.
        # # print("precision:", result["precision"])
        # # print("recall:", result["recall"])
        # # print("f1:", result["f1"])
        #
        file_path = "./result/precision_recall_action_param.xlsx"
        old_df = pd.read_excel(file_path)
        # [gpt-5.3, qwen3-max, qwen-max, deepseek-v3.2, deepseek-r1, Kimi-K2, glm-4.5]
        new_row = pd.DataFrame([{
            "model_name": "qwen3-max",
            "experiment": cell_name,
            "TP": result["TP"],
            "FP": result["FP"],
            "FN": result["FN"]
        }])
        df = pd.concat([old_df, new_row], ignore_index=True)
        df.to_excel(file_path, index=False)
        print("action parameter append complete")

        print("\n=====================================")
        # Optional diagnostic output.
        print("Monitoring Condition Precision / Recall / F1")
        result2 = compute_f1(gt_monitor_param_key_counts, pred_monitor_param_key_counts)

        print("TP(correct monitor parameters):", result2["TP"])
        print("FP(extra predicted monitor parameters):", result2["FP"])  # Number of false positives.
        print("FN(missed monitor parameters):", result2["FN"])  # Number of false negatives.
        # # print("precision:", result2["precision"])
        # # print("recall:", result2["recall"])
        # # print("f1:", result2["f1"])
        # print("=======================\n\n")
        #
        file_path = "./result/precision_recall_monitor_param.xlsx"
        old_df = pd.read_excel(file_path)
        new_row = pd.DataFrame([{
            "model_name": "qwen3-max",
            "experiment": cell_name,
            "TP": result2["TP"],
            "FP": result2["FP"],
            "FN": result2["FN"]
        }])
        df = pd.concat([old_df, new_row], ignore_index=True)
        df.to_excel(file_path, index=False)
        print("monitor parameter append complete")
        print("=======================")


if __name__ == "__main__":
    main()
