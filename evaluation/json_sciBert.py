import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
import re
import pandas as pd


# Load the SciBERT tokenizer and model.
tokenizer = AutoTokenizer.from_pretrained(
    "allenai/scibert_scivocab_uncased"
)
model = AutoModel.from_pretrained(
    "allenai/scibert_scivocab_uncased"
)


def encode(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding


def scibert_score(pred, gt):
    v1 = encode(pred)
    v2 = encode(gt)
    score = F.cosine_similarity(v1, v2)
    return score.item()


# Compute SciBERTScore for all matched action pairs.
def compute_bertScore_for_all_matched_actions(
    gt_matched: list[dict[str, Any]],
    pred_matched: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute parameter SciBERTScore for all matched action pairs.
    Returns the per-action details, the global mean over all parameter scores, and the mean over action-level averages.
    """
    count = 0
    if len(gt_matched) != len(pred_matched):
        raise ValueError("gt_matched and pred_matched must have the same length")

    per_action_results: list[dict[str, Any]] = []  # BLEU or SciBERTScore results for each matched action.

    all_param_scores: list[float] = []  # Record every parameter SciBERTScore value.
    all_action_mean_scores: list[float] = []  # Record the average SciBERTScore for each action.

    # Iterate through each aligned gt/pred pair.
    for gt_item, pred_item in zip(gt_matched, pred_matched):
        # Compute action-level parameter SciBERTScore details.
        # Compute action-level parameter SciBERTScore details.
        # Compute action-level parameter SciBERTScore details.
        # Compute action-level parameter SciBERTScore details.
        # Compute action-level parameter SciBERTScore details.
        #     }
        result = compute_bertscore_for_matched_action(
            gt_item,
            pred_item,
        )
        per_action_results.append(result)

        # Collect the current item's per-parameter scores.
        param_scores = list(result["param_scores"].values())
        # Append these scores to the global parameter-score list.
        all_param_scores.extend(param_scores)

        # Record the mean score for the current action.
        all_action_mean_scores.append(result["mean_bert_score"])

        count += 1
        # Average all parameter SciBERTScore values for this item.

    # Average all parameter SciBERTScore values for this item.
    global_param_mean_bert_score = (
        sum(all_param_scores) / len(all_param_scores)
        if all_param_scores else 0.0
    )

    sum_all_param_score = sum(all_param_scores)
    count_params = len(all_param_scores)

    # Average all parameter SciBERTScore values for this item.
    action_mean_bert_score = (
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
        "global_param_mean_bert_score": global_param_mean_bert_score,
        # Average the per-action mean scores.
        "action_mean_bert_score": action_mean_bert_score,
        "sum_all_param_score": sum_all_param_score,
        "count_params": count_params
    }


# Compute SciBERTScore for all parameters in one matched action pair and its mean.
def compute_bertscore_for_matched_action(
    gt_item: dict[str, Any],
    pred_item: dict[str, Any]
) -> dict[str, Any]:
    """
    Compute SciBERTScore for every parameter in one matched action pair.
    The result records the action name, one score per parameter, and the mean score for this action.
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

        # Compute SciBERTScore for this parameter pair.
        score = scibert_score(pred_value, gt_value)
        param_scores[param_name] = score  # Save this parameter score.

    # Average all parameter SciBERTScore values for this item.
    mean_bert_score = (
        sum(param_scores.values()) / len(param_scores)
        if param_scores else 0.0
    )

    return {
        "action": gt_action,
        "param_scores": param_scores,
        "mean_bert_score": mean_bert_score
    }


# Monitor SciBERTScore helper functions.
# Compute SciBERTScore for all matched monitor pairs.
def compute_bertScore_for_all_matched_monitors(
    gt_matched: list[dict[str, Any]],
    pred_matched: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute parameter SciBERTScore for all matched monitor pairs.
    Returns the per-monitor details, the global mean over all parameter scores, and the mean over monitor-level averages.
    """
    count = 0
    if len(gt_matched) != len(pred_matched):
        raise ValueError("gt_matched and pred_matched must have the same length")

    per_monitor_results: list[dict[str, Any]] = []  # BLEU or SciBERTScore results for each matched monitor.

    all_param_scores: list[float] = []  # Record every parameter SciBERTScore value.
    all_monitor_mean_scores: list[float] = []  # Record the average SciBERTScore for each monitor condition.

    # Iterate through each aligned gt/pred pair.
    for gt_item, pred_item in zip(gt_matched, pred_matched):
        # Compute monitor-level parameter SciBERTScore details.
        # Compute monitor-level parameter SciBERTScore details.
        # Compute monitor-level parameter SciBERTScore details.
        # Compute monitor-level parameter SciBERTScore details.
        # Compute monitor-level parameter SciBERTScore details.
        #     }
        result = compute_bertscore_for_matched_monitor(
            gt_item,
            pred_item,
        )
        per_monitor_results.append(result)

        # Collect the current item's per-parameter scores.
        param_scores = list(result["param_scores"].values())
        # Append these scores to the global parameter-score list.
        all_param_scores.extend(param_scores)

        # Record the mean score for the current monitor.
        all_monitor_mean_scores.append(result["mean_bert_score"])

        count += 1
        # Average all parameter SciBERTScore values for this item.

    # Average all parameter SciBERTScore values for this item.
    global_param_mean_bert_score = (
        sum(all_param_scores) / len(all_param_scores)
        if all_param_scores else 0.0
    )

    sum_all_param_scores = sum(all_param_scores)
    count_params = len(all_param_scores)

    # Average all parameter SciBERTScore values for this item.
    monitor_mean_bert_score = (
        sum(all_monitor_mean_scores) / len(all_monitor_mean_scores)
        if all_monitor_mean_scores else 0.0
    )

    return {
        # Evaluation step for this metric calculation.
        # Evaluation step for this metric calculation.
        # Evaluation step for this metric calculation.
        # Evaluation step for this metric calculation.
        #       }
        "per_monitor_results": per_monitor_results,
        # Average all parameter scores across all matched items.
        "global_param_mean_bert_score": global_param_mean_bert_score,
        # Average the per-monitor mean scores.
        "monitor_mean_bert_score": monitor_mean_bert_score,
        "sum_all_param_scores": sum_all_param_scores,
        "count_params": count_params
    }


# Compute SciBERTScore for all parameters in one matched monitor pair and its mean.
def compute_bertscore_for_matched_monitor(
    gt_item: dict[str, Any],
    pred_item: dict[str, Any]
) -> dict[str, Any]:
    """
    Compute SciBERTScore for every parameter in one matched monitor pair.
    The result records the monitor type, one score per parameter, and the mean score for this monitor.
    """
    gt_type = gt_item.get("type")
    pred_type = pred_item.get("type")

    if gt_type != pred_type:
        raise ValueError(f"Monitor type mismatch; cannot compare: gt={gt_type}, pred={pred_type}")

    # Rebuild the monitor-condition dictionary.
    # Treat every key except type as a parameter.
    gt_params = {k: v for k, v in gt_item.items() if k != "type"}
    pred_params = {k: v for k, v in pred_item.items() if k != "type"}

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

        # Compute SciBERTScore for this parameter pair.
        score = scibert_score(pred_value, gt_value)
        param_scores[param_name] = score  # Save this parameter score.

    # Average all parameter SciBERTScore values for this item.
    mean_bert_score = (
        sum(param_scores.values()) / len(param_scores)
        if param_scores else 0.0
    )

    return {
        "monitor_type": gt_type,
        "param_scores": param_scores,
        "mean_bert_score": mean_bert_score
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

    return (
        all_monitors,
        all_subtasks,
        gt_monitor_counts,
        gt_subtask_counts,
        gt_monitor_params,
        gt_subtask_params,
    )


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

        if gt_action in ["transfer", "shake", "transfer_liquid", "aspirate", "dispense", "discard_medium", "start_device"]:
            continue

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


def match_monitors_for_tp(
    gt_monitors: List[Dict[str, Any]],
    pred_monitors: List[Dict[str, Any]]
):
    """
    Match monitor pairs that count as true positives between gt and pred.
    Matching is one-to-one and preserves the first unused pred item with the same monitor type.
    """

    # Track which pred items have already been used to avoid duplicate matches.
    used = [False] * len(pred_monitors)

    gt_matched = []
    pred_matched = []

    # Iterate through every gt monitor.
    for gt_item in gt_monitors:
        gt_monitor = gt_item.get("type")  # Read the current gt monitor type.

        # Search pred for the next unused item with the same label.
        for i, pred_item in enumerate(pred_monitors):
            if used[i]:
                continue

            pred_monitor = pred_item.get("type")  # Read the current pred monitor type.

            # When monitor types match, record this pair.
            if gt_monitor == pred_monitor:
                # Append the matched gt item.
                gt_matched.append(gt_item)
                pred_matched.append(pred_item)

                used[i] = True  # Mark this pred item as used.
                break

    return gt_matched, pred_matched

# ==========================================================================================================================
# Parameter order for each supported action.
# Parameter order for each supported action.
ACTION_PARAMS = {
    "transfer": ["object", "source", "target"],
    "shake": ["object"],
    "transfer_liquid": ["source", "target"],
    "aspirate": ["object", "amount"],
    "dispense": ["place", "amount"],
    "discard_medium": ["object"],
    "start_device": ["device"]
}


# 5
def action_to_text(action: dict[str, Any]) -> str:
    """
    Convert an action dictionary into text for similarity comparison.
    Only parameters defined for that action are used, and they are appended in a fixed order.
    """
    # Evaluation step for this metric calculation.
    action_type = action.get("action")
    if action_type not in ACTION_PARAMS:
        return ""

    parts = []
    # Iterate through parameter names defined for this action.
    for key in ACTION_PARAMS[action_type]:
        # Read the value for this parameter key.
        value = action.get(key)
        # Evaluation step for this metric calculation.
        if value is not None and str(value).strip() != "":
            parts.append(str(value).strip())

    # Join parameter values into comparison text.
    return " ".join(parts)


# 7
def simple_tokenize(text: str) -> list[str]:
    """
    Tokenizer helper confirmed to work in this evaluation script.
    """
    import re
    return re.findall(
        r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*(?:[%℃])?|[^\sA-Za-z0-9]",
        str(text)
    )


# 6
def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Compute a simple Jaccard similarity without loading an additional model.
    This can be replaced by SciBERTScore if semantic similarity is needed later.
    """
    # Lowercase, tokenize, and deduplicate both texts.
    # Lowercase, tokenize, and deduplicate both texts.
    # Lowercase, tokenize, and deduplicate both texts.
    tokens1 = set(simple_tokenize(text1.lower()))
    tokens2 = set(simple_tokenize(text2.lower()))

    # If both token sets are empty, treat them as identical.
    if not tokens1 and not tokens2:
        return 1.0
    # If only one token set is empty, similarity is zero.
    if not tokens1 or not tokens2:
        return 0.0

    # Compute Jaccard similarity as shared tokens divided by all tokens.
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


# 4
def build_action_similarity_matrix(
    gt_group: list[dict[str, Any]],
    pred_group: list[dict[str, Any]],
    sim_func=jaccard_similarity,
) -> np.ndarray:
    """
    Build the pairwise gt/pred similarity matrix for one action group.
    The matrix shape is [len(gt_group), len(pred_group)].
    """

    # Return an all-zero matrix or empty match when either group is empty.
    if not gt_group or not pred_group:
        return np.zeros((len(gt_group), len(pred_group)))

    # Create an m by n similarity matrix for non-empty groups.
    # Create an m by n similarity matrix for non-empty groups.
    sim_matrix = np.zeros((len(gt_group), len(pred_group)), dtype=float)

    # Iterate through each gt item and its row index.
    for i, gt_item in enumerate(gt_group):
        # Convert the gt item to comparison text.
        gt_text = action_to_text(gt_item)
        # Iterate through each pred item and its column index.
        for j, pred_item in enumerate(pred_group):
            # Convert the pred item to comparison text.
            pred_text = action_to_text(pred_item)
            # Store the text similarity in the matrix.
            sim_matrix[i, j] = sim_func(gt_text, pred_text)

    # Return the completed similarity matrix.
    return sim_matrix


# 3
def match_one_action(
    gt_group: list[dict[str, Any]],
    pred_group: list[dict[str, Any]],
    sim_func=jaccard_similarity,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Find the optimal one-to-one matching for gt and pred items with the same action.
    Returns matched gt and pred lists with aligned positions.
    """

    # Return an all-zero matrix or empty match when either group is empty.
    # Return an all-zero matrix or empty match when either group is empty.
    if not gt_group or not pred_group:
        return [], []

    # Evaluation step for this metric calculation.
    # Evaluation step for this metric calculation.
    # Evaluation step for this metric calculation.
    sim_matrix = build_action_similarity_matrix(gt_group, pred_group, sim_func=sim_func)

    # Hungarian matching minimizes cost, so convert similarity to cost with 1 - similarity.
    # Hungarian matching minimizes cost, so convert similarity to cost with 1 - similarity.
    cost_matrix = 1.0 - sim_matrix

    # Find the best one-to-one assignment with the Hungarian algorithm.
    # Find the best one-to-one assignment with the Hungarian algorithm.
    # Find the best one-to-one assignment with the Hungarian algorithm.
    # Find the best one-to-one assignment with the Hungarian algorithm.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Use selected row indices to collect matched gt items.
    # Use selected row indices to collect matched gt items.
    # matched_gt = [gt_group[0], gt_group[1]]
    matched_gt = [gt_group[i] for i in row_ind]
    # Use selected column indices to collect matched pred items.
    matched_pred = [pred_group[j] for j in col_ind]

    return matched_gt, matched_pred


# 2
def group_actions_by_type(actions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """
    Group action dictionaries by their action name.
    """
    # Create a defaultdict(list) so new labels start with an empty list.
    # Create a defaultdict(list) so new labels start with an empty list.
    # {"visual_state": [{}, {},...]
    #   "wait_time": [{}, {}, ...]
    # }
    grouped = defaultdict(list)
    # Iterate through every action dictionary.
    for item in actions:
        action_type = item.get("action")  # Read the action name.
        # Add the action to the group for its action name.
        grouped[action_type].append(item)
    return grouped


# 1
def match_all_actions(
    gt_all_monitors: list[dict[str, Any]],
    pred_all_monitors: list[dict[str, Any]],
    sim_func=jaccard_similarity,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Match all supported action types.
    First group by action, then match within each group, and finally return aligned gt and pred lists.
    """

    # Group gt and pred actions by action name.
    # {"visual_state": [{}, {},...],
    #   "wait_time": [{}, {}, ...],
    # }
    gt_grouped = group_actions_by_type(gt_all_monitors)
    pred_grouped = group_actions_by_type(pred_all_monitors)

    gt_matched = []
    pred_matched = []

    # Use a fixed action order for stable results.
    all_types = ["transfer", "shake", "transfer_liquid", "aspirate", "dispense", "discard_medium", "start_device"]

    # Match each supported action type in order.
    for action_type in all_types:
        # Read the gt and pred group for this label, defaulting to an empty list.
        # Read the gt and pred group for this label, defaulting to an empty list.
        gt_group = gt_grouped.get(action_type, [])
        pred_group = pred_grouped.get(action_type, [])

        matched_gt_group, matched_pred_group = match_one_action(
            gt_group,
            pred_group,
            sim_func=sim_func,
        )

        gt_matched.extend(matched_gt_group)
        pred_matched.extend(matched_pred_group)

    return gt_matched, pred_matched


# ===============================================================================================================


# ========================================================================================================
# Parameter order for each supported monitor type.
# Parameter order for each supported monitor type.
MONITOR_PARAMS = {
    "device_state": ["device", "state"],
    "wait_time": ["time", "device"],  # device is optional.
    "visual_state": ["observe"],
    "resource_ready": ["prepare"],
}


def monitor_to_text(monitor: dict[str, Any]) -> str:
    """
    Convert a monitor dictionary into text for similarity comparison.
    Only parameters defined for that monitor type are used, and they are appended in a fixed order.
    """
    # Evaluation step for this metric calculation.
    monitor_type = monitor.get("type")
    if monitor_type not in MONITOR_PARAMS:
        return ""

    parts = []
    # Iterate through parameter names defined for this monitor type.
    for key in MONITOR_PARAMS[monitor_type]:
        # Read the value for this parameter key.
        value = monitor.get(key)
        # Evaluation step for this metric calculation.
        if value is not None and str(value).strip() != "":
            parts.append(str(value).strip())

    # Join parameter values into comparison text.
    return " ".join(parts)


def simple_tokenize(text: str) -> list[str]:
    """
    Tokenizer helper confirmed to work in this evaluation script.
    """
    return re.findall(
        r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*(?:[%℃])?|[^\sA-Za-z0-9]",
        str(text)
    )


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Compute a simple Jaccard similarity without loading an additional model.
    This can be replaced by SciBERTScore if semantic similarity is needed later.
    """
    # Lowercase, tokenize, and deduplicate both texts.
    # Lowercase, tokenize, and deduplicate both texts.
    # Lowercase, tokenize, and deduplicate both texts.
    tokens1 = set(simple_tokenize(text1.lower()))
    tokens2 = set(simple_tokenize(text2.lower()))

    # If both token sets are empty, treat them as identical.
    if not tokens1 and not tokens2:
        return 1.0
    # If only one token set is empty, similarity is zero.
    if not tokens1 or not tokens2:
        return 0.0

    # Compute Jaccard similarity as shared tokens divided by all tokens.
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def build_similarity_matrix(
    gt_group: list[dict[str, Any]],
    pred_group: list[dict[str, Any]],
    sim_func=jaccard_similarity,
) -> np.ndarray:
    """
    Build the pairwise gt/pred similarity matrix for one monitor type.
    The matrix shape is [len(gt_group), len(pred_group)].
    """

    # Return an all-zero matrix or empty match when either group is empty.
    if not gt_group or not pred_group:
        return np.zeros((len(gt_group), len(pred_group)))

    # Create an m by n similarity matrix for non-empty groups.
    # Create an m by n similarity matrix for non-empty groups.
    sim_matrix = np.zeros((len(gt_group), len(pred_group)), dtype=float)

    # Iterate through each gt item and its row index.
    for i, gt_item in enumerate(gt_group):
        # Convert the gt item to comparison text.
        gt_text = monitor_to_text(gt_item)
        # Iterate through each pred item and its column index.
        for j, pred_item in enumerate(pred_group):
            # Convert the pred item to comparison text.
            pred_text = monitor_to_text(pred_item)
            # Store the text similarity in the matrix.
            sim_matrix[i, j] = sim_func(gt_text, pred_text)

    # Return the completed similarity matrix.
    return sim_matrix


def match_one_type(
    gt_group: list[dict[str, Any]],
    pred_group: list[dict[str, Any]],
    sim_func=jaccard_similarity,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Find the optimal one-to-one matching for gt and pred monitors with the same type.
    Returns matched gt and pred lists with aligned positions.
    """

    # Return an all-zero matrix or empty match when either group is empty.
    # Return an all-zero matrix or empty match when either group is empty.
    if not gt_group or not pred_group:
        return [], []

    # Evaluation step for this metric calculation.
    # Evaluation step for this metric calculation.
    # Evaluation step for this metric calculation.
    sim_matrix = build_similarity_matrix(gt_group, pred_group, sim_func=sim_func)

    # Hungarian matching minimizes cost, so convert similarity to cost with 1 - similarity.
    # Hungarian matching minimizes cost, so convert similarity to cost with 1 - similarity.
    cost_matrix = 1.0 - sim_matrix

    # Find the best one-to-one assignment with the Hungarian algorithm.
    # Find the best one-to-one assignment with the Hungarian algorithm.
    # Find the best one-to-one assignment with the Hungarian algorithm.
    # Find the best one-to-one assignment with the Hungarian algorithm.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Use selected row indices to collect matched gt items.
    # Use selected row indices to collect matched gt items.
    # matched_gt = [gt_group[0], gt_group[1]]
    matched_gt = [gt_group[i] for i in row_ind]
    # Use selected column indices to collect matched pred items.
    matched_pred = [pred_group[j] for j in col_ind]

    return matched_gt, matched_pred


def group_monitors_by_type(monitors: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """
    Group monitor dictionaries by their type field.
    """
    # Create a defaultdict(list) so new labels start with an empty list.
    # Create a defaultdict(list) so new labels start with an empty list.
    # {"visual_state": [{}, {},...]
    #   "wait_time": [{}, {}, ...]
    # }
    grouped = defaultdict(list)
    # Iterate through every monitor dictionary.
    for item in monitors:
        monitor_type = item.get("type")  # Read the monitor type.
        # Add the monitor to the group for its type.
        grouped[monitor_type].append(item)
    return grouped


def match_all_monitors(
    gt_all_monitors: list[dict[str, Any]],
    pred_all_monitors: list[dict[str, Any]],
    sim_func=jaccard_similarity,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Match all supported monitor types.
    First group by type, then match within each group, and finally return aligned gt and pred lists.
    """

    # Group gt and pred monitors by monitor type.
    # {"visual_state": [{}, {},...],
    #   "wait_time": [{}, {}, ...],
    # }
    gt_grouped = group_monitors_by_type(gt_all_monitors)
    pred_grouped = group_monitors_by_type(pred_all_monitors)

    gt_matched = []
    pred_matched = []

    # Use a fixed monitor-type order for stable results.
    all_types = ["device_state", "wait_time", "visual_state", "resource_ready"]

    # Match each supported monitor type in order.
    for monitor_type in all_types:
        # Read the gt and pred group for this label, defaulting to an empty list.
        # Read the gt and pred group for this label, defaulting to an empty list.
        gt_group = gt_grouped.get(monitor_type, [])
        pred_group = pred_grouped.get(monitor_type, [])

        matched_gt_group, matched_pred_group = match_one_type(
            gt_group,
            pred_group,
            sim_func=sim_func,
        )

        gt_matched.extend(matched_gt_group)
        pred_matched.extend(matched_pred_group)

    return gt_matched, pred_matched
# ========================================================================================================


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
        ) = flatten_and_count(gt_stages)

        (
            pred_all_monitors,  # Flattened list of all predicted monitors.
            pred_all_subtasks,  # Flattened list of all predicted subtasks.
            pred_monitor_counts,  # Count occurrences of each predicted monitor type.
            pred_subtask_counts,  # Count occurrences of each predicted action type.
            pred_monitor_params,  # Parameter-combination counts for each predicted monitor type.
            pred_subtask_params,  # Parameter-combination counts for each predicted action type.
        ) = flatten_and_count(pred_stages)

        gt_matched, pred_matched = [], []

        # Use selected column indices to collect matched pred items.
        gt_matched1, pred_matched1, unmatched_pred = match_actions_for_tp(
            gt_all_subtasks,
            pred_all_subtasks
        )

        gt_matched2, pred_matched2 = match_all_actions(gt_all_subtasks, pred_all_subtasks)

        gt_matched.extend(gt_matched1)
        pred_matched.extend(pred_matched1)
        gt_matched.extend(gt_matched2)
        pred_matched.extend(pred_matched2)

        # for i in range(len(gt_matched)):
        #     print(gt_matched[i], '\n', pred_matched[i])
        #     # print(gt_matched[i])
        #     # print('\n')
        #
        # print(len(gt_matched), len(pred_matched))

        print("Current cell:", cell_name)

        print("\nParameter SciBertScore")

        sciBertScore_summary = compute_bertScore_for_all_matched_actions(
            gt_matched,
            pred_matched,
        )

        print("global_param_mean_SciBertScore:", round(sciBertScore_summary["global_param_mean_bert_score"], 3))
        # print("action_mean_SciBertScore:", sciBertScore_summary["action_mean_bert_score"])
        print("action_sum_all_param_score:", round(sciBertScore_summary["sum_all_param_score"], 3))
        print("action_count_params:", sciBertScore_summary["count_params"])

        file_path = "./result/scibert_action.xlsx"
        old_df = pd.read_excel(file_path)
        # [gpt-5.3, qwen3-max, qwen-max, deepseek-v3.2, deepseek-r1, Kimi-K2, glm-4.5]
        new_row = pd.DataFrame([{
            "model_name": "qwen3-max",
            "experiment": cell_name,
            "sum_scibertscore": round(sciBertScore_summary["sum_all_param_score"], 3),
            "count_params": sciBertScore_summary["count_params"],
        }])
        # df = pd.concat([old_df, new_row], ignore_index=True)
        if old_df.empty:
            df = new_row.copy().reset_index(drop=True)
        else:
            df = pd.concat([old_df, new_row], ignore_index=True)
        df.to_excel(file_path, index=False)
        print(cell_name, "appended successfully\n\n")


        print("Monitor Parameter SciBertScore")
        # Match monitor pairs before monitor-parameter evaluation.
        gt_matched, pred_matched = match_all_monitors(gt_all_monitors, pred_all_monitors)

        # for i in range(len(gt_matched)):
        #     print("gt:", gt_matched[i], '\n', "pred:", pred_matched[i])
        #     print('\n')
        #
        # # print(len(gt_matched), len(pred_all_monitors), len(gt_all_monitors))

        sciBertScore_summary2 = compute_bertScore_for_all_matched_monitors(
            gt_matched,
            pred_matched,
        )

        print("global_param_mean_SciBertScore:", round(sciBertScore_summary2["global_param_mean_bert_score"], 3))
        print("monitor_sum_all_param_scores:", round(sciBertScore_summary2["sum_all_param_scores"], 3))
        print("monitor_count_params:", sciBertScore_summary2["count_params"])

        file_path2 = "./result/scibert_monitor.xlsx"
        old_df = pd.read_excel(file_path2)
        # [gpt-5.3, qwen3-max, qwen-max, deepseek-v3.2, deepseek-r1, Kimi-K2, glm-4.5]
        new_row = pd.DataFrame([{
            "model_name": "qwen3-max",
            "experiment": cell_name,
            "sum_scibertscore": round(sciBertScore_summary2["sum_all_param_scores"], 3),
            "count_params": sciBertScore_summary2["count_params"],
        }])
        # df = pd.concat([old_df, new_row], ignore_index=True)
        if old_df.empty:
            df = new_row.copy().reset_index(drop=True)
        else:
            df = pd.concat([old_df, new_row], ignore_index=True)
        df.to_excel(file_path2, index=False)
        print(cell_name, "appended successfully\n\n")


if __name__ == "__main__":
    main()
