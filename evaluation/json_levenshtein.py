import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any
import pandas as pd


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


def get_actions(
    gt_subtasks: List[Dict[str, Any]],
    pred_subtasks: List[Dict[str, Any]]
):
    """
    Extract action names from gt and pred subtasks in their original order.
    Returns two aligned action-name lists for sequence-level distance evaluation.
    """

    gt_matched = []
    pred_matched = []

    # Iterate through every gt subtask.
    for gt_item in gt_subtasks:
        gt_action = gt_item.get("action")  # Read the current gt action name.
        gt_matched.append(gt_action)  # Append the gt action name to the sequence.

    for pred_item in pred_subtasks:
        pred_action = pred_item.get("action")
        pred_matched.append(pred_action)

    return gt_matched, pred_matched


def levenshtein_distance(pred, gt):
    """
    Compute the Levenshtein edit distance between two function/action sequences.
    pred is the predicted sequence, and gt is the ground-truth sequence.
    """
    m = len(pred)
    n = len(gt)

    # dp[i][j] stores the minimum edit distance from pred[:i] to gt[:j].
    # In other words, it is the fewest edits needed to turn the first i pred items into the first j gt items.
    # Create a zero-filled (m + 1) x (n + 1) table; the extra row and column store the empty sequence.
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first column dp[i][0].
    # This means converting the first i pred items into an empty sequence.
    for i in range(m + 1):
        dp[i][0] = i  # A length-i pred prefix needs i deletions to become empty.
    # Initialize the first row dp[0][j].
    # This means converting an empty sequence into the first j gt items.
    for j in range(n + 1):
        dp[0][j] = j  # An empty sequence needs j insertions to match a length-j gt prefix.

    # Fill the dynamic-programming table; row 0 and column 0 have already been initialized.
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # If the current elements match, carry over dp[i-1][j-1].
            # The last elements are equal, so no extra edit is needed.
            if pred[i - 1] == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            # If the current elements differ, choose the cheapest delete, insert, or replace operation.
            else:
                # Delete: turn pred[:i-1] into gt[:j], then delete the final pred item.
                delete_cost = dp[i - 1][j] + 1
                # Insert: turn pred[:i] into gt[:j-1], then insert one item to reach gt[:j].
                insert_cost = dp[i][j - 1] + 1
                # Replace: turn pred[:i-1] into gt[:j-1], then replace the final pred item.
                replace_cost = dp[i - 1][j - 1] + 1
                dp[i][j] = min(delete_cost, insert_cost, replace_cost)

    # Return the bottom-right cell: the minimum insert/delete/replace count needed to adjust pred to gt.
    return dp[m][n]


def normalized_levenshtein_distance(pred, gt):
    """
    Compute normalized Levenshtein distance as L_d / N.
    N is the length of the ground-truth sequence.
    """
    if len(gt) == 0:
        raise ValueError("Ground-truth sequence length cannot be 0")

    ld = levenshtein_distance(pred, gt)
    ldn = ld / len(gt)
    return ld, ldn


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

        # Extract gt and pred action-name sequences.
        gt_matched, pred_matched = get_actions(
            gt_all_subtasks,
            pred_all_subtasks
        )
        # Optional diagnostic output.
        print("Number of gt actions:", len(gt_matched))

        print("Parameter Levenshtein Distance")
        ld, ldn = normalized_levenshtein_distance(pred_matched, gt_matched)
        print("Levenshtein distance:", ld)
        print("Normalized Levenshtein distance:", round(ldn, 3))

        file_path = "./result/Ldn.xlsx"
        old_df = pd.read_excel(file_path)
        # [gpt-5.3, qwen3-max, qwen-max, deepseek-v3.2, deepseek-r1, Kimi-K2, glm-4.5]
        new_row = pd.DataFrame([{
            "model_name": "qwen3-max",
            "experiment": cell_name,
            "Ld": ld,
            "gt_n": len(gt_matched),
        }])
        df = pd.concat([old_df, new_row], ignore_index=True)
        df.to_excel(file_path, index=False)
        print(cell_name, "appended successfully\n\n")


if __name__ == "__main__":
    main()
