from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# 1) Hardcoded runtime settings
# -----------------------------
BASIC_PATH = "/mnt/hdd/huggingface/hub"
FOLDER_NAME = "UnifoLM_G1_Brainco_Dataset"

# IMPORTANT: this must include the "datasets--" prefix.
DATASET_NAME = "datasets--unitreerobotics--G1_Brainco_GraspOreo_Dataset"

# Plot first N episodes only.
MAX_EPISODES_TO_PLOT = 7

# Rolling STD window size for movement annotation.
STD_WINDOW_SIZE = 21

# Label smoothing window (odd number recommended).
LABEL_SMOOTH_WINDOW = 31

# Minimum segment length to avoid over-fragmented annotations.
MIN_SEGMENT_LENGTH = 20

# Visualization filter: show joint only if it dominated enough frames.
MIN_DOMINANT_FRAMES_TO_PLOT = 20
MIN_DOMINANT_RATIO_TO_PLOT = 0.05

# Segment highlight settings: thicken only for sufficiently long segments.
MIN_HIGHLIGHT_SEGMENT_LEN = 12
HIGHLIGHT_LINEWIDTH = 2.2
HIGHLIGHT_ALPHA = 0.95

# Output settings
DATASET_TAG = "g1_with_brainco_hand"
ALGORITHM_TAG = "rolling_std_dominant_joint"
ANNOTATED_CSV_BASE_DIR = Path(
    "/home/taeung/g1_datasets_huggingface/joint_angle_graphs/g1_with_brainco_hand/csv files"
)
GRAPH_IMAGE_BASE_DIR = Path(
    "/home/taeung/g1_datasets_huggingface/joint_angle_graphs/g1_with_brainco_hand/graph_images"
)

# ----------------------------------
# 2) Dataset joint order (fixed spec in huggingface. 
# Lerobot form of g1_with_brainco_hand dataset has 26 joints, including 14 arm joints and 12 hand joints)
# ----------------------------------
DATASET_JOINT_ORDER = [
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
    "kLeftHandThumb",
    "kLeftHandThumbAux",
    "kLeftHandIndex",
    "kLeftHandMiddle",
    "kLeftHandRing",
    "kLeftHandPinky",
    "kRightHandThumb",
    "kRightHandThumbAux",
    "kRightHandIndex",
    "kRightHandMiddle",
    "kRightHandRing",
    "kRightHandPinky",
]


# ---------------------------------
# 3) Joint groups for visualization. Dictionary . 
# ---------------------------------
JOINT_GROUPS = {
    "body": [
        "kLeftHipPitch",
        "kLeftHipRoll",
        "kLeftHipYaw",
        "kLeftKnee",
        "kLeftAnklePitch",
        "kLeftAnkleRoll",
        "kRightHipPitch",
        "kRightHipRoll",
        "kRightHipYaw",
        "kRightKnee",
        "kRightAnklePitch",
        "kRightAnkleRoll",
    ],
    "left_leg": [
        "kLeftHipPitch",
        "kLeftHipRoll",
        "kLeftHipYaw",
        "kLeftKnee",
        "kLeftAnklePitch",
        "kLeftAnkleRoll",
    ],
    "right_leg": [
        "kRightHipPitch",
        "kRightHipRoll",
        "kRightHipYaw",
        "kRightKnee",
        "kRightAnklePitch",
        "kRightAnkleRoll",
    ],
    "left_arm": [
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
    ],
    "right_arm": [
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
    ],
    "left_hand": [
        "kLeftHandThumb",
        "kLeftHandThumbAux",
        "kLeftHandIndex",
        "kLeftHandMiddle",
        "kLeftHandRing",
        "kLeftHandPinky",
    ],
    "right_hand": [
        "kRightHandThumb",
        "kRightHandThumbAux",
        "kRightHandIndex",
        "kRightHandMiddle",
        "kRightHandRing",
        "kRightHandPinky",
    ],
}

# Groups used for movement annotation. 각 그룹에 이름을 붙임
ANNOTATION_GROUPS = {
    "left_arm": JOINT_GROUPS["left_arm"],
    "right_arm": JOINT_GROUPS["right_arm"],
    "left_hand": JOINT_GROUPS["left_hand"],
    "right_hand": JOINT_GROUPS["right_hand"],
}
# 라벨값 이름 지정
GROUP_TO_LABEL = {
    "left_arm": ["left", "manipulation"],
    "right_arm": ["right", "manipulation"],
    "left_hand": ["left", "grasping"],
    "right_hand": ["right", "grasping"],
}

# Create reverse mapping from joint name to its annotation group for quick lookup.
# ex) JOINT_TO_GROUP["kLeftShoulderPitch"] -> "left_arm", used for determining which group a dominant joint belongs to.
# ex) JOINT_TO_GROUP["kLeftHandIndex"] -> "left_hand" 같이 자신이 속한 그룹 이름을 반환함.
JOINT_TO_GROUP: dict[str, str] = {}
for _group_name, _joints in ANNOTATION_GROUPS.items():
    for _joint_name in _joints:
        JOINT_TO_GROUP[_joint_name] = _group_name

LABEL_TO_COLOR = {
    "left|manipulation": "#d6f5d6",
    "right|manipulation": "#d6e6ff",
    "left|grasping": "#ffe6cc",
    "right|grasping": "#ffd6e7",
}

# Display-only short tags for plotting text (data labels remain unchanged).
LABEL_TO_SHORT = {
    "left|manipulation": "lm",
    "right|manipulation": "rm",
    "left|grasping": "lg",
    "right|grasping": "rg",
}

SHORT_LABEL_GUIDE = "\n".join(
    [
        "lm: left manipulation",
        "rm: right manipulation",
        "lg: left grasping",
        "rg: right grasping",
    ]
)

# finding parquet file under snapshots/*/data/** is more robust than hardcoding the full path
# ex) parquet file path: /home/taeung/g1_datasets_huggingface/datasets--unitreerobotics--G1_Brainco_GraspOreo_Dataset/snapshots/0/data/part-00000-xxxx.parquet
def find_parquet_file(dataset_root: Path) -> Path:
    """Find a parquet file under snapshots/*/data/**."""
    candidates = sorted(dataset_root.glob("snapshots/*/data/**/*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet file found under: {dataset_root}")
    return candidates[-1]


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize and annotate robot parquet dataset")
    parser.add_argument(
        "input_parquet",
        nargs="?",
        help="Optional parquet file path. If omitted, use hardcoded dataset root fallback.",
    )
    return parser.parse_args()


def resolve_parquet_input(input_parquet: str | None) -> Path:
    """Resolve parquet input path from CLI arg first, then fallback to default dataset root."""
    if input_parquet:
        parquet_path = Path(input_parquet).expanduser().resolve()
        if not parquet_path.exists():
            raise FileNotFoundError(f"Input parquet file not found: {parquet_path}")
        if parquet_path.suffix.lower() != ".parquet":
            raise ValueError(f"Input file must be a .parquet file: {parquet_path}")
        return parquet_path

    dataset_root = Path(BASIC_PATH) / FOLDER_NAME / DATASET_NAME
    return find_parquet_file(dataset_root)

# ex) validate_annotation_groups() -> {"left_arm": [0,1,2,3,4,5,6], "right_arm": [7,8,9,10,11,12,13], "left_hand": [14,15,16,17,18,19], "right_hand": [20,21,22,23,24,25]}, body
def validate_annotation_groups() -> dict[str, list[int]]:
    """Return index mapping for each annotation group in DATASET_JOINT_ORDER."""
    group_indices: dict[str, list[int]] = {}

    # 안전장치
    for group_name, joints in ANNOTATION_GROUPS.items():
        missing = [joint for joint in joints if joint not in DATASET_JOINT_ORDER]
        if missing:
            raise ValueError(
                f"Group '{group_name}' contains joints not in DATASET_JOINT_ORDER: {missing}"
            )
        # 각 그룹의 joint 이름을 DATASET_JOINT_ORDER에서 인덱스로 변환하여 저장
        group_indices[group_name] = [DATASET_JOINT_ORDER.index(joint) for joint in joints]
    return group_indices

# ex) add_group_state_columns(df, group_indices) -> df 왼쪽에 new columns 생김. ex) 'left_arm_state' containing numpy arrays of the corresponding joint values for each row.
# .copy() -> 원본 df를 변경하지 않고 새로운 df(out_df)를 반환하기 위해 사용됨. 각 그룹에 대해 'left_arm_state'와 같은 새로운 컬럼을 추가하는데, 이 컬럼은 'observation.state'에서 해당 그룹에 속한 joint들의 값만 추출하여 numpy array로 저장함.
def add_group_state_columns(df: pd.DataFrame, group_indices: dict[str, list[int]]) -> pd.DataFrame:
    """Add per-group state columns, e.g., 'left_arm_state', as numpy arrays."""
    out_df = df.copy()
    for group_name, indices in group_indices.items():
        col_name = f"{group_name}_state"
        out_df[col_name] = out_df["observation.state"].apply(
            lambda state: np.asarray(state, dtype=float)[indices]
        )
    return out_df

# ex) _compute_group_scores(episode_df, group_indices, window_size) -> df with columns like 'left_arm_score' containing rolling mean of std for the joints in that group.
# ex) group_indices = {"left_arm": [0,1,2,3,4,5,6], "right_arm": [7,8,9,10,11,12,13], "left_hand": [14,15,16,17,18,19], "right_hand": [20,21,22,23,24,25]}
# ex) episode_df["observation.state"]에서 각 행마다 observation.state 벡터를 numpy array로 변환하고 각 joint별로 rolling std를 계산하여 joint_std_series에 저장. 
def _compute_group_scores(
    episode_df: pd.DataFrame,
    group_indices: dict[str, list[int]],
    window_size: int,
) -> pd.DataFrame:
    """Compute rolling STD score per group for one episode."""
    state_matrix = np.vstack(episode_df["observation.state"].apply(np.asarray).values).astype(float)
    scores: dict[str, pd.Series] = {}
    for group_name, indices in group_indices.items():
        group_matrix = state_matrix[:, indices]
        joint_std_series = []
        for col_idx in range(group_matrix.shape[1]):
            series = pd.Series(group_matrix[:, col_idx], index=episode_df.index)
            rolling_std = series.rolling(
                window=window_size,
                center=True,
                min_periods=max(3, window_size // 3),
            ).std()
            rolling_std = rolling_std.fillna(method="bfill").fillna(method="ffill").fillna(0.0)
            joint_std_series.append(rolling_std)

        group_score = pd.concat(joint_std_series, axis=1).mean(axis=1)
        scores[group_name] = group_score
    return pd.DataFrame(scores, index=episode_df.index)

# ex) _compute_joint_scores(episode_df, window_size) -> df with columns for each joint containing rolling std values.
def _compute_joint_scores(episode_df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Compute rolling STD score for each joint in DATASET_JOINT_ORDER."""
    state_matrix = np.vstack(episode_df["observation.state"].apply(np.asarray).values).astype(float)
    joint_scores: dict[str, pd.Series] = {}

    for joint_idx, joint_name in enumerate(DATASET_JOINT_ORDER):
        series = pd.Series(state_matrix[:, joint_idx], index=episode_df.index)
        rolling_std = series.rolling(
            window=window_size,
            center=True,
            min_periods=max(3, window_size // 3),
        ).std()
        rolling_std = rolling_std.bfill().ffill().fillna(0.0)
        joint_scores[joint_name] = rolling_std

    return pd.DataFrame(joint_scores, index=episode_df.index)

#  _smooth_label_keys(label_keys, smooth_window) -> list of label keys after applying majority-vote smoothing in a sliding window.
# ex) label_keys: ['left|manipulation', 'left|manipulation', 'right|manipulation', 'right|manipulation', 'right|manipulation', 'left|manipulation']
def _smooth_label_keys(label_keys: list[str], smooth_window: int) -> list[str]:
    """Majority-vote smoothing in a sliding temporal window."""
    if len(label_keys) == 0:
        return []

    window = max(1, int(smooth_window))
    half = window // 2
    smoothed: list[str] = []

    for idx in range(len(label_keys)):
        lo = max(0, idx - half)
        hi = min(len(label_keys), idx + half + 1)
        votes = label_keys[lo:hi]
        counts = Counter(votes)
        smoothed.append(max(counts.items(), key=lambda kv: kv[1])[0])

    return smoothed

# _merge_short_segments(label_keys, min_segment_len) -> list of label keys after merging segments shorter than threshold into neighboring dominant segment.
# ex) merge_short_segments(['left|manipulation', 'left|manipulation', 'right|manipulation', 'right|manipulation', 'right|manipulation', 'left|manipulation'], min_segment_len=3) -> ['left|manipulation', 'left|manipulation', 'right|manipulation', 'right|manipulation', 'right|manipulation', 'right|manipulation'] (the last segment is merged into the previous one because it's shorter than 3)
def _merge_short_segments(label_keys: list[str], min_segment_len: int) -> list[str]:
    """Merge segments shorter than threshold into neighboring dominant segment."""
    if len(label_keys) == 0:
        return []

    min_len = max(1, int(min_segment_len))
    labels = label_keys[:]

    for _ in range(10):
        segments = _find_annotation_segments(labels)
        changed = False
        if len(segments) <= 1:
            break

        for seg_idx, (start_i, end_i, key) in enumerate(segments):
            seg_len = end_i - start_i
            if seg_len >= min_len:
                continue

            if seg_idx == 0:
                replacement = segments[seg_idx + 1][2]
            elif seg_idx == len(segments) - 1:
                replacement = segments[seg_idx - 1][2]
            else:
                prev_seg = segments[seg_idx - 1]
                next_seg = segments[seg_idx + 1]
                prev_len = prev_seg[1] - prev_seg[0]
                next_len = next_seg[1] - next_seg[0]
                replacement = prev_seg[2] if prev_len >= next_len else next_seg[2]

            for i in range(start_i, end_i):
                labels[i] = replacement
            changed = True

        if not changed:
            break

    return labels

# annotate_episode(episode_df, group_indices, window_size, smooth_window, min_segment_len) -> df copy with 'robot_move_annotation' column containing list of labels for each row.
# ex) annotate_episode(df_for_one_episode, group_indices, window_size=21, smooth_window=31, min_segment_len=20) -> df with new column 'robot_move_annotation' where each row has a list of labels like ['left', 'manipulation'] based on the dominant joint's group determined by rolling std.
# ex) 초기 df의 열이 7개였다면, 반환된 df는 원본 7개 열 + 'robot_move_annotation' + 'dominant_joint_by_std' 이렇게 2개의 열이 추가되어 총 9개 열이 됨. 
# 'robot_move_annotation' 열은 각 행마다 dominant joint가 속한 그룹의 라벨을 리스트 형태로 저장
# 'dominant_joint_by_std' 열은 각 행마다 rolling std가 가장 높은 joint의 이름을 저장함.
def annotate_episode(
    episode_df: pd.DataFrame,
    group_indices: dict[str, list[int]],
    window_size: int = STD_WINDOW_SIZE,
    smooth_window: int = LABEL_SMOOTH_WINDOW,
    min_segment_len: int = MIN_SEGMENT_LENGTH,
) -> pd.DataFrame:
    """
    Annotate one episode using rolling STD dominance.

    Returns a copy with column 'robot_move_annotation'.
    """
    if len(episode_df) == 0:
        out_df = episode_df.copy()
        out_df["robot_move_annotation"] = []
        return out_df

    _ = group_indices  # keep function signature modular for future model replacement
    joint_scores_df = _compute_joint_scores(episode_df, window_size=window_size)
    dominant_joints = joint_scores_df.idxmax(axis=1).tolist()

    raw_label_keys = [
        "|".join(GROUP_TO_LABEL[JOINT_TO_GROUP[joint_name]])
        for joint_name in dominant_joints
    ]
    smoothed_label_keys = _smooth_label_keys(raw_label_keys, smooth_window=smooth_window)
    final_label_keys = _merge_short_segments(smoothed_label_keys, min_segment_len=min_segment_len)

    out_df = episode_df.copy()
    out_df["robot_move_annotation"] = [key.split("|") for key in final_label_keys]
    out_df["dominant_joint_by_std"] = dominant_joints
    return out_df

# annotate_dataframe_by_episode(df, group_indices, window_size, smooth_window, min_segment_len) -> df with 'robot_move_annotation' column added for each episode.
# ex) annotate_dataframe_by_episode(df, group_indices, window_size=21, smooth_window=31, min_segment_len=20) -> df with new column 'robot_move_annotation' where each row has a list of labels like ['left', 'manipulation'] based on the dominant joint's group determined by rolling std, applied episode-wise.
def annotate_dataframe_by_episode(
    df: pd.DataFrame,
    group_indices: dict[str, list[int]],
    window_size: int = STD_WINDOW_SIZE,
    smooth_window: int = LABEL_SMOOTH_WINDOW,
    min_segment_len: int = MIN_SEGMENT_LENGTH,
) -> pd.DataFrame:
    """Apply episode-wise annotation and preserve the main DataFrame."""
    annotated_parts: list[pd.DataFrame] = []
    for _, ep in df.groupby("episode_index", sort=True):
        annotated_parts.append(
            annotate_episode(
                ep,
                group_indices=group_indices,
                window_size=window_size,
                smooth_window=smooth_window,
                min_segment_len=min_segment_len,
            )
        )

    if not annotated_parts:
        out = df.copy()
        out["robot_move_annotation"] = []
        out["dominant_joint_by_std"] = []
        return out

    return pd.concat(annotated_parts, axis=0).sort_index().reset_index(drop=True)

# _label_key(label) -> string key by joining label list with '|', e.g., ['left', 'manipulation'] -> 'left|manipulation'
def _label_key(label: list[str]) -> str:
    return "|".join(label)

# _find_annotation_segments(label_keys) -> list of segments as (start_idx, end_idx, label_key), where end_idx is exclusive. 
# ex) _find_annotation_segments(['left|manipulation', 'left|manipulation', 'right|manipulation', 'right|manipulation', 'right|manipulation', 'left|manipulation']) -> [(0, 2, 'left|manipulation'), (2, 5, 'right|manipulation'), (5, 6, 'left|manipulation')]
def _find_annotation_segments(label_keys: list[str]) -> list[tuple[int, int, str]]:
    """Return segments as (start_idx, end_idx, label_key), end_idx is exclusive."""
    if len(label_keys) == 0:
        return []
    segments: list[tuple[int, int, str]] = []
    start = 0
    current = label_keys[0]
    for idx in range(1, len(label_keys)):
        if label_keys[idx] != current:
            segments.append((start, idx, current))
            start = idx
            current = label_keys[idx]
    segments.append((start, len(label_keys), current))
    return segments


def _build_csv_output_path(output_dir: Path) -> Path:
    """Build a unique CSV filename with date + daily run index + algorithm tag."""
    date_str = datetime.now().strftime("%Y%m%d")
    candidates = list(output_dir.glob(f"{DATASET_TAG}_{date_str}_run*_{ALGORITHM_TAG}.csv"))

    run_indices: list[int] = []
    regex = re.compile(
        rf"^{re.escape(DATASET_TAG)}_{date_str}_run(\d+)_{re.escape(ALGORITHM_TAG)}\.csv$"
    )
    for path in candidates:
        match = regex.match(path.name)
        if match:
            run_indices.append(int(match.group(1)))

    next_run_idx = (max(run_indices) + 1) if run_indices else 1
    file_name = f"{DATASET_TAG}_{date_str}_run{next_run_idx:02d}_{ALGORITHM_TAG}.csv"
    return output_dir / file_name


def _dataset_output_dir_name(dataset_name: str) -> str:
    """Extract a clean dataset folder name from huggingface-style dataset path."""
    # Example: datasets--unitreerobotics--G1_Brainco_GraspOreo_Dataset -> G1_Brainco_GraspOreo_Dataset
    return dataset_name.split("--")[-1] if "--" in dataset_name else dataset_name


def save_annotated_dataframe_csv(df_annotated: pd.DataFrame, output_dir: Path) -> Path:
    """Save annotated DataFrame to CSV and return saved path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _build_csv_output_path(output_dir)
    df_annotated.to_csv(output_path, index=False, encoding="utf-8")
    return output_path

# visualize_annotated_episodes(df, max_episodes_to_plot) -> plot of joint values with annotation overlays for up to max episodes.
# ex) visualize_annotated_episodes(df, max_episodes_to_plot=6) -> shows plots for the first 6 episodes with joint values and colored segments based on 'robot_move_annotation'.
def visualize_annotated_episodes(
    df: pd.DataFrame,
    max_episodes_to_plot: int = MAX_EPISODES_TO_PLOT,
) -> None:
    """Plot joint values and annotation overlays for up to max episodes."""
    grouped_episodes = df.groupby("episode_index", sort=True)
    episode_ids = sorted(grouped_episodes.groups.keys())[:max_episodes_to_plot]
    if not episode_ids:
        raise ValueError("No episodes found for visualization.")

    fig, axes = plt.subplots(
        nrows=len(episode_ids),
        ncols=1,
        figsize=(14, 3.3 * len(episode_ids)),
        squeeze=False,
    )

    for row_idx, episode_id in enumerate(episode_ids):
        episode_data = grouped_episodes.get_group(episode_id)
        x = (
            episode_data["frame_index"].to_numpy()
            if "frame_index" in episode_data.columns
            else np.arange(len(episode_data), dtype=int)
        )

        annotation_keys = [
            _label_key(label) if isinstance(label, list) else str(label)
            for label in episode_data["robot_move_annotation"].tolist()
        ]
        segments = _find_annotation_segments(annotation_keys)

        # Plot original per-joint values (pre-aggregation) for all annotation joints.
        state_matrix = np.vstack(episode_data["observation.state"].apply(np.asarray).values).astype(float)
        joint_scores_df = _compute_joint_scores(episode_data, window_size=STD_WINDOW_SIZE)

        # Keep only joints that actually influenced max-STD decision enough times.
        dominant_counts = joint_scores_df.idxmax(axis=1).value_counts()
        min_required = max(
            MIN_DOMINANT_FRAMES_TO_PLOT,
            int(len(episode_data) * MIN_DOMINANT_RATIO_TO_PLOT),
        )
        selected_joints = {
            joint_name
            for joint_name, cnt in dominant_counts.items()
            if int(cnt) >= min_required
        }

        # Also keep joints that dominate long annotation segments for visual emphasis.
        long_segment_joints: set[str] = set()
        if "dominant_joint_by_std" in episode_data.columns:
            dominant_seq = episode_data["dominant_joint_by_std"].tolist()
            for start_i, end_i, _ in segments:
                if (end_i - start_i) < MIN_HIGHLIGHT_SEGMENT_LEN:
                    continue
                segment_joints = dominant_seq[start_i:end_i]
                if not segment_joints:
                    continue
                long_segment_joints.add(Counter(segment_joints).most_common(1)[0][0])
        selected_joints.update(long_segment_joints)

        # Fallback: always show at least the most dominant joint.
        if not selected_joints and len(dominant_counts) > 0:
            selected_joints.add(str(dominant_counts.index[0]))

        ax = axes[row_idx, 0]
        line_by_joint: dict[str, object] = {}
        for j_idx, joint_name in enumerate(DATASET_JOINT_ORDER):
            if joint_name not in selected_joints:
                continue
            (line_obj,) = ax.plot(x, state_matrix[:, j_idx], linewidth=0.9, alpha=0.55, label=joint_name)
            line_by_joint[joint_name] = line_obj

        for seg_idx, (start_i, end_i, key) in enumerate(segments):
            x_start = x[start_i]
            x_end = x[end_i - 1]
            color = LABEL_TO_COLOR.get(key, "#f0f0f0")
            ax.axvspan(x_start, x_end, color=color, alpha=0.28)

            # Emphasize segment-level dominant joint only for sufficiently long segments.
            if "dominant_joint_by_std" in episode_data.columns and (end_i - start_i) >= MIN_HIGHLIGHT_SEGMENT_LEN:
                segment_joints = episode_data["dominant_joint_by_std"].iloc[start_i:end_i].tolist()
                if segment_joints:
                    highlight_joint = Counter(segment_joints).most_common(1)[0][0]
                    if highlight_joint in DATASET_JOINT_ORDER:
                        hj_idx = DATASET_JOINT_ORDER.index(highlight_joint)
                        highlight_color = (
                            line_by_joint[highlight_joint].get_color()
                            if highlight_joint in line_by_joint
                            else "#111111"
                        )
                        ax.plot(
                            x[start_i:end_i],
                            state_matrix[start_i:end_i, hj_idx],
                            linewidth=HIGHLIGHT_LINEWIDTH,
                            alpha=HIGHLIGHT_ALPHA,
                            color=highlight_color,
                            zorder=4,
                        )

            if seg_idx > 0:
                ax.axvline(x_start, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

            if (end_i - start_i) >= 8:
                short_key = LABEL_TO_SHORT.get(key, key)
                ax.text(
                    (x_start + x_end) / 2.0,
                    ax.get_ylim()[1],
                    short_key,
                    fontsize=8,
                    rotation=0,
                    va="top",
                    ha="center",
                    alpha=0.85,
                )

        ax.set_title(
            f"Episode {episode_id} g1_with_brainco_hand (max STD joint based)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Frame Index" if "frame_index" in episode_data.columns else "Timestep")
        ax.set_ylabel("Joint Value")
        ax.grid(True, linestyle="--", alpha=0.45)

        # Show compact short-label guide for better readability.
        ax.text(
            1.01,
            0.02,
            SHORT_LABEL_GUIDE,
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
        )

        if row_idx == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=2, fontsize=7)

    plt.tight_layout()

    # If running in a non-interactive environment (e.g., script or headless server), save the figure instead of showing it.
    # ex) linux server에서 실행할 때 plt.show() 대신 figures 폴더에 annotated_episodes_preview.png로 저장함. 
    backend_name = plt.get_backend().lower()
    if "agg" in backend_name:
        output_dir = GRAPH_IMAGE_BASE_DIR / _dataset_output_dir_name(DATASET_NAME)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "annotated_episodes_preview.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"[INFO] non-interactive backend detected. saved figure: {output_path}")
        plt.close(fig)
    else:
        plt.show()


def save_joint_group_episode_previews(
    df: pd.DataFrame,
    output_dir: Path,
    max_episodes_to_plot: int = MAX_EPISODES_TO_PLOT,
) -> list[Path]:
    """Save 4 per-group episode preview plots (left/right arm/hand)."""
    grouped_episodes = df.groupby("episode_index", sort=True)
    episode_ids = sorted(grouped_episodes.groups.keys())[:max_episodes_to_plot]
    if not episode_ids:
        raise ValueError("No episodes found for group preview visualization.")

    output_dir.mkdir(parents=True, exist_ok=True)
    group_names = ["left_arm", "left_hand", "right_arm", "right_hand"]
    saved_paths: list[Path] = []

    for group_name in group_names:
        joint_names = JOINT_GROUPS[group_name]
        fig, axes = plt.subplots(
            nrows=len(episode_ids),
            ncols=1,
            figsize=(14, 3.1 * len(episode_ids)),
            squeeze=False,
        )

        for row_idx, episode_id in enumerate(episode_ids):
            episode_data = grouped_episodes.get_group(episode_id)
            x = (
                episode_data["frame_index"].to_numpy()
                if "frame_index" in episode_data.columns
                else np.arange(len(episode_data), dtype=int)
            )
            state_matrix = np.vstack(episode_data["observation.state"].apply(np.asarray).values).astype(float)
            ax = axes[row_idx, 0]

            for joint_name in joint_names:
                joint_idx = DATASET_JOINT_ORDER.index(joint_name)
                ax.plot(x, state_matrix[:, joint_idx], linewidth=1.0, alpha=0.85, label=joint_name)

            ax.set_title(
                f"Episode {episode_id} {group_name} joint values",
                fontsize=10,
                fontweight="bold",
            )
            ax.set_xlabel("Frame Index" if "frame_index" in episode_data.columns else "Timestep")
            ax.set_ylabel("Joint Value")
            ax.grid(True, linestyle="--", alpha=0.4)

            if row_idx == 0:
                ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1, fontsize=7)

        plt.tight_layout()
        file_path = output_dir / f"{group_name}_episodes_preview.png"
        fig.savefig(file_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(file_path)

    return saved_paths





def main() -> None:
    args = parse_cli_args()

    # Load dataset
    parquet_file_path = resolve_parquet_input(args.input_parquet)
    print(f"[INFO] parquet file: {parquet_file_path}")
    
    # Read the parquet file into a DataFrame and check for required columns.
    df = pd.read_parquet(parquet_file_path)
    print(f"[INFO] Downloaded dataset --> loaded rows={len(df)}, cols={len(df.columns)}")
    print()

    # define needed columns and check existence in the DataFrame.
    required_columns = {"observation.state", "episode_index"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    # ex) validate_annotation_groups() -> {"left_arm": [0,1,2,3,4,5,6], "right_arm": [7,8,9,10,11,12,13], "left_hand": [14,15,16,17,18,19], "right_hand": [20,21,22,23,24,25]}, body
    group_indices = validate_annotation_groups()

    # ex) annotate_dataframe_by_episode(df, group_indices, window_size=21, smooth_window=31, min_segment_len=20) 
    # df_annotated의 열은 원본 df의 열 + 'robot_move_annotation' + 'dominant_joint_by_std' 이렇게 2개가 추가됨.
    # 추가된 열1: 'robot_move_annotation' 열은 각 행마다 dominant joint가 속한 그룹의 라벨을 리스트 형태로 저장함. 예를 들어, ['left', 'manipulation'] 또는 ['right', 'grasping'] 같은 값이 들어갈 수 있음.
    # 추가된 열2: 'dominant_joint_by_std' 열은 각 행마다 rolling std가 가장 높은 joint의 이름을 저장함. 예를 들어, 'kLeftShoulderPitch' 또는 'kRightHandIndex' 같은 joint 이름이 들어갈 수 있음.    
    df_annotated = annotate_dataframe_by_episode(
        df,
        group_indices=group_indices,
        window_size=STD_WINDOW_SIZE,
        smooth_window=LABEL_SMOOTH_WINDOW,
        min_segment_len=MIN_SEGMENT_LENGTH,
    )

    print()
    print(f"[INFO] after annotation --> rows={len(df_annotated)}, cols={len(df_annotated.columns)}")
    print()
    print(f"[INFO] added column_1 name: {df_annotated.columns[-2]}")
    print(f"[INFO] added column_2 name: {df_annotated.columns[-1]}")
    print()
    print(
        "[INFO] robot_move_annotation sample(top 3): "
        f"{df_annotated['robot_move_annotation'].head(3).tolist()}"
    )
    print(
        "[INFO] dominant_joint_by_std sample(top 3): "
        f"{df_annotated['dominant_joint_by_std'].head(3).tolist()}"
    )
    print()

    if args.input_parquet:
        dataset_output_name = parquet_file_path.stem
    else:
        dataset_output_name = _dataset_output_dir_name(DATASET_NAME)
    csv_output_dir = ANNOTATED_CSV_BASE_DIR / dataset_output_name
    graph_output_dir = GRAPH_IMAGE_BASE_DIR / dataset_output_name

    csv_output_path = save_annotated_dataframe_csv(df_annotated, csv_output_dir)
    print(f"[INFO] saved annotated dataframe csv: {csv_output_path}")
    print()
    print(f"[INFO] saved csv --> rows={len(df_annotated)}, cols={len(df_annotated.columns)}")

    group_preview_paths = save_joint_group_episode_previews(
        df_annotated,
        output_dir=graph_output_dir,
        max_episodes_to_plot=MAX_EPISODES_TO_PLOT,
    )
    for p in group_preview_paths:
        print(f"[INFO] saved group preview image: {p}")

    print()
    visualize_annotated_episodes(df_annotated, max_episodes_to_plot=MAX_EPISODES_TO_PLOT)


if __name__ == "__main__":
    main()