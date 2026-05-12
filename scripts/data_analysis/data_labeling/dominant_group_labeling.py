"""
dominant_group_labeling.py
────────────────────────────────────────────────────────────────────
목적 : 정규화된 관절 데이터에 Rolling STD + 3-Sigma 임계값을 적용하여
       에피소드별 "Dominant Movement Group" 레이블을 산출하고 시각화한다.

처리 순서 :
    parquet → 관절각 추출 → 정규화(joint_limit_range 계열)
    → Rolling STD → Group Score = 그룹 내 max STD
    → 보정 구간(첫 30프레임) 통계로 임계값 산출
    → Raw 레이블 지정 (la / ra / lh / rh / lm / st)
    → Smooth 레이블 (Majority Vote)
    → 그래프 저장

단축 레이블:
    la : left_arm        ra : right_arm
    lh : left_hand       rh : right_hand
    lm : locomotion      st : stop (임계값 미달)
────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# ─── sys.path 설정 ────────────────────────────────────────────────────────────
# 이 파일 위치: scripts/data_analysis/data_labeling/
# 상위:         scripts/data_analysis/   → parents[1]
_DATA_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_DATA_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_ANALYSIS_DIR))

from stop_state_std_per_each_joint import (  # noqa: E402
    FPS,
    UPPER_BODY_JOINT_ORDER_BRAINCO,
    UPPER_BODY_JOINT_ORDER_DEX3,
    WBT_BODY_JOINT_ORDER,
    WBT_HAND_ORDER_BRAINCO,
    WBT_HAND_ORDER_INSPIRE,
    extract_joint_matrix,
)
from data_normalization.joint_limit_range_normalization import (  # noqa: E402
    apply_limit_range_normalization,
)
from data_normalization.joint_limit_range_0to1_normalization import (  # noqa: E402
    normalize as limit_range_0to1_normalize,
)

# ─── 출력 경로 ───────────────────────────────────────────────────────────────
GRAPH_ROOT          = Path("/home/taeung/g1_datasets_huggingface/graphs")
RAW_OUTPUT_DIR      = GRAPH_ROOT / "raw_labeling"
SMOOTHED_OUTPUT_DIR = GRAPH_ROOT / "smoothed_labeling"

# ─── 하이퍼파라미터 ──────────────────────────────────────────────────────────
X_TICK_SECS   = 3.0
CALIB_FRAMES  = 30   # 보정 구간 (에피소드 시작 1초, 정지 상태)

# ─── 그룹 단축 레이블 / 색상 ─────────────────────────────────────────────────
GROUP_SHORT = {
    "left_arm":   "la",
    "right_arm":  "ra",
    "left_hand":  "lh",
    "right_hand": "rh",
    "locomotion": "lm",
}
STOP_LABEL = "st"

GROUP_COLORS = {
    "la": "#cce5ff",   # 파랑  (left arm)
    "ra": "#ffd6d6",   # 빨강  (right arm)
    "lh": "#d4edda",   # 초록  (left hand)
    "rh": "#ffe8cc",   # 주황  (right hand)
    "lm": "#e8d6ff",   # 보라  (locomotion)
    "st": "#eeeeee",   # 회색  (stop)
}
GROUP_LINE_COLORS = {
    "left_arm":   "#1565C0",
    "right_arm":  "#c62828",
    "left_hand":  "#2e7d32",
    "right_hand": "#e65100",
    "locomotion": "#6a1b9a",
}


# ════════════════════════════════════════════════════════════════════
# ① 데이터셋 이름 추출
# ════════════════════════════════════════════════════════════════════
def extract_dataset_name(parquet_path: Path) -> str:
    """
    parquet 경로에서 'datasets--' 로 시작하는 디렉토리 이름을 추출한다.

    예)
    .../UnifoLM_G1_Brainco_Dataset/datasets--unitreerobotics--G1_Brainco_GraspOreo_Dataset/...
    → "datasets--unitreerobotics--G1_Brainco_GraspOreo_Dataset"
    """
    for part in parquet_path.parts:
        if part.startswith("datasets--"):
            return part
    # fallback: parquet 파일의 조부모 디렉토리명 사용
    return parquet_path.parents[2].name


# ════════════════════════════════════════════════════════════════════
# ② 데이터셋 정보 자동 감지
# ════════════════════════════════════════════════════════════════════
def detect_dataset_info(
    df: pd.DataFrame,
    hand_type_hint: str | None = None,
) -> tuple[str, str, list[str]]:
    """
    parquet DataFrame 의 열 구성으로 control_type / hand_type / joint_order 를 추론한다.

    WBT 데이터셋은 Brainco / Inspire 를 파일 경로만으로 구분할 수 없으므로
    --hand-type 인자로 명시해야 한다.

    Returns:
        (control_type, hand_type, joint_order)
    """
    if "observation.state" in df.columns:
        control_type = "Upper_body_control"
        sample       = np.asarray(df["observation.state"].iloc[0])
        n = len(sample)
        if n == 26:
            return "Upper_body_control", "g1_with_brainco", list(UPPER_BODY_JOINT_ORDER_BRAINCO)
        elif n == 28:
            return "Upper_body_control", "g1_with_dex3",    list(UPPER_BODY_JOINT_ORDER_DEX3)
        else:
            raise ValueError(f"알 수 없는 상체 관절 수: {n}")

    elif "observation.state.robot_q_current" in df.columns:
        if hand_type_hint is None:
            raise ValueError(
                "WBT 데이터셋입니다. --hand-type 을 명시해야 합니다.\n"
                "  선택지: g1_with_brainco | g1_with_inspire"
            )
        hand_type = hand_type_hint
        if hand_type == "g1_with_brainco":
            return "WBT", hand_type, list(WBT_BODY_JOINT_ORDER + WBT_HAND_ORDER_BRAINCO)
        elif hand_type == "g1_with_inspire":
            return "WBT", hand_type, list(WBT_BODY_JOINT_ORDER + WBT_HAND_ORDER_INSPIRE)
        else:
            raise ValueError(f"알 수 없는 hand_type: {hand_type}")

    else:
        raise ValueError("알 수 없는 데이터셋 형식: 예상 열을 찾을 수 없습니다.")


# ════════════════════════════════════════════════════════════════════
# ② 레이블링 그룹 정의
# ════════════════════════════════════════════════════════════════════
def get_labeling_groups(
    control_type: str,
    hand_type: str,
    joint_order: list[str],
) -> dict[str, list[str]]:
    """
    데이터셋 타입에 따라 레이블링에 사용할 그룹별 관절 목록을 반환한다.

    WBT       : locomotion / left_arm / right_arm / left_hand / right_hand
    상체 제어  : left_arm / right_arm / left_hand / right_hand
    """
    joint_set = set(joint_order)
    groups: dict[str, list[str]] = {}

    if control_type == "WBT":
        # locomotion = legs (0-11) + waist (12-14)
        loco = [j for j in WBT_BODY_JOINT_ORDER[:15] if j in joint_set]
        la   = [j for j in WBT_BODY_JOINT_ORDER[15:22] if j in joint_set]
        ra   = [j for j in WBT_BODY_JOINT_ORDER[22:29] if j in joint_set]
        hand_src = WBT_HAND_ORDER_BRAINCO if hand_type == "g1_with_brainco" \
                   else WBT_HAND_ORDER_INSPIRE
        lh = [j for j in hand_src[:6]  if j in joint_set]
        rh = [j for j in hand_src[6:]  if j in joint_set]
        if loco: groups["locomotion"] = loco
        if la:   groups["left_arm"]   = la
        if ra:   groups["right_arm"]  = ra
        if lh:   groups["left_hand"]  = lh
        if rh:   groups["right_hand"] = rh
    else:
        if hand_type == "g1_with_dex3":
            src = UPPER_BODY_JOINT_ORDER_DEX3
            la, ra, lh, rh = src[:7], src[7:14], src[14:21], src[21:28]
        else:
            src = UPPER_BODY_JOINT_ORDER_BRAINCO
            la, ra, lh, rh = src[:7], src[7:14], src[14:20], src[20:26]
        for name, joints in [
            ("left_arm",  la), ("right_arm",  ra),
            ("left_hand", lh), ("right_hand", rh),
        ]:
            valid = [j for j in joints if j in joint_set]
            if valid:
                groups[name] = valid

    return groups


# ════════════════════════════════════════════════════════════════════
# ③ 정규화
# ════════════════════════════════════════════════════════════════════
def apply_normalization(
    state_matrix: np.ndarray,
    normalize_method: str,
    joint_order: list[str],
    hand_type: str,
) -> np.ndarray:
    """선택한 방법으로 관절각 행렬을 정규화한다."""
    if normalize_method == "joint_limit_range":
        return apply_limit_range_normalization(state_matrix, joint_order, hand_type)
    elif normalize_method == "joint_limit_range_0to1":
        return limit_range_0to1_normalize(
            state_matrix, joint_order=joint_order, hand_type=hand_type
        )
    else:
        raise ValueError(f"지원하지 않는 normalize_method: {normalize_method}")


# ════════════════════════════════════════════════════════════════════
# ④ Rolling STD 계산
# ════════════════════════════════════════════════════════════════════
def compute_rolling_std(state_matrix: np.ndarray, window: int) -> np.ndarray:
    """
    관절별 Rolling STD 를 계산한다. (N_frames, N_joints)
    앞부분 NaN 은 bfill 로 채운다.
    """
    df = pd.DataFrame(state_matrix)
    rolling_std = (
        df.rolling(window=window, center=True, min_periods=max(3, window // 3))
        .std()
        .bfill()
        .ffill()
        .fillna(0.0)
    )
    return rolling_std.to_numpy(dtype=float)


# ════════════════════════════════════════════════════════════════════
# ⑤ 그룹 스코어 계산
# ════════════════════════════════════════════════════════════════════
def compute_group_scores(
    rolling_std_matrix: np.ndarray,
    groups: dict[str, list[str]],
    joint_order: list[str],
) -> dict[str, np.ndarray]:
    """
    그룹 스코어 = 해당 그룹 내 관절들의 Max Rolling STD.
    Returns: {group_name: (N_frames,) ndarray}
    """
    joint_to_idx = {name: i for i, name in enumerate(joint_order)}
    scores: dict[str, np.ndarray] = {}
    for group_name, joints in groups.items():
        indices = [joint_to_idx[j] for j in joints if j in joint_to_idx]
        if not indices:
            continue
        scores[group_name] = rolling_std_matrix[:, indices].max(axis=1)
    return scores


# ════════════════════════════════════════════════════════════════════
# ⑥-A 데이터셋 전체 정지 구간 정규화 STD 중앙값 계산
# ════════════════════════════════════════════════════════════════════
def compute_baseline_std_normalized(
    df: pd.DataFrame,
    control_type: str,
    hand_type: str,
    joint_order: list[str],
    normalize_method: str,
    calib_frames: int = CALIB_FRAMES,
) -> tuple[np.ndarray, int]:
    """
    stop_state_std_per_each_joint.py 의 compute_stop_state_median_std() 로직을
    '정규화 후 버전'으로 구현한다.

    전체 에피소드의 정지 구간(frame_index 0 ~ calib_frames-1) 에서
    ① 관절각 추출 → ② 정규화 적용 → ③ 관절별 STD 계산
    을 에피소드별로 수행하고, 각 관절의 중앙값(median)을 반환한다.

    임계값 산출에 사용되는 기준값이 정규화된 단위와 동일하므로
    그룹 스코어(Rolling STD of normalized data)와 직접 비교할 수 있다.

    Returns:
        baseline_std  : (N_joints,) — 정지 구간 정규화 STD 중앙값
        n_episodes    : 사용된 에피소드 수
    """
    stationary_df = df[df["frame_index"] <= calib_frames - 1]
    episode_stds: list[np.ndarray] = []

    for _, ep_df in stationary_df.groupby("episode_index"):
        if len(ep_df) < 2:
            continue
        state_matrix = extract_joint_matrix(ep_df, control_type, hand_type, joint_order)
        state_norm   = apply_normalization(state_matrix, normalize_method, joint_order, hand_type)
        episode_stds.append(state_norm.std(axis=0))   # (N_joints,)

    if not episode_stds:
        return np.zeros(len(joint_order)), 0

    all_stds     = np.array(episode_stds)             # (N_ep, N_joints)
    median_std   = np.median(all_stds, axis=0)        # (N_joints,)
    return median_std, len(episode_stds)


# ════════════════════════════════════════════════════════════════════
# ⑥-B 베이스라인 STD → 그룹별 임계값 변환
# ════════════════════════════════════════════════════════════════════
def compute_thresholds_from_baseline(
    baseline_std: np.ndarray,
    groups: dict[str, list[str]],
    joint_order: list[str],
    threshold_mult: float,
) -> dict[str, float]:
    """
    정규화된 정지 구간 STD 중앙값으로 그룹별 임계값을 계산한다.

    group_threshold = threshold_mult × max(baseline_std[그룹 내 관절])

    그룹 스코어 = 그룹 내 max Rolling STD 이므로,
    기준값도 그룹 내 max baseline STD 의 K배로 설정한다.
    """
    joint_to_idx = {name: i for i, name in enumerate(joint_order)}
    thresholds: dict[str, float] = {}
    for group_name, joints in groups.items():
        indices = [joint_to_idx[j] for j in joints if j in joint_to_idx]
        if not indices:
            continue
        group_baseline          = float(baseline_std[indices].max())
        thresholds[group_name]  = threshold_mult * group_baseline
    return thresholds


# ════════════════════════════════════════════════════════════════════
# ⑦ Raw 레이블 지정
# ════════════════════════════════════════════════════════════════════
def assign_raw_labels(
    group_scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> list[str]:
    """
    프레임별로 임계값을 초과하는 그룹 중 스코어가 가장 높은 그룹의
    단축 레이블을 반환한다. 임계값 초과 그룹이 없으면 "st".

    Returns: (N_frames,) 레이블 리스트
    """
    n_frames = next(iter(group_scores.values())).shape[0]
    labels: list[str] = []

    for t in range(n_frames):
        winner     = STOP_LABEL
        best_score = -np.inf
        for group_name, scores in group_scores.items():
            score = float(scores[t])
            thr   = thresholds.get(group_name, 0.0)
            if score > thr and score > best_score:
                best_score = score
                winner     = GROUP_SHORT.get(group_name, group_name)
        labels.append(winner)

    return labels


# ════════════════════════════════════════════════════════════════════
# ⑧ Smoothing (Majority Vote)
# ════════════════════════════════════════════════════════════════════
def smooth_labels(raw_labels: list[str], smooth_window: int) -> list[str]:
    """
    rolling majority vote (과반수 표결) 방식으로 레이블을 평활화한다.
    빠른 깜빡임(flickering) 을 제거한다.
    """
    half = smooth_window // 2
    smoothed: list[str] = []
    n = len(raw_labels)
    for i in range(n):
        window = raw_labels[max(0, i - half): min(n, i + half + 1)]
        smoothed.append(Counter(window).most_common(1)[0][0])
    return smoothed


# ════════════════════════════════════════════════════════════════════
# ⑨ 구간 분할
# ════════════════════════════════════════════════════════════════════
def find_segments(labels: list[str]) -> list[tuple[int, int, str]]:
    """
    연속된 동일 레이블을 하나의 구간으로 묶는다.
    Returns: [(start_idx, end_idx_exclusive, label), ...]
    """
    if not labels:
        return []
    segments: list[tuple[int, int, str]] = []
    start   = 0
    current = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != current:
            segments.append((start, i, current))
            start   = i
            current = labels[i]
    segments.append((start, len(labels), current))
    return segments


# ════════════════════════════════════════════════════════════════════
# ⑩ 시각화
# ════════════════════════════════════════════════════════════════════
def plot_episode_labels(
    state_normalized: np.ndarray,
    frame_seconds: np.ndarray,
    group_scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    labels: list[str],
    episode_index: int,
    normalize_method: str,
    label_type: str,
    output_dir: Path,
) -> None:
    """
    관절 상태(상단) + 그룹 스코어(하단) 2-패널 그래프를 저장한다.
    shaded region 으로 지배 그룹을 표현하고 단축 레이블 텍스트로 주석을 단다.
    """
    segments = find_segments(labels)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(18, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.3]},
    )

    # ── 상단: 정규화 관절 상태 ──────────────────────────────────────────────
    n_joints = state_normalized.shape[1]
    for j in range(n_joints):
        ax_top.plot(
            frame_seconds, state_normalized[:, j],
            linewidth=0.6, alpha=0.25, color="steelblue",
        )

    # 배경 shading + 단축 레이블 텍스트
    y_min, y_max = state_normalized.min(), state_normalized.max()
    y_range = y_max - y_min + 1e-6
    for start_i, end_i, lbl in segments:
        x_s = frame_seconds[start_i]
        x_e = frame_seconds[end_i - 1]
        ax_top.axvspan(x_s, x_e, color=GROUP_COLORS.get(lbl, GROUP_COLORS["st"]), alpha=0.45)
        # 단축 레이블 텍스트 (충분히 긴 구간만)
        if (end_i - start_i) >= 8:
            ax_top.text(
                (x_s + x_e) / 2, y_max + 0.03 * y_range,
                lbl, ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#333333",
            )

    ax_top.set_ylabel(f"state  [{normalize_method}]", fontsize=9)
    ax_top.set_title(
        f"[{label_type.upper()}]  Dominant Group Labeling  —  Episode {episode_index}",
        fontsize=11, fontweight="bold",
    )
    ax_top.xaxis.set_major_locator(MultipleLocator(X_TICK_SECS))
    ax_top.grid(True, linestyle=":", alpha=0.35)

    # ── 하단: 그룹 스코어 ───────────────────────────────────────────────────
    for group_name, scores in group_scores.items():
        short = GROUP_SHORT.get(group_name, group_name)
        ax_bot.plot(
            frame_seconds, scores,
            label=f"{short} ({group_name})",
            linewidth=1.1, alpha=0.85,
            color=GROUP_LINE_COLORS.get(group_name, "gray"),
        )

    # 임계값 수평선 (그룹별)
    for group_name, thr in thresholds.items():
        ax_bot.axhline(
            thr, color=GROUP_LINE_COLORS.get(group_name, "gray"),
            linestyle="--", linewidth=0.9, alpha=0.55,
        )

    # 배경 shading (상단과 동일)
    for start_i, end_i, lbl in segments:
        x_s = frame_seconds[start_i]
        x_e = frame_seconds[end_i - 1]
        ax_bot.axvspan(x_s, x_e, color=GROUP_COLORS.get(lbl, GROUP_COLORS["st"]), alpha=0.30)

    ax_bot.set_ylabel("Group Score\n(max rolling STD)", fontsize=9)
    ax_bot.set_xlabel("time (s)", fontsize=9)
    ax_bot.legend(loc="upper right", fontsize=7, ncol=3)
    ax_bot.xaxis.set_major_locator(MultipleLocator(X_TICK_SECS))
    ax_bot.grid(True, linestyle=":", alpha=0.35)

    # ── 공통 범례 (색상 → 레이블 의미) ─────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=GROUP_COLORS[v], edgecolor="gray",
                       label=f"{v}  ({k})")
        for k, v in GROUP_SHORT.items()
    ]
    legend_items.append(
        mpatches.Patch(facecolor=GROUP_COLORS["st"], edgecolor="gray", label="st  (stop)")
    )
    ax_top.legend(
        handles=legend_items, loc="upper left",
        fontsize=7, ncol=3, framealpha=0.85,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"episode_{episode_index}_{normalize_method}_{label_type}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [{label_type:8s}] {out_path}")


# ════════════════════════════════════════════════════════════════════
# ⑪ CLI
# ════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dominant Movement Group 레이블링 스크립트\n"
            "Rolling STD + 3-Sigma 임계값으로 에피소드별 지배 그룹을 결정하고 시각화한다."
        )
    )
    parser.add_argument(
        "--input_file", required=True, metavar="PATH",
        help="입력 parquet 파일 경로.",
    )
    parser.add_argument(
        "--normalize_method",
        choices=["joint_limit_range", "joint_limit_range_0to1"],
        default="joint_limit_range_0to1",
        help=(
            "정규화 방법.\n"
            "  joint_limit_range      : x / (Upper - Lower)\n"
            "  joint_limit_range_0to1 : (x - Lower) / (Upper - Lower)\n"
            "  default: joint_limit_range_0to1"
        ),
    )
    parser.add_argument(
        "--hand-type",
        choices=["g1_with_brainco", "g1_with_inspire"],
        default=None,
        metavar="HAND_TYPE",
        help=(
            "WBT 데이터셋에서만 필요. 상체 데이터셋은 자동 감지.\n"
            "  선택지: g1_with_brainco | g1_with_inspire"
        ),
    )
    parser.add_argument(
        "--threshold_mult", type=float, default=3.0, metavar="K",
        help="임계값 배수 K.  threshold = mu + K × sigma  (default: 3.0)",
    )
    parser.add_argument(
        "--window_size", type=int, default=30, metavar="FRAMES",
        help=f"Rolling STD 윈도우 크기 (프레임).  default: 30 ({30 / FPS:.1f}s at {FPS:.0f}fps)",
    )
    parser.add_argument(
        "--smooth_window", type=int, default=15, metavar="FRAMES",
        help="Majority Vote 평활화 윈도우 크기 (프레임).  default: 15",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=None, metavar="N",
        help="처리할 최대 에피소드 수.  default: 전체",
    )
    return parser.parse_args()


# ════════════════════════════════════════════════════════════════════
# ⑫ main
# ════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_args()
    parquet_path = Path(args.input_file).expanduser().resolve()

    if not parquet_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {parquet_path}")

    dataset_name = extract_dataset_name(parquet_path)
    print(f"[입력 파일]  {parquet_path}")
    print(f"[데이터셋]   {dataset_name}")
    print(f"[정규화]     {args.normalize_method}")
    print(f"[임계값 배수] K = {args.threshold_mult}")
    print(f"[Rolling STD 윈도우] {args.window_size} frames ({args.window_size / FPS:.2f}s)")
    print(f"[평활화 윈도우] {args.smooth_window} frames")

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    print("\n[1/4] 데이터 로딩 중...")
    df = pd.read_parquet(parquet_path)
    print(f"      shape = {df.shape},  열: {list(df.columns)}")

    # ── 데이터셋 정보 감지 ────────────────────────────────────────────────────
    print("\n[2/4] 데이터셋 정보 감지 중...")
    control_type, hand_type, joint_order = detect_dataset_info(df, args.hand_type)
    groups = get_labeling_groups(control_type, hand_type, joint_order)
    print(f"      control_type : {control_type}")
    print(f"      hand_type    : {hand_type}")
    print(f"      관절 수      : {len(joint_order)}")
    print(f"      레이블 그룹  : {list(groups.keys())}")

    # ── [3/4] 데이터셋 전체 정지 구간 정규화 베이스라인 STD 계산 ────────────────
    # stop_state_std_per_each_joint.py 와 동일한 방식으로
    # 모든 에피소드의 frame_index 0~29(1초) 구간 STD 중앙값을 정규화 단위로 계산
    print(f"\n[3/4] 정규화 베이스라인 STD 계산 중 "
          f"(frame_index 0~{CALIB_FRAMES - 1}, 전체 에피소드)...")
    baseline_std, n_calib_ep = compute_baseline_std_normalized(
        df, control_type, hand_type, joint_order, args.normalize_method
    )
    print(f"      사용 에피소드 수 : {n_calib_ep}")
    print(f"      관절별 baseline STD (정규화 단위, 중앙값):")
    for i, (name, val) in enumerate(zip(joint_order, baseline_std)):
        print(f"        {i:>3}  {name:<32}  {val:.8f}")

    # 그룹별 임계값 (전체 에피소드 기준, 에피소드 루프 밖에서 1회 계산)
    thresholds = compute_thresholds_from_baseline(
        baseline_std, groups, joint_order, args.threshold_mult
    )
    print(f"\n      그룹별 임계값 (K={args.threshold_mult} × max baseline STD):")
    for g, v in thresholds.items():
        print(f"        {GROUP_SHORT.get(g, g)} ({g:<12}) : {v:.8f}")

    # ── 에피소드 처리 ─────────────────────────────────────────────────────────
    episode_list = sorted(df["episode_index"].unique().tolist())
    if args.max_episodes is not None:
        episode_list = episode_list[: args.max_episodes]

    print(f"\n[4/4] 레이블 산출 + 그래프 저장  (총 {len(episode_list)}개 에피소드)")

    for ep_idx in episode_list:
        ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        if ep_df.empty:
            print(f"  [SKIP] episode {ep_idx}: 데이터 없음")
            continue

        n_frames      = len(ep_df)
        duration      = ep_df["frame_index"].max() / FPS
        frame_seconds = ep_df["frame_index"].to_numpy(dtype=float) / FPS
        print(f"\n  episode {ep_idx:>3} : {n_frames} frames ({duration:.1f}s)")

        # 관절각 추출
        state_matrix = extract_joint_matrix(ep_df, control_type, hand_type, joint_order)

        # 정규화
        state_norm = apply_normalization(
            state_matrix, args.normalize_method, joint_order, hand_type
        )

        # Rolling STD
        rolling_std = compute_rolling_std(state_norm, window=args.window_size)

        # 그룹 스코어
        group_scores = compute_group_scores(rolling_std, groups, joint_order)

        # 임계값은 데이터셋 전체 베이스라인에서 이미 계산됨 (thresholds 재사용)

        # Raw 레이블
        raw_labels      = assign_raw_labels(group_scores, thresholds)
        smoothed_labels = smooth_labels(raw_labels, args.smooth_window)

        # 레이블 분포 출력
        for ltype, lbls in [("raw", raw_labels), ("smoothed", smoothed_labels)]:
            dist = Counter(lbls)
            dist_str = "  ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
            print(f"      [{ltype:8s}] {dist_str}")

        # 시각화
        print(f"      → 그래프 저장 중...")
        raw_out_dir      = RAW_OUTPUT_DIR      / control_type / hand_type / dataset_name
        smoothed_out_dir = SMOOTHED_OUTPUT_DIR / control_type / hand_type / dataset_name
        for label_type, labels, out_dir in [
            ("raw",      raw_labels,      raw_out_dir),
            ("smoothed", smoothed_labels, smoothed_out_dir),
        ]:
            plot_episode_labels(
                state_normalized=state_norm,
                frame_seconds=frame_seconds,
                group_scores=group_scores,
                thresholds=thresholds,
                labels=labels,
                episode_index=ep_idx,
                normalize_method=args.normalize_method,
                label_type=label_type,
                output_dir=out_dir,
            )

    print(f"\n완료.")
    print(f"  Raw 레이블 그래프    : {RAW_OUTPUT_DIR / control_type / hand_type / dataset_name}")
    print(f"  Smooth 레이블 그래프 : {SMOOTHED_OUTPUT_DIR / control_type / hand_type / dataset_name}")


if __name__ == "__main__":
    main()
