"""
data_preprocessing_baseline_gaussian_filter.py
────────────────────────────────────────────────────────────────────
목적 : 로봇 텔레오퍼레이션 시계열 데이터에
       ① 가우시안 스무딩 필터 적용 후
       ② 선택한 정규화 방법을 적용하여 시각화한다.

처리 순서 :
    raw state → [Gaussian Filter] → [정규화] → 시각화

정규화 방법 (--norm-method 로 선택):
    baseline           : V_i(t) = Moving_STD / (Baseline_STD + ε)
    z_score            : (x - mean) / std  (관절별)
    min_max            : (x - min) / (max - min)  (관절별)
    moving_std_min_max : Moving_STD → min-max  (관절별)
    delta              : 프레임 간 변화량 / max|δ|  (관절별)
────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
from argparse import RawTextHelpFormatter
import importlib
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter1d

matplotlib.use("Agg")

# ─── stop_state_std_per_each_joint 모듈 import ───────────────────────────────
_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

# ─── visualize_1sec_std 모듈 import ──────────────────────────────────────────
_VIZ_DIR = Path(__file__).resolve().parents[2] / "graph_visualization"
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))

from visualize_1sec_std import log_and_plot_1sec_std  # noqa: E402

from stop_state_std_per_each_joint import (  # noqa: E402
    BASIC_PATH,
    FPS,
    STATIONARY_FRAMES,
    FOLDER_TO_DATASETS,
    ALL_FOLDER_NAMES,
    classify_control_type,
    classify_hand_type,
    validate_dataset_selection,
    resolve_snapshot_dir,
    find_first_parquet_file,
    get_joint_order,
    extract_joint_matrix,
    load_raw_dataframe,
    compute_stop_state_median_std,
)

# ─── 런타임 설정 ──────────────────────────────────────────────────────────────
FOLDER_NAME  = "UnifoLM_G1_Brainco_Dataset"
DATASET_NAME = "datasets--unitreerobotics--G1_Brainco_GraspOreo_Dataset"

MAX_EPISODES = 5      # 시각화할 최대 에피소드 수
STD_WINDOW   = 21     # Moving STD 윈도우 기본값 (baseline, moving_std_min_max 에서 사용)
X_TICK_SECS  = 3.0    # X축 눈금 간격 (초)

# ─── [Gaussian] 필터 하이퍼파라미터 ──────────────────────────────────────────
GAUSSIAN_SIGMA = 9.0  # 가우시안 커널 표준편차 (프레임 단위).
                      # 클수록 더 강한 평활화. 30fps 기준: sigma=9 ≈ 0.3s
                      # 권장 범위: 1~30

# ─── 정규화 방법 매핑 ─────────────────────────────────────────────────────────
# key   : --norm-method 에 입력하는 이름
# value : data_normalization/ 폴더 내 모듈 파일명
NORM_MODULE_MAP = {
    "baseline":           "baseline_normalization",
    "z_score":            "z_score_normalization",
    "min_max":            "min_max_normalization",
    "moving_std_min_max": "moving_std_min_max_normalization",
    "delta":              "delta_normalization",
    "joint_limit_range":      "joint_limit_range_normalization",
    "joint_limit_range_0to1": "joint_limit_range_0to1_normalization",
}
DEFAULT_NORM_METHOD = "baseline"
NOISE_MULTIPLIER    = 3.0   # 동적 노이즈 임계값 배수 K (min_max, moving_std_min_max 에서 사용)
                             # peak 변화량 > K × resting_std 일 때만 실질적 움직임으로 판정

# ─── 출력 경로 ────────────────────────────────────────────────────────────────
# 정규화 결과: graphs/normalized_{norm_method}_gaussian/...
# 필터 상태:  graphs/filtered_state_gaussian/...
FILTERED_GRAPH_ROOT = Path("/home/taeung/g1_datasets_huggingface/graphs/filtered_state_gaussian")
_GRAPH_BASE = Path("/home/taeung/g1_datasets_huggingface/graphs")

# ─── YAML config 경로 ────────────────────────────────────────────────────────
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

_HAND_CONFIG_FILES = {
    "g1_with_brainco": "hand_brainco.yaml",
    "g1_with_dex3":    "hand_dex3.yaml",
    "g1_with_inspire": "hand_inspire.yaml",
}


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── 관절 그룹 정의 (YAML config 에서 읽기) ──────────────────────────────────
def get_joint_groups(control_type: str, hand_type: str) -> dict[str, list[str]]:
    """
    WBT (7개 그룹): g1_base.yaml + hand_*.yaml
    Upper_body_control (4개 그룹): g1_base.yaml 팔 그룹 + hand_*.yaml
    """
    base_groups = _load_yaml(_CONFIG_DIR / "g1_base.yaml")["joint_groups"]
    hand_groups = _load_yaml(_CONFIG_DIR / _HAND_CONFIG_FILES[hand_type])["joint_groups"]

    if control_type == "WBT":
        return {
            "left_leg":   base_groups["left_leg"],
            "right_leg":  base_groups["right_leg"],
            "waist":      base_groups["waist"],
            "left_arm":   base_groups["left_arm"],
            "right_arm":  base_groups["right_arm"],
            "left_hand":  hand_groups["left_hand"],
            "right_hand": hand_groups["right_hand"],
        }
    else:
        return {
            "left_arm":   base_groups["left_arm"],
            "right_arm":  base_groups["right_arm"],
            "left_hand":  hand_groups["left_hand"],
            "right_hand": hand_groups["right_hand"],
        }


# ─── [Gaussian] 가우시안 스무딩 필터 ─────────────────────────────────────────
def apply_gaussian_filter(
    state_matrix: np.ndarray,
    sigma: float = GAUSSIAN_SIGMA,
) -> np.ndarray:
    """
    가우시안 스무딩 필터를 관절 시계열 전체에 적용한다.
    위상 지연 없음. sigma 클수록 강한 평활화.
    """
    return gaussian_filter1d(state_matrix, sigma=sigma, axis=0)


# ─── 출력 경로 빌더 ───────────────────────────────────────────────────────────
def build_output_dir(
    norm_method: str, control_type: str, hand_type: str, dataset_name: str
) -> Path:
    """정규화 결과 그래프 저장 경로: graphs/normalized_{norm_method}_gaussian/..."""
    return _GRAPH_BASE / f"normalized_{norm_method}_gaussian" / control_type / hand_type / dataset_name


def build_filtered_output_dir(
    control_type: str, hand_type: str, dataset_name: str
) -> Path:
    """필터 적용 후 관절각 그래프 저장 경로: graphs/filtered_state_gaussian/..."""
    return FILTERED_GRAPH_ROOT / control_type / hand_type / dataset_name


# ─── 정규화 결과 시각화 ───────────────────────────────────────────────────────
def plot_normalized_episode(
    episode_df: pd.DataFrame,
    episode_index: int,
    joint_order: list[str],
    joint_groups: dict[str, list[str]],
    output_dir: Path,
    control_type: str,
    hand_type: str,
    norm_fn,
    y_label: str,
    norm_method: str,
    baseline_std: np.ndarray | None,
    std_window: int,
    noise_multiplier: float,
    sigma: float,
) -> None:
    """
    Gaussian → norm_method 정규화 후 시각화한다.
    joint_limit_range 선택 시: Gaussian → 가동 범위(Upper-Lower)로 나누기
    """
    episode_df = episode_df.sort_values("frame_index")
    frame_seconds = episode_df["frame_index"].to_numpy(dtype=float) / FPS

    # ① 가우시안 필터 적용
    state_matrix   = extract_joint_matrix(episode_df, control_type, hand_type, joint_order)
    state_filtered = apply_gaussian_filter(state_matrix, sigma=sigma)

    # ② norm_method 정규화 적용
    # joint_order, hand_type 을 kwargs 로 전달 → joint_limit_range 방법에서 사용
    normalized = norm_fn(
        state_filtered,
        baseline_std=baseline_std,
        std_window=std_window,
        noise_multiplier=noise_multiplier,
        joint_order=joint_order,
        hand_type=hand_type,
    )

    joint_to_idx = {name: i for i, name in enumerate(joint_order)}
    group_items  = list(joint_groups.items())
    n_groups     = len(group_items)

    fig, axes = plt.subplots(
        nrows=n_groups, ncols=1,
        figsize=(16, max(3.0, 2.5 * n_groups)),
        sharex=True,
    )
    if n_groups == 1:
        axes = [axes]

    for ax, (group_name, joints) in zip(axes, group_items):
        for joint_name in joints:
            if joint_name not in joint_to_idx:
                continue
            ax.plot(
                frame_seconds, normalized[:, joint_to_idx[joint_name]],
                label=joint_name, linewidth=0.9,
            )
        if norm_method == "baseline":
            ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.9,
                       alpha=0.7, label="baseline (V=1)")
        ax.axvspan(0, STATIONARY_FRAMES / FPS, color="#e0f0ff", alpha=0.35, label="stop zone")
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(group_name, fontsize=10)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.xaxis.set_major_locator(MultipleLocator(X_TICK_SECS))
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        f"[Gaussian + {norm_method}] Episode {episode_index}"
        f"  |  sigma={sigma}f ({sigma / FPS:.2f}s)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"episode_{episode_index}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [저장-정규화] {out_path}")


# ─── 필터 적용 후 관절각 시각화 ───────────────────────────────────────────────
def plot_filtered_state_episode(
    episode_df: pd.DataFrame,
    episode_index: int,
    joint_order: list[str],
    joint_groups: dict[str, list[str]],
    output_dir: Path,
    control_type: str,
    hand_type: str,
    sigma: float,
) -> None:
    """가우시안 필터 적용 후 관절각 시계열을 그룹별 서브플롯으로 시각화한다."""
    episode_df = episode_df.sort_values("frame_index")
    frame_seconds = episode_df["frame_index"].to_numpy(dtype=float) / FPS

    state_matrix   = extract_joint_matrix(episode_df, control_type, hand_type, joint_order)
    state_filtered = apply_gaussian_filter(state_matrix, sigma=sigma)

    joint_to_idx = {name: i for i, name in enumerate(joint_order)}
    group_items  = list(joint_groups.items())
    n_groups     = len(group_items)

    fig, axes = plt.subplots(
        nrows=n_groups, ncols=1,
        figsize=(16, max(3.0, 2.5 * n_groups)),
        sharex=True,
    )
    if n_groups == 1:
        axes = [axes]

    for ax, (group_name, joints) in zip(axes, group_items):
        for joint_name in joints:
            if joint_name not in joint_to_idx:
                continue
            ax.plot(
                frame_seconds, state_filtered[:, joint_to_idx[joint_name]],
                label=joint_name, linewidth=0.9,
            )
        ax.axvspan(0, STATIONARY_FRAMES / FPS, color="#e0f0ff", alpha=0.35, label="stop zone")
        ax.set_ylabel("state (rad)", fontsize=9)
        ax.set_title(group_name, fontsize=10)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.xaxis.set_major_locator(MultipleLocator(X_TICK_SECS))
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        f"Filtered Joint State (Gaussian) — Episode {episode_index}"
        f"  |  sigma={sigma}f ({sigma / FPS:.2f}s)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"episode_{episode_index}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [저장-필터상태] {out_path}")


# ─── 구간 STD 분석 ────────────────────────────────────────────────────────────
def analyze_interval_std(
    state_filtered: np.ndarray,
    frame_seconds: np.ndarray,
    t1: float,
    t2: float,
    joint_order: list[str],
    joint_groups: dict[str, list[str]],
    episode_index: int,
) -> tuple[np.ndarray, dict[str, list[tuple[str, float]]]]:
    """
    [t1, t2] 구간의 관절별 STD 를 계산하고 터미널에 출력한다.
    그룹별로 STD 상위 2개 관절을 반환한다.

    Returns:
        interval_stds  : (N_joints,) — 구간 내 관절별 STD
        top2_per_group : {group_name: [(joint_name, std_val), ...]}
    """
    mask        = (frame_seconds >= t1) & (frame_seconds <= t2)
    n_frames_in = int(mask.sum())

    if n_frames_in == 0:
        print(f"  [경고] episode {episode_index}: [{t1:.2f}s, {t2:.2f}s] 구간에 데이터가 없습니다.")
        return np.zeros(len(joint_order)), {}

    interval_stds = state_filtered[mask].std(axis=0)  # (N_joints,)
    joint_to_idx  = {name: i for i, name in enumerate(joint_order)}
    top2_per_group: dict[str, list[tuple[str, float]]] = {}

    print()
    print("=" * 72)
    print(
        f"  [구간 STD 분석]  Episode {episode_index}"
        f"  |  구간: [{t1:.2f}s, {t2:.2f}s]  ({n_frames_in} 프레임)"
    )
    print("=" * 72)

    for group_name, joints in joint_groups.items():
        group_stds = [
            (name, float(interval_stds[joint_to_idx[name]]))
            for name in joints if name in joint_to_idx
        ]
        group_stds.sort(key=lambda x: x[1], reverse=True)
        top2_per_group[group_name] = group_stds[:2]

        print(f"\n  [{group_name}]")
        for rank, (name, val) in enumerate(group_stds, start=1):
            badge = "  ★ TOP1" if rank == 1 else ("  ★ TOP2" if rank == 2 else "")
            print(f"    {rank:>2}.  {name:<32}  STD: {val:.8f}{badge}")

    print("=" * 72)
    print()
    return interval_stds, top2_per_group


def plot_finding_std_episode(
    state_filtered: np.ndarray,
    frame_seconds: np.ndarray,
    top2_per_group: dict[str, list[tuple[str, float]]],
    joint_order: list[str],
    joint_groups: dict[str, list[str]],
    t1: float,
    t2: float,
    episode_index: int,
    output_dir: Path,
    filter_desc: str = "",
) -> None:
    """
    [t1, t2] 구간에서 그룹별 STD 상위 2개 관절을 강조한 그래프를 저장한다.

    - 배경 관절 : 연한 회청색 가는 선 (alpha=0.2)
    - TOP1 관절 : 빨간 굵은 실선 + STD 어노테이션
    - TOP2 관절 : 주황 굵은 점선 + STD 어노테이션
    - [t1, t2]  : 노란색 배경 + 경계 수직선
    """
    joint_to_idx = {name: i for i, name in enumerate(joint_order)}
    group_items  = list(joint_groups.items())
    n_groups     = len(group_items)

    fig, axes = plt.subplots(
        nrows=n_groups, ncols=1,
        figsize=(16, max(3.0, 2.5 * n_groups)),
        sharex=True,
    )
    if n_groups == 1:
        axes = [axes]

    HIGHLIGHT = [
        {"lw": 2.2, "ls": "-",  "color": "#e63946", "badge": "TOP1"},
        {"lw": 2.0, "ls": "--", "color": "#f4a261", "badge": "TOP2"},
    ]

    for ax, (group_name, joints) in zip(axes, group_items):
        top2       = top2_per_group.get(group_name, [])
        top2_names = {name for name, _ in top2}

        # 배경 관절 (연하게)
        for joint_name in joints:
            if joint_name not in joint_to_idx:
                continue
            ax.plot(
                frame_seconds, state_filtered[:, joint_to_idx[joint_name]],
                linewidth=0.7, alpha=0.2, color="steelblue",
            )

        # TOP1 / TOP2 강조
        for rank_idx, (joint_name, std_val) in enumerate(top2):
            if joint_name not in joint_to_idx:
                continue
            st = HIGHLIGHT[rank_idx]
            y  = state_filtered[:, joint_to_idx[joint_name]]
            ax.plot(
                frame_seconds, y,
                linewidth=st["lw"], linestyle=st["ls"], color=st["color"], zorder=5,
                label=f"[{st['badge']}] {joint_name}  STD={std_val:.5f}",
            )
            # 어노테이션: 구간 중앙 상단
            t_mid   = (t1 + t2) / 2
            y_range = y.max() - y.min() + 1e-6
            ax.annotate(
                f"{st['badge']}: STD={std_val:.5f}",
                xy=(t_mid, y.max() + 0.04 * y_range),
                fontsize=7, color=st["color"], ha="center", va="bottom",
                fontweight="bold",
            )

        # [t1, t2] 구간 강조
        ax.axvspan(t1, t2, color="#ffb703", alpha=0.15, label=f"[{t1:.2f}s ~ {t2:.2f}s]")
        ax.axvline(t1, color="#ffb703", linewidth=1.3, alpha=0.9)
        ax.axvline(t2, color="#ffb703", linewidth=1.3, alpha=0.9)

        ax.set_ylabel("state (rad)", fontsize=9)
        ax.set_title(f"{group_name}  —  TOP2 highlighted", fontsize=10)
        ax.legend(loc="upper right", fontsize=7, ncol=1)
        ax.xaxis.set_major_locator(MultipleLocator(X_TICK_SECS))
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        f"Finding STD  [{t1:.2f}s ~ {t2:.2f}s]  —  Episode {episode_index}  {filter_desc}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"episode_{episode_index}_finding_std.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [저장-finding_std] {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    class HF(RawTextHelpFormatter):
        def __init__(self, prog: str) -> None:
            super().__init__(prog, max_help_position=38, width=110)

    parser = argparse.ArgumentParser(
        description=(
            "Gaussian 스무딩 전처리 스크립트\n"
            "처리 순서: raw state → Gaussian Filter → 정규화 → 시각화"
        ),
        formatter_class=HF,
    )
    parser.add_argument(
        "--folder-name", default=FOLDER_NAME, choices=ALL_FOLDER_NAMES,
        metavar="FOLDER_NAME",
        help=f"HuggingFace hub 캐시 폴더 이름.\n  default: {FOLDER_NAME}",
    )
    parser.add_argument(
        "--dataset-name", default=DATASET_NAME, metavar="DATASET_NAME",
        help=f"데이터셋 이름 ('datasets--' 접두사 포함).\n  default: {DATASET_NAME}",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=MAX_EPISODES, metavar="N",
        help=f"시각화할 최대 에피소드 수.\n  default: {MAX_EPISODES}",
    )
    parser.add_argument(
        "--std-window", type=int, default=STD_WINDOW, metavar="FRAMES",
        help=(
            f"Moving STD 슬라이딩 윈도우 크기 (프레임).\n"
            f"  baseline / moving_std_min_max 방법에서 사용.\n"
            f"  default: {STD_WINDOW} ({STD_WINDOW / FPS:.2f}s)"
        ),
    )
    parser.add_argument(
        "--sigma", type=float, default=GAUSSIAN_SIGMA, metavar="SIGMA",
        help=(
            f"가우시안 커널 표준편차 (프레임).\n"
            f"  default: {GAUSSIAN_SIGMA} ({GAUSSIAN_SIGMA / FPS:.2f}s at {FPS}fps)"
        ),
    )
    parser.add_argument(
        "--norm-method",
        default=DEFAULT_NORM_METHOD,
        choices=list(NORM_MODULE_MAP.keys()),
        metavar="METHOD",
        help=(
            "정규화 방법.\n"
            "  선택지: " + " | ".join(NORM_MODULE_MAP.keys()) + "\n"
            f"  default: {DEFAULT_NORM_METHOD}"
        ),
    )
    parser.add_argument(
        "--noise-multiplier", type=float, default=NOISE_MULTIPLIER, metavar="K",
        help=(
            "동적 노이즈 임계값 배수 K (min_max, moving_std_min_max 에서 사용).\n"
            "  peak 변화량 > K × resting_std 일 때만 실질적 움직임으로 판정.\n"
            f"  default: {NOISE_MULTIPLIER}  (3-시그마 규칙 기준)"
        ),
    )
    parser.add_argument(
        "--list-supported-datasets", action="store_true",
        help="지원되는 폴더/데이터셋 목록 출력 후 종료.",
    )
    parser.add_argument(
        "--show_1sec_std",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        metavar="BOOL",
        help=(
            "True 일 때 Gaussian 필터 적용 후 데이터에 대해\n"
            "  1초 구간 STD 로그·시각화를 에피소드별로 실행한다.\n"
            "  저장 경로: graphs/1sec_std_gaussian/\n"
            "  default: False"
        ),
    )
    parser.add_argument(
        "--finding_std",
        nargs=2,
        type=float,
        default=None,
        metavar=("T1", "T2"),
        help=(
            "[t1, t2] 구간의 관절별 STD 를 분석하고\n"
            "  그룹별 상위 2개 관절을 강조한 그래프를 저장한다.\n"
            "  저장 경로: graphs/finding_std_gaussian/\n"
            "  비활성(default): None\n"
            "  예: --finding_std 5.0 20.0"
        ),
    )
    return parser.parse_args()


# ─── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    if args.list_supported_datasets:
        print("지원되는 폴더/데이터셋 목록:")
        for folder in ALL_FOLDER_NAMES:
            print(f"- {folder}")
            for ds in FOLDER_TO_DATASETS[folder]:
                print(f"    * {ds}")
        return

    # ── 정규화 모듈 동적 로드 ─────────────────────────────────────────────────
    module_name = NORM_MODULE_MAP[args.norm_method]
    norm_mod    = importlib.import_module(f"data_normalization.{module_name}")
    norm_fn     = norm_mod.normalize
    y_label     = norm_mod.Y_LABEL
    print(f"[정규화] 모듈 로드: data_normalization/{module_name}.py  →  y축: '{y_label}'")

    # ── 데이터셋 메타데이터 결정 ──────────────────────────────────────────────
    validate_dataset_selection(args.folder_name, args.dataset_name)
    control_type = classify_control_type(args.folder_name, args.dataset_name)
    hand_type    = classify_hand_type(args.folder_name, args.dataset_name)
    joint_order  = get_joint_order(control_type, hand_type)
    joint_groups = get_joint_groups(control_type, hand_type)

    print("=" * 60)
    print(f"[설정] 폴더명        : {args.folder_name}")
    print(f"[설정] 데이터셋명    : {args.dataset_name}")
    print(f"[설정] 제어 타입     : {control_type}")
    print(f"[설정] 핸드 타입     : {hand_type}")
    print(f"[설정] 관절 수       : {len(joint_order)}")
    print(f"[설정] 그룹          : {list(joint_groups.keys())}")
    print(f"[설정] 정규화 방법   : {args.norm_method}")
    print(f"[설정] STD 윈도우    : {args.std_window} frames ({args.std_window / FPS:.2f}s)")
    print(f"[설정] 노이즈 배수 K : {args.noise_multiplier}  (min_max / moving_std_min_max 에서 사용)")
    print(f"[설정] 최대 에피소드 : {args.max_episodes}")
    print(f"[Gaussian] sigma     : {args.sigma} frames ({args.sigma / FPS:.2f}s)")
    print("=" * 60)

    # ── [1/5] parquet 파일 검색 ───────────────────────────────────────────────
    print("\n[1/5] parquet 파일 검색 중...")
    snapshot_dir = resolve_snapshot_dir(BASIC_PATH, args.folder_name, args.dataset_name)
    parquet_path = find_first_parquet_file(snapshot_dir)
    print(f"      심볼릭 링크 : {parquet_path}")
    print(f"      실제 파일   : {parquet_path.resolve()}")

    # ── [2/5] DataFrame 로드 ─────────────────────────────────────────────────
    print("\n[2/5] 데이터 로딩 중...")
    df = load_raw_dataframe(parquet_path, control_type)
    print(f"      DataFrame shape  : {df.shape}  (행 {df.shape[0]}, 열 {df.shape[1]})")
    print(f"      열 이름          : {list(df.columns)}")
    ep_list = sorted(df["episode_index"].unique().tolist())
    print(f"      전체 에피소드    : {len(ep_list)}개  {ep_list[:5]}{'...' if len(ep_list) > 5 else ''}")

    # ── [3/5] 통계 계산 (정규화 방법에 따라 다름) ────────────────────────────
    baseline_std = None
    if args.norm_method == "baseline":
        print(f"\n[3/5] 베이스라인 STD 계산 중  (frame_index 0~{STATIONARY_FRAMES - 1})...")
        baseline_std, all_stds = compute_stop_state_median_std(
            df, control_type, hand_type, joint_order
        )
        print(f"      사용된 에피소드 수 : {all_stds.shape[0]}")
        print(f"      관절별 베이스라인 STD (중앙값):")
        for i, (name, val) in enumerate(zip(joint_order, baseline_std)):
            print(f"        {i:>3}  {name:<32}  {val:.8f}")
    else:
        print(f"\n[3/5] 정규화 방법 '{args.norm_method}' — 베이스라인 STD 계산 불필요, 건너뜀")

    # ── [4/5] 정규화 결과 시각화 ─────────────────────────────────────────────
    output_dir = build_output_dir(args.norm_method, control_type, hand_type, args.dataset_name)
    print(f"\n[4/5] 정규화 결과 시각화 중  →  {output_dir}")
    print(f"      (대상: episode 0 ~ {args.max_episodes - 1})\n")

    for ep_idx in range(args.max_episodes):
        ep_df = df[df["episode_index"] == ep_idx]
        if ep_df.empty:
            print(f"  [SKIP] episode {ep_idx}: 데이터 없음")
            continue
        print(f"  episode {ep_idx:>3} : {len(ep_df)} frames  ({ep_df['frame_index'].max() / FPS:.1f}s)")
        plot_normalized_episode(
            ep_df,
            episode_index=ep_idx,
            joint_order=joint_order,
            joint_groups=joint_groups,
            output_dir=output_dir,
            control_type=control_type,
            hand_type=hand_type,
            norm_fn=norm_fn,
            y_label=y_label,
            norm_method=args.norm_method,
            baseline_std=baseline_std,
            std_window=args.std_window,
            noise_multiplier=args.noise_multiplier,
            sigma=args.sigma,
        )

    # ── [5/5] 필터 적용 후 관절각 시각화 ─────────────────────────────────────
    filtered_output_dir = build_filtered_output_dir(control_type, hand_type, args.dataset_name)
    print(f"\n[5/5] 필터 적용 후 관절각 시각화 중  →  {filtered_output_dir}")
    print(f"      (대상: episode 0 ~ {args.max_episodes - 1})\n")

    for ep_idx in range(args.max_episodes):
        ep_df = df[df["episode_index"] == ep_idx]
        if ep_df.empty:
            continue
        print(f"  episode {ep_idx:>3} : {len(ep_df)} frames")
        plot_filtered_state_episode(
            ep_df,
            episode_index=ep_idx,
            joint_order=joint_order,
            joint_groups=joint_groups,
            output_dir=filtered_output_dir,
            control_type=control_type,
            hand_type=hand_type,
            sigma=args.sigma,
        )

    # ── [보조] Gaussian 적용 후 1초 구간 STD 시각화 (--show_1sec_std True 일 때만) ─
    if args.show_1sec_std:
        std1sec_output_dir = (
            _GRAPH_BASE / "1sec_std_gaussian" / control_type / hand_type / args.dataset_name
        )
        print(f"\n[보조] 1초 구간 STD 시각화 중  →  {std1sec_output_dir}")
        print(f"       (Gaussian sigma={args.sigma}f 적용 후 데이터 기준)\n")

        for ep_idx in range(args.max_episodes):
            ep_df = df[df["episode_index"] == ep_idx]
            if ep_df.empty:
                continue
            state_matrix   = extract_joint_matrix(ep_df.sort_values("frame_index"),
                                                   control_type, hand_type, joint_order)
            state_filtered = apply_gaussian_filter(state_matrix, sigma=args.sigma)
            print(f"  episode {ep_idx:>3} : {len(ep_df)} frames")
            log_and_plot_1sec_std(
                state_matrix=state_filtered,
                joint_names=joint_order,
                episode_index=ep_idx,
                fps=FPS,
                output_dir=std1sec_output_dir,
                title_prefix=f"[Gaussian sigma={args.sigma}f] ",
            )

    # ── [보조-finding_std] 구간 STD 분석 (--finding_std t1 t2 일 때만) ─────────
    if args.finding_std is not None:
        t1, t2 = args.finding_std
        finding_output_dir = (
            _GRAPH_BASE / "finding_std_gaussian" / control_type / hand_type / args.dataset_name
        )
        print(f"\n[보조-finding_std] 구간 [{t1:.2f}s, {t2:.2f}s] STD 분석 중"
              f"  →  {finding_output_dir}\n")

        for ep_idx in range(args.max_episodes):
            ep_df = df[df["episode_index"] == ep_idx]
            if ep_df.empty:
                continue
            ep_sorted      = ep_df.sort_values("frame_index")
            frame_seconds  = ep_sorted["frame_index"].to_numpy(dtype=float) / FPS
            state_matrix   = extract_joint_matrix(ep_sorted, control_type, hand_type, joint_order)
            state_filtered = apply_gaussian_filter(state_matrix, sigma=args.sigma)
            print(f"  episode {ep_idx:>3} : {len(ep_df)} frames  "
                  f"({ep_sorted['frame_index'].max() / FPS:.1f}s)")

            _, top2_per_group = analyze_interval_std(
                state_filtered, frame_seconds,
                t1, t2,
                joint_order, joint_groups, ep_idx,
            )
            plot_finding_std_episode(
                state_filtered, frame_seconds,
                top2_per_group,
                joint_order, joint_groups,
                t1, t2,
                episode_index=ep_idx,
                output_dir=finding_output_dir,
                filter_desc=f"[Gaussian sigma={args.sigma}f]",
            )

    print(f"\n완료.")
    print(f"  정규화 결과  : {output_dir}")
    print(f"  필터 적용 상태: {filtered_output_dir}")
    if args.show_1sec_std:
        print(f"  1초 구간 STD  : {_GRAPH_BASE / '1sec_std_gaussian' / control_type / hand_type / args.dataset_name}")
    if args.finding_std is not None:
        print(f"  finding_std   : {_GRAPH_BASE / 'finding_std_gaussian' / control_type / hand_type / args.dataset_name}")


if __name__ == "__main__":
    main()
