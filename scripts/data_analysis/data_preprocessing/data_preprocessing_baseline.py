"""
data_preprocessing_baseline.py
────────────────────────────────────────────────────────────────────
목적 : 로봇 텔레오퍼레이션 시계열 데이터에 베이스라인 정규화를 적용하여
       관절별 정규화 변동성 V_i(t) 를 계산하고 시각화한다.

핵심 수식 :
    V_i(t) = Moving_STD_i(t) / (Baseline_STD_i + 1e-6)

    - Baseline_STD_i : 에피소드 시작 후 1초(정지 구간, frame_index 0~29)의
                       관절별 표준편차 중앙값 (jitter 기준)
    - Moving_STD_i(t): 전체 시계열에 걸친 슬라이딩 윈도우 표준편차
────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")

# ─── stop_state_std_per_each_joint 모듈을 부모 디렉토리에서 import ───────────
# 이 파일은 scripts/data_analysis/data_preprocessing/ 에 위치하므로
# parents[1] = scripts/data_analysis/ 에 있는 분석 모듈을 참조한다.
_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from stop_state_std_per_each_joint import (  # noqa: E402
    BASIC_PATH,
    FPS,
    STATIONARY_FRAMES,
    FOLDER_TO_DATASETS,
    ALL_FOLDER_NAMES,
    ALL_DATASET_NAMES,
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
STD_WINDOW   = 21     # 슬라이딩 윈도우 크기 (프레임). 30fps 기준 약 0.7초
EPSILON      = 1e-6   # 0 나눗셈 방지용
X_TICK_SECS  = 3.0    # X축 눈금 간격 (초)

GRAPH_ROOT = Path("/home/taeung/g1_datasets_huggingface/graphs/normalized_volatility")

# ─── YAML config 경로 ────────────────────────────────────────────────────────
# 이 파일 위치: scripts/data_analysis/data_preprocessing/
# config 위치:  scripts/config/  → parents[2] / "config"
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
    scripts/config/ 의 YAML 파일에서 관절 그룹을 읽어 반환한다.

    WBT (7개 그룹):
        g1_base.yaml   → left_leg, right_leg, waist, left_arm, right_arm
        hand_*.yaml    → left_hand, right_hand

    Upper_body_control (4개 그룹):
        g1_base.yaml   → left_arm, right_arm  (팔 관절 이름은 WBT 와 동일)
        hand_*.yaml    → left_hand, right_hand
    """
    base_groups = _load_yaml(_CONFIG_DIR / "g1_base.yaml")["joint_groups"]
    hand_groups = _load_yaml(_CONFIG_DIR / _HAND_CONFIG_FILES[hand_type])["joint_groups"]

    if control_type == "WBT":
        # WBT: 다리·허리·팔·손 전체 7개 그룹
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
        # Upper_body_control: 팔·손 4개 그룹
        return {
            "left_arm":   base_groups["left_arm"],
            "right_arm":  base_groups["right_arm"],
            "left_hand":  hand_groups["left_hand"],
            "right_hand": hand_groups["right_hand"],
        }


# ─── 이동 표준편차 계산 ───────────────────────────────────────────────────────
def compute_moving_std_matrix(state_matrix: np.ndarray, window: int) -> np.ndarray:
    """
    전체 시계열에 대해 관절별 이동 표준편차(Moving STD)를 계산한다.

    pandas rolling() 을 사용하여 모든 관절을 한 번에 벡터화 처리.
    center=True 이므로 윈도우가 현재 시점을 중앙에 두고 계산된다.

    Args:
        state_matrix : (N_frames, N_joints) ndarray
        window       : 슬라이딩 윈도우 크기 (프레임 수)
    Returns:
        (N_frames, N_joints) ndarray — 각 시점의 이동 표준편차
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


# ─── 정규화 변동성 계산 ───────────────────────────────────────────────────────
def compute_normalized_volatility(
    moving_std_matrix: np.ndarray,
    baseline_std: np.ndarray,
    eps: float = EPSILON,
) -> np.ndarray:
    """
    V_i(t) = Moving_STD_i(t) / (Baseline_STD_i + eps)

    V_i(t) == 1.0 : 이동 STD 가 베이스라인(정지 jitter)과 동일한 수준
    V_i(t)  > 1.0 : 베이스라인 jitter 를 초과하는 실제 관절 움직임 존재

    Args:
        moving_std_matrix : (N_frames, N_joints)
        baseline_std      : (N_joints,) — 정지 구간 중앙값 STD
        eps               : 0 나눗셈 방지 epsilon
    Returns:
        (N_frames, N_joints) — 정규화 변동성
    """
    return moving_std_matrix / (baseline_std[np.newaxis, :] + eps)


# ─── 에피소드 시각화 ──────────────────────────────────────────────────────────
def plot_volatility_episode(
    episode_df: pd.DataFrame,
    episode_index: int,
    baseline_std: np.ndarray,
    joint_order: list[str],
    joint_groups: dict[str, list[str]],
    output_dir: Path,
    control_type: str,
    hand_type: str,
    std_window: int,
) -> None:
    """
    한 에피소드의 정규화 변동성 V_i(t) 를 그룹별 서브플롯으로 시각화한다.
    unannotated_graph_visualization.py 와 동일한 레이아웃/스타일을 따른다.
    """
    episode_df = episode_df.sort_values("frame_index")
    frame_seconds = episode_df["frame_index"].to_numpy(dtype=float) / FPS

    # 관절 행렬 추출 후 이동 STD → 정규화 변동성 계산
    state_matrix = extract_joint_matrix(episode_df, control_type, hand_type, joint_order)
    moving_std   = compute_moving_std_matrix(state_matrix, window=std_window)
    volatility   = compute_normalized_volatility(moving_std, baseline_std)

    joint_to_idx = {name: i for i, name in enumerate(joint_order)}
    group_items  = list(joint_groups.items())
    n_groups     = len(group_items)

    fig, axes = plt.subplots(
        nrows=n_groups,
        ncols=1,
        figsize=(16, max(3.0, 2.5 * n_groups)),
        sharex=True,
    )
    if n_groups == 1:
        axes = [axes]

    for ax, (group_name, joints) in zip(axes, group_items):
        for joint_name in joints:
            if joint_name not in joint_to_idx:
                continue
            j_idx = joint_to_idx[joint_name]
            ax.plot(frame_seconds, volatility[:, j_idx], label=joint_name, linewidth=0.9)

        # V=1 기준선: 이 선을 초과하면 베이스라인 jitter 이상의 움직임
        ax.axhline(
            y=1.0, color="gray", linestyle="--", linewidth=0.9,
            alpha=0.7, label="baseline (V=1)",
        )
        # 정지 구간(에피소드 시작 1초) 배경 표시
        ax.axvspan(
            0, STATIONARY_FRAMES / FPS,
            color="#e0f0ff", alpha=0.35, label="stop zone",
        )

        ax.set_ylabel("V (normalized)", fontsize=9)
        ax.set_title(group_name, fontsize=10)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.xaxis.set_major_locator(MultipleLocator(X_TICK_SECS))
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        f"Normalized Volatility — Episode {episode_index}"
        f"  |  window={std_window} frames ({std_window / FPS:.2f}s)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"episode_{episode_index}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [저장] {out_path}")


def build_output_dir(control_type: str, hand_type: str, dataset_name: str) -> Path:
    return GRAPH_ROOT / control_type / hand_type / dataset_name


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    class HF(RawTextHelpFormatter):
        def __init__(self, prog: str) -> None:
            super().__init__(prog, max_help_position=36, width=110)

    parser = argparse.ArgumentParser(
        description=(
            "베이스라인 정규화 전처리 스크립트\n"
            "정지 구간 STD 를 기준으로 각 관절의 정규화 변동성 V_i(t) 를 계산·시각화한다."
        ),
        formatter_class=HF,
    )
    parser.add_argument(
        "--folder-name",
        default=FOLDER_NAME,
        choices=ALL_FOLDER_NAMES,
        metavar="FOLDER_NAME",
        help=f"HuggingFace hub 캐시 폴더 이름.\n  default: {FOLDER_NAME}",
    )
    parser.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        metavar="DATASET_NAME",
        help=f"데이터셋 이름 ('datasets--' 접두사 포함).\n  default: {DATASET_NAME}",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=MAX_EPISODES,
        metavar="N",
        help=f"시각화할 최대 에피소드 수.\n  default: {MAX_EPISODES}",
    )
    parser.add_argument(
        "--std-window",
        type=int,
        default=STD_WINDOW,
        metavar="FRAMES",
        help=(
            f"슬라이딩 윈도우 크기 (프레임).\n"
            f"  default: {STD_WINDOW}  ({STD_WINDOW / FPS:.2f}s at {FPS}fps)"
        ),
    )
    parser.add_argument(
        "--list-supported-datasets",
        action="store_true",
        help="지원되는 폴더/데이터셋 목록 출력 후 종료.",
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

    # ── 데이터셋 메타데이터 결정 ──────────────────────────────────────────────
    validate_dataset_selection(args.folder_name, args.dataset_name)
    control_type = classify_control_type(args.folder_name, args.dataset_name)
    hand_type    = classify_hand_type(args.folder_name, args.dataset_name)
    joint_order  = get_joint_order(control_type, hand_type)
    joint_groups = get_joint_groups(control_type, hand_type)

    print("=" * 60)
    print(f"[설정] 폴더명       : {args.folder_name}")
    print(f"[설정] 데이터셋명   : {args.dataset_name}")
    print(f"[설정] 제어 타입    : {control_type}")
    print(f"[설정] 핸드 타입    : {hand_type}")
    print(f"[설정] 관절 수      : {len(joint_order)}")
    print(f"[설정] 그룹         : {list(joint_groups.keys())}")
    print(f"[설정] 윈도우 크기  : {args.std_window} frames ({args.std_window / FPS:.2f}s)")
    print(f"[설정] 최대 에피소드: {args.max_episodes}")
    print("=" * 60)

    # ── [1/4] parquet 파일 로드 ───────────────────────────────────────────────
    print("\n[1/4] parquet 파일 검색 중...")
    snapshot_dir = resolve_snapshot_dir(BASIC_PATH, args.folder_name, args.dataset_name)
    parquet_path = find_first_parquet_file(snapshot_dir)
    print(f"      심볼릭 링크 : {parquet_path}")
    print(f"      실제 파일   : {parquet_path.resolve()}")

    # ── [2/4] DataFrame 로드 및 검사 ─────────────────────────────────────────
    print("\n[2/4] 데이터 로딩 중...")
    df = load_raw_dataframe(parquet_path, control_type)
    print(f"      DataFrame shape  : {df.shape}  (행 {df.shape[0]}, 열 {df.shape[1]})")
    print(f"      열 이름          : {list(df.columns)}")
    ep_list = sorted(df["episode_index"].unique().tolist())
    print(f"      전체 에피소드    : {len(ep_list)}개  {ep_list[:5]}{'...' if len(ep_list) > 5 else ''}")

    # ── [3/4] 베이스라인 STD 계산 (정지 구간 frame_index 0~29) ───────────────
    print(f"\n[3/4] 베이스라인 STD 계산 중  (frame_index 0~{STATIONARY_FRAMES - 1})...")
    baseline_std, all_stds = compute_stop_state_median_std(
        df, control_type, hand_type, joint_order
    )
    print(f"      사용된 에피소드 수 : {all_stds.shape[0]}")
    print(f"      관절별 베이스라인 STD (중앙값):")
    for i, (name, val) in enumerate(zip(joint_order, baseline_std)):
        print(f"        {i:>3}  {name:<32}  {val:.8f}")

    # ── [4/4] 에피소드별 정규화 변동성 시각화 ────────────────────────────────
    output_dir = build_output_dir(control_type, hand_type, args.dataset_name)
    print(f"\n[4/4] V_i(t) 시각화 중  →  {output_dir}")
    print(f"      (대상: episode 0 ~ {args.max_episodes - 1})\n")

    for ep_idx in range(args.max_episodes):
        ep_df = df[df["episode_index"] == ep_idx]
        if ep_df.empty:
            print(f"  [SKIP] episode {ep_idx}: 데이터 없음")
            continue
        n_frames = len(ep_df)
        duration = ep_df["frame_index"].max() / FPS
        print(f"  episode {ep_idx:>3} : {n_frames} frames  ({duration:.1f}s)")
        plot_volatility_episode(
            ep_df,
            episode_index=ep_idx,
            baseline_std=baseline_std,
            joint_order=joint_order,
            joint_groups=joint_groups,
            output_dir=output_dir,
            control_type=control_type,
            hand_type=hand_type,
            std_window=args.std_window,
        )

    print(f"\n완료. 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()
