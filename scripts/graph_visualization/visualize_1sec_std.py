"""
visualize_1sec_std.py
────────────────────────────────────────────────────────────────────
목적 : 에피소드 시계열 데이터를 1초 구간으로 분할하여
       구간별 표준편차(STD)를 계산하고 터미널에 로그 출력 및
       matplotlib 으로 시각화한다.

설계 원칙 :
    핵심 함수 log_and_plot_1sec_std() 는 독립 재사용 가능하도록 설계되어
    data_preprocessing_baseline_gaussian_filter.py 등 전처리 파이프라인에
    그대로 통합할 수 있다.

단독 실행 시 :
    모의(mock) 에피소드 데이터를 자동 생성하여 동작을 검증한다.
    --show_1sec_std True 인자를 전달해야 로그·시각화가 활성화된다.
────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

matplotlib.use("Agg")

# ─── 하이퍼파라미터 ──────────────────────────────────────────────────────────
FPS            = 30.0   # 샘플링 주파수 (Hz). compute_1sec_std_windows() 기본값
WINDOW_SEC     = 1.0    # 구간 길이 (초)
X_TICK_SECS    = 2.0    # X축 눈금 간격 (초)
GRAPH_ROOT     = Path("/home/taeung/g1_datasets_huggingface/graphs/1sec_std_analysis")


# ════════════════════════════════════════════════════════════════════
# ① 1초 구간 STD 계산
# ════════════════════════════════════════════════════════════════════
def compute_1sec_std_windows(
    state_matrix: np.ndarray,
    fps: float = FPS,
    window_sec: float = WINDOW_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    """
    시계열 데이터를 window_sec 단위 구간으로 분할하고
    각 구간의 관절별 STD 를 계산한다.

    Args:
        state_matrix : (N_frames, N_joints) — 관절각 시계열 행렬
        fps          : 샘플링 주파수 (Hz)
        window_sec   : 구간 길이 (초)

    Returns:
        window_stds    : (N_windows, N_joints) — 구간별 관절 STD
        window_centers : (N_windows,)          — 각 구간 중앙 시각 (초)
    """
    frames_per_window = max(1, int(round(fps * window_sec)))
    n_frames, n_joints = state_matrix.shape
    n_windows = n_frames // frames_per_window

    window_stds    = np.zeros((n_windows, n_joints))
    window_centers = np.zeros(n_windows)

    for w in range(n_windows):
        start = w * frames_per_window
        end   = start + frames_per_window
        window_stds[w]    = state_matrix[start:end].std(axis=0)
        window_centers[w] = (start + end) / 2.0 / fps

    return window_stds, window_centers


# ════════════════════════════════════════════════════════════════════
# ② 터미널 로그 출력
# ════════════════════════════════════════════════════════════════════
def log_1sec_std(
    window_stds: np.ndarray,
    window_centers: np.ndarray,
    joint_names: list[str],
    episode_index: int,
    fps: float = FPS,
    window_sec: float = WINDOW_SEC,
) -> None:
    """
    1초 구간별 STD 값을 터미널에 표 형식으로 출력한다.

    Args:
        window_stds    : (N_windows, N_joints)
        window_centers : (N_windows,)
        joint_names    : 관절 이름 리스트
        episode_index  : 에피소드 번호 (로그 헤더용)
        fps            : 샘플링 주파수
        window_sec     : 구간 길이 (초)
    """
    frames_per_window = int(round(fps * window_sec))
    n_windows, n_joints = window_stds.shape
    col_w = 10  # 각 관절 값 칸 너비

    print()
    print("=" * 70)
    print(
        f"  [1초 구간 STD 분석]  Episode {episode_index}"
        f"  |  총 {n_windows}구간"
        f"  ({fps:.0f}fps × {window_sec:.1f}초 = {frames_per_window}프레임/구간)"
    )
    print("=" * 70)

    # 헤더: 관절 이름 (길면 잘라서 표시)
    header_names = [name[-10:] if len(name) > 10 else name for name in joint_names]
    header = "  구간    시간범위     | " + "  ".join(f"{n:>{col_w}}" for n in header_names)
    print(header)
    print("-" * max(70, len(header)))

    for w in range(n_windows):
        t_start = window_centers[w] - window_sec / 2
        t_end   = window_centers[w] + window_sec / 2
        vals    = "  ".join(f"{v:>{col_w}.5f}" for v in window_stds[w])
        print(f"  {w+1:>3}   {t_start:>5.1f}s~{t_end:<5.1f}s  | {vals}")

    # 요약: 전 구간 평균 STD
    mean_std = window_stds.mean(axis=0)
    print("-" * max(70, len(header)))
    print("  [평균]               | " + "  ".join(f"{v:>{col_w}.5f}" for v in mean_std))
    print("=" * 70)
    print()


# ════════════════════════════════════════════════════════════════════
# ③ 시각화
# ════════════════════════════════════════════════════════════════════
def plot_1sec_std(
    state_matrix: np.ndarray,
    window_stds: np.ndarray,
    window_centers: np.ndarray,
    joint_names: list[str],
    episode_index: int,
    fps: float = FPS,
    window_sec: float = WINDOW_SEC,
    output_dir: Path | None = None,
    title_prefix: str = "",
) -> None:
    """
    원본(또는 필터링된) 시계열과 1초 구간 STD 를 2-패널 그래프로 시각화한다.

    상단 패널 : 관절각 원본 시계열 (시간축)
    하단 패널 : 1초 구간별 STD (step 플롯 + 마커)
    세로 점선 : 1초 경계선

    Args:
        state_matrix   : (N_frames, N_joints)
        window_stds    : (N_windows, N_joints)
        window_centers : (N_windows,)
        joint_names    : 관절 이름 리스트
        episode_index  : 에피소드 번호
        fps            : 샘플링 주파수
        window_sec     : 구간 길이 (초)
        output_dir     : 저장 디렉토리 (None 이면 저장 안 함)
        title_prefix   : 그래프 제목 앞에 붙을 문자열
    """
    n_frames, n_joints = state_matrix.shape
    frame_seconds = np.arange(n_frames) / fps

    # 1초 경계 시각 목록
    n_windows = len(window_centers)
    boundaries = [window_centers[w] - window_sec / 2 for w in range(n_windows)]
    boundaries.append(window_centers[-1] + window_sec / 2)

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1,
        figsize=(16, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # ── 상단: 관절각 원본 시계열 ──────────────────────────────────────────────
    for j, name in enumerate(joint_names):
        ax_top.plot(frame_seconds, state_matrix[:, j], linewidth=0.8, label=name, alpha=0.85)

    # 1초 경계 세로 점선
    for b in boundaries[1:-1]:
        ax_top.axvline(b, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)

    ax_top.set_ylabel("state (rad)", fontsize=9)
    ax_top.set_title(
        f"{title_prefix}Joint State — Episode {episode_index}", fontsize=10
    )
    ax_top.legend(loc="upper right", fontsize=7, ncol=3)
    ax_top.xaxis.set_major_locator(MultipleLocator(X_TICK_SECS))
    ax_top.grid(True, linestyle=":", alpha=0.3)

    # ── 하단: 1초 구간 STD ───────────────────────────────────────────────────
    # step 플롯: 구간 시작~끝을 가로로 잇고 마커로 중앙을 표시
    for j, name in enumerate(joint_names):
        color = ax_top.lines[j].get_color()
        # step 그래프 (구간 내 일정값 표현)
        step_x = []
        step_y = []
        for w in range(n_windows):
            t_start = window_centers[w] - window_sec / 2
            t_end   = window_centers[w] + window_sec / 2
            step_x += [t_start, t_end]
            step_y += [window_stds[w, j], window_stds[w, j]]
        ax_bot.plot(step_x, step_y, linewidth=1.2, color=color, alpha=0.7)
        # 구간 중앙 마커
        ax_bot.plot(
            window_centers, window_stds[:, j],
            "o", markersize=4, color=color, alpha=0.9,
        )

    # 1초 경계 세로 점선
    for b in boundaries[1:-1]:
        ax_bot.axvline(b, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)

    ax_bot.set_ylabel("1-sec window STD", fontsize=9)
    ax_bot.set_xlabel("time (s)", fontsize=9)
    ax_bot.set_title("1-Second Window STD", fontsize=10)
    ax_bot.xaxis.set_major_locator(MultipleLocator(X_TICK_SECS))
    ax_bot.grid(True, linestyle=":", alpha=0.3)

    fig.suptitle(
        f"1-Second Interval STD Analysis  |  Episode {episode_index}"
        f"  ({fps:.0f}fps, window={window_sec:.1f}s)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"episode_{episode_index}_1sec_std.png"
        fig.savefig(out_path, dpi=150)
        print(f"  [저장] {out_path}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# ④ 통합 진입점 (재사용 가능 함수)
# ════════════════════════════════════════════════════════════════════
def log_and_plot_1sec_std(
    state_matrix: np.ndarray,
    joint_names: list[str],
    episode_index: int,
    fps: float = FPS,
    window_sec: float = WINDOW_SEC,
    output_dir: Path | None = None,
    title_prefix: str = "",
) -> None:
    """
    1초 구간 STD 계산 → 터미널 로그 → 시각화를 한 번에 수행하는 통합 함수.

    전처리 파이프라인(data_preprocessing_baseline_gaussian_filter.py 등)에서
    --show_1sec_std True 일 때 이 함수를 호출하면 된다.

    Args:
        state_matrix  : (N_frames, N_joints) — 필터링 후 관절각 행렬
        joint_names   : 관절 이름 리스트 (길이 == N_joints)
        episode_index : 에피소드 번호
        fps           : 샘플링 주파수 (Hz)
        window_sec    : 구간 길이 (초)
        output_dir    : 그래프 저장 경로 (None 이면 저장 안 함)
        title_prefix  : 그래프 제목 접두사 (필터 종류 등 표시용)
    """
    window_stds, window_centers = compute_1sec_std_windows(
        state_matrix, fps=fps, window_sec=window_sec
    )
    log_1sec_std(window_stds, window_centers, joint_names, episode_index, fps, window_sec)
    plot_1sec_std(
        state_matrix, window_stds, window_centers,
        joint_names, episode_index,
        fps=fps, window_sec=window_sec,
        output_dir=output_dir, title_prefix=title_prefix,
    )


# ════════════════════════════════════════════════════════════════════
# ⑤ 모의 데이터 생성 (단독 실행 테스트용)
# ════════════════════════════════════════════════════════════════════
def _generate_mock_episode(
    fps: float = FPS,
    duration_sec: float = 12.0,
    n_joints: int = 6,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """
    단독 테스트용 모의 에피소드 데이터를 생성한다.

    관절별 특성:
        joint0 : 정지 상태 (미세 노이즈만)
        joint1 : 느린 사인파 (저주파 움직임)
        joint2 : 빠른 사인파 (고주파 움직임)
        joint3 : 스텝 함수 (갑작스러운 자세 변환)
        joint4 : 점진적 드리프트 + 노이즈
        joint5 : 처음 3초 정지 후 활발한 움직임
    """
    rng    = np.random.default_rng(seed)
    n_frames = int(fps * duration_sec)
    t      = np.arange(n_frames) / fps
    matrix = np.zeros((n_frames, n_joints))

    matrix[:, 0] = rng.normal(0, 0.003, n_frames)                        # 정지
    matrix[:, 1] = 0.3 * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.005, n_frames)
    matrix[:, 2] = 0.15 * np.sin(2 * np.pi * 1.5 * t) + rng.normal(0, 0.005, n_frames)
    matrix[:, 3] = 0.4 * (t > 4).astype(float) - 0.2 * (t > 8).astype(float) \
                   + rng.normal(0, 0.004, n_frames)
    matrix[:, 4] = 0.02 * t + 0.1 * np.sin(2 * np.pi * 0.7 * t) + rng.normal(0, 0.006, n_frames)
    active        = (t >= 3.0).astype(float)
    matrix[:, 5] = active * 0.25 * np.sin(2 * np.pi * 0.5 * t) + rng.normal(0, 0.004, n_frames)

    joint_names = [
        "kJoint_still",
        "kJoint_slow",
        "kJoint_fast",
        "kJoint_step",
        "kJoint_drift",
        "kJoint_delayed",
    ]
    return matrix, joint_names


# ════════════════════════════════════════════════════════════════════
# ⑥ CLI 진입점
# ════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "에피소드 시계열 데이터의 1초 구간 STD 를 계산·로그·시각화한다.\n"
            "단독 실행 시 모의(mock) 데이터로 동작을 검증한다."
        )
    )
    parser.add_argument(
        "--show_1sec_std",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        metavar="BOOL",
        help=(
            "True 일 때만 1초 구간 STD 로그·시각화를 실행한다.\n"
            "  예: --show_1sec_std True\n"
            "  default: False"
        ),
    )
    parser.add_argument(
        "--fps", type=float, default=FPS, metavar="HZ",
        help=f"샘플링 주파수 (Hz).  default: {FPS}",
    )
    parser.add_argument(
        "--duration", type=float, default=12.0, metavar="SEC",
        help="모의 데이터 길이 (초).  default: 12.0",
    )
    parser.add_argument(
        "--episode", type=int, default=0, metavar="N",
        help="에피소드 번호 (로그·파일명용).  default: 0",
    )
    parser.add_argument(
        "--save", action="store_true",
        help=f"그래프를 {GRAPH_ROOT} 에 저장한다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[설정] fps={args.fps}  duration={args.duration}s  episode={args.episode}")
    print(f"[설정] --show_1sec_std = {args.show_1sec_std}")

    # 모의 에피소드 데이터 생성
    state_matrix, joint_names = _generate_mock_episode(
        fps=args.fps, duration_sec=args.duration
    )
    print(f"[모의 데이터] shape={state_matrix.shape}  joints={joint_names}")

    if not args.show_1sec_std:
        print("\n--show_1sec_std False → 1초 구간 STD 분석을 건너뜁니다.")
        print("활성화하려면: --show_1sec_std True")
        return

    output_dir = GRAPH_ROOT if args.save else None

    log_and_plot_1sec_std(
        state_matrix=state_matrix,
        joint_names=joint_names,
        episode_index=args.episode,
        fps=args.fps,
        output_dir=output_dir,
        title_prefix="[Mock] ",
    )

    if output_dir:
        print(f"\n그래프 저장 완료: {output_dir}")
    else:
        print("\n(--save 없이 실행 → 그래프 파일 저장 안 함)")


if __name__ == "__main__":
    main()
