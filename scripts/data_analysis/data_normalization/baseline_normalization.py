"""
baseline_normalization.py
─────────────────────────────────────────────────────────────────
정규화 방법 : Baseline Normalization
수식        : V_i(t) = Moving_STD_i(t) / (Baseline_STD_i + ε)

- Moving_STD  : 슬라이딩 윈도우로 계산한 이동 표준편차
- Baseline_STD: 에피소드 시작 1초(정지 구간) 관절별 STD 중앙값
- V_i(t) > 1 이면 jitter 이상의 실질적 움직임으로 판단
-────────────────────────────────────────────────────────────────
"""
import numpy as np
import pandas as pd

# ─── 하이퍼파라미터 ──────────────────────────────────────────────────────────
STD_WINDOW = 21    # Moving STD 슬라이딩 윈도우 크기 (프레임). 30fps 기준 ≈ 0.7초
EPSILON    = 1e-6  # 0 나눗셈 방지용 epsilon

# 그래프 y축 레이블
Y_LABEL = "V (baseline-normalized)"


def _compute_moving_std(state_matrix: np.ndarray, window: int) -> np.ndarray:
    """관절별 이동 표준편차 계산. pandas rolling() 벡터화 처리."""
    df = pd.DataFrame(state_matrix)
    rolling_std = (
        df.rolling(window=window, center=True, min_periods=max(3, window // 3))
        .std()
        .bfill()
        .ffill()
        .fillna(0.0)
    )
    return rolling_std.to_numpy(dtype=float)


def normalize(
    state_matrix: np.ndarray,
    baseline_std: np.ndarray | None = None,
    std_window: int = STD_WINDOW,
    eps: float = EPSILON,
    **kwargs,
) -> np.ndarray:
    """
    Baseline 정규화 적용.

    Args:
        state_matrix : (N_frames, N_joints) — 필터링된 관절각 행렬
        baseline_std : (N_joints,) — 정지 구간 중앙값 STD (필수)
        std_window   : Moving STD 윈도우 크기
        eps          : 0 나눗셈 방지 epsilon
    Returns:
        (N_frames, N_joints) — 정규화 변동성 V_i(t)
    """
    if baseline_std is None:
        raise ValueError(
            "baseline 정규화에는 baseline_std 가 필요합니다. "
            "--norm-method baseline 선택 시 자동 계산됩니다."
        )
    moving_std = _compute_moving_std(state_matrix, window=std_window)
    return moving_std / (baseline_std[np.newaxis, :] + eps)
