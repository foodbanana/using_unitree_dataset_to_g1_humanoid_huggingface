"""
z_score_normalization.py
─────────────────────────────────────────────────────────────────
정규화 방법 : Z-Score Normalization (표준화)
수식        : z_i(t) = (x_i(t) - μ_i) / (σ_i + ε)

- μ_i, σ_i : 해당 에피소드 내 관절 i 의 평균·표준편차
- 관절별로 독립 계산 → 관절 간 활성화 수준 공정 비교 가능
- 결과: 평균=0, 표준편차≈1 의 분포
─────────────────────────────────────────────────────────────────
"""
import numpy as np

# ─── 하이퍼파라미터 ──────────────────────────────────────────────────────────
EPSILON = 1e-6  # σ=0 인 관절(완전 정지)에서의 0 나눗셈 방지

# 그래프 y축 레이블
Y_LABEL = "z-score"


def normalize(
    state_matrix: np.ndarray,
    eps: float = EPSILON,
    **kwargs,
) -> np.ndarray:
    """
    관절별 Z-Score 정규화.

    각 관절의 평균·표준편차는 해당 에피소드 내에서 독립적으로 계산한다.

    Args:
        state_matrix : (N_frames, N_joints) — 필터링된 관절각 행렬
        eps          : 0 나눗셈 방지 epsilon
    Returns:
        (N_frames, N_joints) — z-score 정규화 결과
    """
    mean = state_matrix.mean(axis=0)  # (N_joints,)
    std  = state_matrix.std(axis=0)   # (N_joints,)
    return (state_matrix - mean[np.newaxis, :]) / (std[np.newaxis, :] + eps)
