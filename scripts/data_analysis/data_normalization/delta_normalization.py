"""
delta_normalization.py
─────────────────────────────────────────────────────────────────
정규화 방법 : Delta (변화율) Normalization
수식        : δ_i(t) = (x_i(t) - x_i(t-1)) / (max|δ_i| + ε)

- 프레임 간 관절 변화량을 관절별 최대 절대 변화량으로 정규화
- 움직임의 방향(+/-) 과 상대적 크기를 함께 표현
- 첫 프레임은 0 으로 패딩
─────────────────────────────────────────────────────────────────
"""
import numpy as np

# ─── 하이퍼파라미터 ──────────────────────────────────────────────────────────
EPSILON = 1e-6  # 전혀 움직이지 않는 관절에서의 0 나눗셈 방지

# 그래프 y축 레이블
Y_LABEL = "delta (normalized)"


def normalize(
    state_matrix: np.ndarray,
    eps: float = EPSILON,
    **kwargs,
) -> np.ndarray:
    """
    Delta 정규화.

    각 관절의 프레임 간 변화량을 해당 관절의 최대 절대 변화량으로 정규화한다.

    Args:
        state_matrix : (N_frames, N_joints) — 필터링된 관절각 행렬
        eps          : 0 나눗셈 방지 epsilon
    Returns:
        (N_frames, N_joints) — 정규화된 프레임 간 변화량 [-1, 1]
    """
    # 프레임 간 차분 → (N_frames-1, N_joints)
    delta = np.diff(state_matrix, axis=0)
    # 첫 행 0 패딩 → (N_frames, N_joints)
    delta = np.vstack([np.zeros((1, state_matrix.shape[1]), dtype=float), delta])
    # 관절별 최대 절대 변화량으로 정규화
    max_abs = np.abs(delta).max(axis=0)  # (N_joints,)
    return delta / (max_abs[np.newaxis, :] + eps)
