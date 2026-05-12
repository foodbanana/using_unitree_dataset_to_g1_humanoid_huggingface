"""
joint_limit_range_0to1_normalization.py
────────────────────────────────────────────────────────────────────
정규화 방법 : Joint Limit Range [0, 1] Normalization

수식 :
            x_t - Lower
    norm = ─────────────────
            Upper - Lower

- Lower, Upper : docs/joint_limits_by_limb.md 에 정의된 물리적 관절 한계
- 결과 범위    : [0.0, 1.0]
  - x_t == Lower 이면 0.0 (관절이 최솟값에 위치)
  - x_t == Upper 이면 1.0 (관절이 최댓값에 위치)

joint_limit_range_normalization.py 와의 차이 :
  - joint_limit_range      : x_t / (Upper - Lower)
                              → Lower 기준 이동 없이 범위로만 나눔
  - joint_limit_range_0to1 : (x_t - Lower) / (Upper - Lower)
                              → Lower 를 뺀 뒤 범위로 나눔 → 완전한 [0, 1]
────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import numpy as np

from data_normalization.joint_limit_range_normalization import (
    build_lower_upper_vectors,
    EPSILON,
)

# 그래프 y축 레이블
Y_LABEL = "limit-range 0~1 normalized"


def normalize(
    state_matrix: np.ndarray,
    joint_order: list[str] | None = None,
    hand_type: str | None = None,
    eps: float = EPSILON,
    **_kwargs,
) -> np.ndarray:
    """
    물리적 관절 한계 기준 [0, 1] 정규화.

    수식: norm_i(t) = (x_i(t) - Lower_i) / (Upper_i - Lower_i)

    처리 규칙:
        - Upper > Lower : (x - Lower) / (Upper - Lower)
        - Upper == Lower: 고정 관절 → 0.0 마스킹

    Args:
        state_matrix : (N_frames, N_joints) — 필터링된 관절각 행렬
        joint_order  : 관절 이름 리스트 (필수)
        hand_type    : "g1_with_brainco" | "g1_with_dex3" | "g1_with_inspire" (필수)
        eps          : zero-range 판정 임계값
    Returns:
        (N_frames, N_joints) — [0, 1] 범위 정규화 결과
    """
    if joint_order is None or hand_type is None:
        raise ValueError(
            "joint_limit_range_0to1 정규화는 joint_order 와 hand_type 이 필요합니다."
        )

    lower_vec, upper_vec = build_lower_upper_vectors(joint_order, hand_type)
    range_vec  = upper_vec - lower_vec           # (N_joints,)

    valid_mask = range_vec > eps                  # zero-range 관절 마스크
    safe_range = np.where(valid_mask, range_vec, 1.0)   # 0 나눗셈 방지

    # (x - Lower) / (Upper - Lower)
    normalized = (state_matrix - lower_vec[np.newaxis, :]) / safe_range[np.newaxis, :]

    normalized[:, ~valid_mask] = 0.0             # zero-range 관절 마스킹

    return normalized
