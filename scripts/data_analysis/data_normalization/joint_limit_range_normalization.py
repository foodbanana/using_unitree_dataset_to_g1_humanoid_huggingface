"""
joint_limit_range_normalization.py
────────────────────────────────────────────────────────────────────
정규화 방법 : Joint Limit Range Normalization

수식 : norm_i(t) = x_i(t) / (Upper_i - Lower_i)

- Upper_i, Lower_i : docs/joint_limits_by_limb.md 에 정의된 물리적 관절 한계
- 결과값의 의미   : 해당 관절이 물리적 가동 범위 중 얼마나 움직였는지의 비율
- Zero-range 관절 : Upper == Lower 인 더미/고정 관절은 0.0 으로 마스킹
                    (tip joint 등)

소스 파일 (docs/joint_limits_by_limb.md):
    - assets/g1_base/g1_29dof_rev_1_0.urdf
    - assets/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf
    - assets/g1_with_dex3_hand/g1_with_dex3_hand.xml   (MuJoCo menagerie 기준)
    - assets/g1_with_inspire_hand/ ... (DFQ URDF 기준)
────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import numpy as np

# ─── 하이퍼파라미터 ──────────────────────────────────────────────────────────
EPSILON = 1e-6   # Zero-range 판정 기준 (limit_range < EPSILON 이면 고정 관절로 간주)

# 그래프 y축 레이블 (전처리 파이프라인에서 참조용)
Y_LABEL = "limit-range normalized"


# ════════════════════════════════════════════════════════════════════
# 관절 한계 딕셔너리
# key   : HuggingFace 데이터셋 joint 이름 (kPascalCase)
# value : (Lower, Upper) in radians
#
# 출처 : docs/joint_limits_by_limb.md
#        + HF to MuJoCo 매핑 테이블
# ════════════════════════════════════════════════════════════════════

# ── 몸통 관절 (모든 설정 공통) ─────────────────────────────────────────────
_BASE_LIMITS: dict[str, tuple[float, float]] = {
    # ── 다리 ──────────────────────────────────────────────────────────────
    "kLeftHipPitch":      (-2.5307,        2.8798),
    "kLeftHipRoll":       (-0.5236,        2.9671),
    "kLeftHipYaw":        (-2.7576,        2.7576),
    "kLeftKnee":          (-0.087267,      2.8798),
    "kLeftAnklePitch":    (-0.87267,       0.5236),
    "kLeftAnkleRoll":     (-0.2618,        0.2618),
    "kRightHipPitch":     (-2.5307,        2.8798),
    "kRightHipRoll":      (-2.9671,        0.5236),
    "kRightHipYaw":       (-2.7576,        2.7576),
    "kRightKnee":         (-0.087267,      2.8798),
    "kRightAnklePitch":   (-0.87267,       0.5236),
    "kRightAnkleRoll":    (-0.2618,        0.2618),
    # ── 허리 (WBT 전용) ───────────────────────────────────────────────────
    "kWaistYaw":          (-2.618,         2.618),
    "kWaistRoll":         (-0.52,          0.52),
    "kWaistPitch":        (-0.52,          0.52),
    # ── 팔 ────────────────────────────────────────────────────────────────
    "kLeftShoulderPitch": (-3.0892,        2.6704),
    "kLeftShoulderRoll":  (-1.5882,        2.2515),
    "kLeftShoulderYaw":   (-2.618,         2.618),
    "kLeftElbow":         (-1.0472,        2.0944),
    "kLeftWristRoll":     (-1.972222054,   1.972222054),
    "kLeftWristPitch":    (-1.614429558,   1.614429558),
    "kLeftWristYaw":      (-1.614429558,   1.614429558),
    "kRightShoulderPitch":(-3.0892,        2.6704),
    "kRightShoulderRoll": (-2.2515,        1.5882),
    "kRightShoulderYaw":  (-2.618,         2.618),
    "kRightElbow":        (-1.0472,        2.0944),
    "kRightWristRoll":    (-1.972222054,   1.972222054),
    "kRightWristPitch":   (-1.614429558,   1.614429558),
    "kRightWristYaw":     (-1.614429558,   1.614429558),
}

# ── BrainCo 핸드 관절 ──────────────────────────────────────────────────────
# HF key → MuJoCo joint → (Lower, Upper)
# 출처: HF-to-MuJoCo 매핑 + G1 + BrainCo hand URDF
_BRAINCO_HAND_LIMITS: dict[str, tuple[float, float]] = {
    "kLeftHandThumb":     (0.0,  1.0472),   # left_thumb_proximal_joint
    "kLeftHandThumbAux":  (0.0,  1.5184),   # left_thumb_metacarpal_joint
    "kLeftHandIndex":     (0.0,  1.4661),   # left_index_proximal_joint
    "kLeftHandMiddle":    (0.0,  1.4661),   # left_middle_proximal_joint
    "kLeftHandRing":      (0.0,  1.4661),   # left_ring_proximal_joint
    "kLeftHandPinky":     (0.0,  1.4661),   # left_pinky_proximal_joint
    "kRightHandThumb":    (0.0,  1.0472),   # right_thumb_proximal_joint
    "kRightHandThumbAux": (0.0,  1.5184),   # right_thumb_metacarpal_joint
    "kRightHandIndex":    (0.0,  1.4661),   # right_index_proximal_joint
    "kRightHandMiddle":   (0.0,  1.4661),   # right_middle_proximal_joint
    "kRightHandRing":     (0.0,  1.4661),   # right_ring_proximal_joint
    "kRightHandPinky":    (0.0,  1.4661),   # right_pinky_proximal_joint
}

# ── Dex3 핸드 관절 ─────────────────────────────────────────────────────────
# 출처: MuJoCo menagerie (g1_with_hands.xml) - 더 정확한 한계값
_DEX3_HAND_LIMITS: dict[str, tuple[float, float]] = {
    "kLeftHandThumb0":    (-1.0472,   1.0472),    # left_hand_thumb_0_joint
    "kLeftHandThumb1":    (-0.724312, 1.0472),    # left_hand_thumb_1_joint
    "kLeftHandThumb2":    ( 0.0,      1.74533),   # left_hand_thumb_2_joint
    "kLeftHandMiddle0":   (-1.5708,   0.0),       # left_hand_middle_0_joint
    "kLeftHandMiddle1":   (-1.74533,  0.0),       # left_hand_middle_1_joint
    "kLeftHandIndex0":    (-1.5708,   0.0),       # left_hand_index_0_joint
    "kLeftHandIndex1":    (-1.74533,  0.0),       # left_hand_index_1_joint
    "kRightHandThumb0":   (-1.0472,   1.0472),    # right_hand_thumb_0_joint
    "kRightHandThumb1":   (-1.0472,   0.724312),  # right_hand_thumb_1_joint
    "kRightHandThumb2":   (-1.74533,  0.0),       # right_hand_thumb_2_joint
    "kRightHandMiddle0":  ( 0.0,      1.5708),    # right_hand_middle_0_joint
    "kRightHandMiddle1":  ( 0.0,      1.74533),   # right_hand_middle_1_joint
    "kRightHandIndex0":   ( 0.0,      1.5708),    # right_hand_index_0_joint
    "kRightHandIndex1":   ( 0.0,      1.74533),   # right_hand_index_1_joint
}

# ── Inspire 핸드 관절 ──────────────────────────────────────────────────────
# 출처: G1 + Inspire hand (DFQ URDF)
_INSPIRE_HAND_LIMITS: dict[str, tuple[float, float]] = {
    "kLeftHandIndex":     ( 0.0, 1.7),    # L_index_proximal_joint
    "kLeftHandMiddle":    ( 0.0, 1.7),    # L_middle_proximal_joint
    "kLeftHandRing":      ( 0.0, 1.7),    # L_ring_proximal_joint
    "kLeftHandPinky":     ( 0.0, 1.7),    # L_pinky_proximal_joint
    "kLeftHandThumb":     (-0.1, 0.6),    # L_thumb_proximal_pitch_joint
    "kLeftHandThumbAux":  (-0.1, 1.3),    # L_thumb_proximal_yaw_joint
    "kRightHandIndex":    ( 0.0, 1.7),    # R_index_proximal_joint
    "kRightHandMiddle":   ( 0.0, 1.7),    # R_middle_proximal_joint
    "kRightHandRing":     ( 0.0, 1.7),    # R_ring_proximal_joint
    "kRightHandPinky":    ( 0.0, 1.7),    # R_pinky_proximal_joint
    "kRightHandThumb":    (-0.1, 0.6),    # R_thumb_proximal_pitch_joint
    "kRightHandThumbAux": (-0.1, 1.3),    # R_thumb_proximal_yaw_joint
}

# ── hand_type → 핸드 한계 딕셔너리 매핑 ───────────────────────────────────
_HAND_LIMITS_MAP: dict[str, dict[str, tuple[float, float]]] = {
    "g1_with_brainco": _BRAINCO_HAND_LIMITS,
    "g1_with_dex3":    _DEX3_HAND_LIMITS,
    "g1_with_inspire": _INSPIRE_HAND_LIMITS,
}


# ════════════════════════════════════════════════════════════════════
# 공개 API
# ════════════════════════════════════════════════════════════════════
def build_limit_range_dict(hand_type: str) -> dict[str, float]:
    """
    hand_type 에 맞는 (몸통 + 핸드) 전체 limit range 딕셔너리를 반환한다.

    Args:
        hand_type : "g1_with_brainco" | "g1_with_dex3" | "g1_with_inspire"
    Returns:
        {joint_name: limit_range}  (limit_range = Upper - Lower)
    """
    hand_limits = _HAND_LIMITS_MAP.get(hand_type)
    if hand_limits is None:
        raise ValueError(
            f"지원하지 않는 hand_type: '{hand_type}'. "
            f"선택 가능: {list(_HAND_LIMITS_MAP.keys())}"
        )
    combined = {**_BASE_LIMITS, **hand_limits}
    return {name: (upper - lower) for name, (lower, upper) in combined.items()}


def build_lower_upper_vectors(
    joint_order: list[str],
    hand_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    joint_order 순서에 맞는 (Lower 벡터, Upper 벡터) 를 반환한다.
    joint_order 에 없는 관절 → (0.0, 0.0)

    Args:
        joint_order : HF 데이터셋 관절 이름 순서 리스트
        hand_type   : "g1_with_brainco" | "g1_with_dex3" | "g1_with_inspire"
    Returns:
        lower_vec : (N_joints,) ndarray
        upper_vec : (N_joints,) ndarray
    """
    hand_limits = _HAND_LIMITS_MAP.get(hand_type)
    if hand_limits is None:
        raise ValueError(
            f"지원하지 않는 hand_type: '{hand_type}'. "
            f"선택 가능: {list(_HAND_LIMITS_MAP.keys())}"
        )
    combined = {**_BASE_LIMITS, **hand_limits}
    lower_vec = np.array(
        [combined.get(name, (0.0, 0.0))[0] for name in joint_order], dtype=float
    )
    upper_vec = np.array(
        [combined.get(name, (0.0, 0.0))[1] for name in joint_order], dtype=float
    )
    return lower_vec, upper_vec


def build_limit_range_vector(
    joint_order: list[str],
    hand_type: str,
    eps: float = EPSILON,
) -> np.ndarray:
    """
    joint_order 순서에 맞는 limit range 벡터 (N_joints,) 를 반환한다.

    joint_order 에 없는 관절 → 0.0 (zero-range 처리와 동일하게 마스킹됨)
    limit_range < eps 인 관절 → 0.0 (Upper == Lower 고정 관절)

    Args:
        joint_order : HF 데이터셋 관절 이름 순서 리스트
        hand_type   : "g1_with_brainco" | "g1_with_dex3" | "g1_with_inspire"
        eps         : zero-range 판정 임계값
    Returns:
        (N_joints,) ndarray — limit range 벡터
    """
    limit_dict = build_limit_range_dict(hand_type)
    ranges = np.array(
        [limit_dict.get(name, 0.0) for name in joint_order],
        dtype=float,
    )
    # 음수 range (잘못된 데이터) 도 0.0 으로 처리
    ranges = np.where(ranges > eps, ranges, 0.0)
    return ranges


def apply_limit_range_normalization(
    state_matrix: np.ndarray,
    joint_order: list[str],
    hand_type: str,
    eps: float = EPSILON,
) -> np.ndarray:
    """
    각 관절 각도를 물리적 가동 범위(Upper - Lower)로 나눠 정규화한다.

    처리 규칙:
        - limit_range > eps : state / limit_range
        - limit_range == 0  : 고정 관절(tip joint 등) → 0.0 으로 마스킹
        - joint_order 에 limit 정보가 없는 관절 → 0.0 으로 마스킹

    Args:
        state_matrix : (N_frames, N_joints) — 필터링된 관절각 행렬
        joint_order  : 관절 이름 리스트 (길이 == N_joints)
        hand_type    : "g1_with_brainco" | "g1_with_dex3" | "g1_with_inspire"
        eps          : zero-range 판정 임계값
    Returns:
        (N_frames, N_joints) — limit range 정규화 결과
    """
    limit_ranges = build_limit_range_vector(joint_order, hand_type, eps)

    # zero-range 관절 마스크
    valid_mask = limit_ranges > eps          # (N_joints,) bool

    # 나눗셈 수행: zero-range 관절은 임시 1.0 으로 채워 ZeroDivisionError 방지
    safe_ranges = np.where(valid_mask, limit_ranges, 1.0)
    normalized  = state_matrix / safe_ranges[np.newaxis, :]

    # zero-range 관절 출력을 0.0 으로 마스킹
    normalized[:, ~valid_mask] = 0.0

    return normalized


def normalize(
    state_matrix: np.ndarray,
    joint_order: list[str] | None = None,
    hand_type: str | None = None,
    eps: float = EPSILON,
    **_kwargs,
) -> np.ndarray:
    """
    다른 정규화 모듈과 동일한 인터페이스를 제공하는 진입점.
    --norm-method joint_limit_range 로 호출된다.

    Args:
        state_matrix : (N_frames, N_joints) — 필터링된 관절각 행렬
        joint_order  : HF 데이터셋 관절 이름 리스트 (필수)
        hand_type    : "g1_with_brainco" | "g1_with_dex3" | "g1_with_inspire" (필수)
        eps          : zero-range 판정 임계값
    Returns:
        (N_frames, N_joints) — limit range 정규화 결과
    """
    if joint_order is None or hand_type is None:
        raise ValueError(
            "joint_limit_range 정규화는 joint_order 와 hand_type 이 필요합니다."
        )
    return apply_limit_range_normalization(state_matrix, joint_order, hand_type, eps)


def log_limit_ranges(
    joint_order: list[str],
    hand_type: str,
    eps: float = EPSILON,
) -> None:
    """
    각 관절의 limit range 값을 터미널에 출력한다 (디버깅·확인용).

    Args:
        joint_order : 관절 이름 리스트
        hand_type   : 핸드 타입
        eps         : zero-range 판정 임계값
    """
    limit_ranges = build_limit_range_vector(joint_order, hand_type, eps)
    print()
    print("=" * 60)
    print(f"  [Joint Limit Ranges]  hand_type: {hand_type}")
    print("=" * 60)
    for i, (name, r) in enumerate(zip(joint_order, limit_ranges)):
        status = "ZERO (masked)" if r <= eps else f"{r:.6f} rad"
        print(f"  {i:>3}  {name:<32}  {status}")
    print("=" * 60)
    print()
