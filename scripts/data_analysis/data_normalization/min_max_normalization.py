"""
min_max_normalization.py
─────────────────────────────────────────────────────────────────
정규화 방법 : Min-Max Normalization + 동적 노이즈 임계값(Dynamic Noise Threshold)

수식        : norm_i(t) = (x_i(t) - min_i) / (max_i - min_i + ε)

노이즈 억제 로직 (3-시그마 규칙 개념 적용):
    ① 에피소드 시작 RESTING_FRAMES 프레임(정지 구간)의 관절별 STD = resting_std_i
    ② 에피소드 전체 관절 변화량(range) = max_i - min_i
    ③ 조건 : range_i  ≤  NOISE_MULTIPLIER × resting_std_i
       → 해당 관절은 실질적 움직임 없음으로 판단 → 정규화 결과를 0.0 으로 마스킹

- min_i, max_i, resting_std_i : 해당 에피소드 내에서 관절별 독립 계산
- 결과 범위: [0.0, 1.0]  (정지 관절은 0.0 으로 고정)
─────────────────────────────────────────────────────────────────
"""
import numpy as np

# ─── 하이퍼파라미터 ──────────────────────────────────────────────────────────
EPSILON          = 1e-6  # max == min 인 관절에서의 0 나눗셈 방지

NOISE_MULTIPLIER = 3.0   # K 값 (3-시그마 규칙).
                          # 에피소드 전체 range 가 K × resting_std 이하이면 정지로 간주.
                          # 값이 클수록 임계값이 높아져 더 많은 관절이 정지 처리됨.
                          # 권장 범위: 2.0 ~ 5.0

RESTING_FRAMES   = 30    # 정지 구간으로 간주할 초기 프레임 수.
                          # 30fps × 1초 = 30프레임 (stop_state_std_per_each_joint.py 와 동일)

# 그래프 y축 레이블
Y_LABEL = "min-max (0~1)"


def _compute_resting_std(state_matrix: np.ndarray, resting_frames: int) -> np.ndarray:
    """
    에피소드 초기 정지 구간의 관절별 표준편차를 계산한다.
    stop_state_std_per_each_joint.py 의 로직을 에피소드 단위로 적용한 것이다.

    Args:
        state_matrix   : (N_frames, N_joints) — 필터링된 관절각 행렬
        resting_frames : 정지 구간으로 볼 초기 프레임 수
    Returns:
        (N_joints,) — 관절별 정지 구간 STD
    """
    # 에피소드가 resting_frames 보다 짧으면 가능한 전부 사용
    n = min(resting_frames, state_matrix.shape[0])
    return state_matrix[:n].std(axis=0)  # (N_joints,)


def normalize(
    state_matrix: np.ndarray,
    eps: float = EPSILON,
    noise_multiplier: float = NOISE_MULTIPLIER,
    resting_frames: int = RESTING_FRAMES,
    **kwargs,
) -> np.ndarray:
    """
    동적 노이즈 임계값이 적용된 관절별 Min-Max 정규화.

    처리 순서:
        ① 정지 구간(초기 resting_frames 프레임) STD 계산 → resting_std
        ② 에피소드 전체 range(max - min) 계산 → per joint
        ③ range ≤ noise_multiplier × resting_std 인 관절 → 정지 관절로 판정
        ④ Min-Max 정규화 수행
        ⑤ 정지 관절의 출력을 0.0 으로 마스킹

    Args:
        state_matrix     : (N_frames, N_joints) — 필터링된 관절각 행렬
        eps              : 0 나눗셈 방지 epsilon
        noise_multiplier : 노이즈 임계값 배수 K
        resting_frames   : 정지 구간 프레임 수
    Returns:
        (N_frames, N_joints) — [0, 1] 범위 정규화 결과 (정지 관절은 0.0)
    """
    # ① 정지 구간 STD 계산 (관절별 resting noise 기준)
    resting_std = _compute_resting_std(state_matrix, resting_frames)  # (N_joints,)

    # ② 에피소드 전체 관절 변화량(range) 계산
    min_val   = state_matrix.min(axis=0)  # (N_joints,)
    max_val   = state_matrix.max(axis=0)  # (N_joints,)
    variation = max_val - min_val          # (N_joints,)

    # ③ 동적 임계값: range > K × resting_std 인 관절만 실질적 움직임으로 판정
    threshold   = noise_multiplier * resting_std          # (N_joints,)
    active_mask = variation > threshold                    # (N_joints,) bool

    # ④ Min-Max 정규화
    normalized = (state_matrix - min_val[np.newaxis, :]) / (
        variation[np.newaxis, :] + eps
    )

    # ⑤ 정지 관절(임계값 미달)의 출력을 0.0 으로 마스킹
    normalized[:, ~active_mask] = 0.0

    return normalized
