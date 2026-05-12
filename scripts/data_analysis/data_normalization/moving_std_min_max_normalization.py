"""
moving_std_min_max_normalization.py
─────────────────────────────────────────────────────────────────
정규화 방법 : Moving STD → Min-Max Normalization + 동적 노이즈 임계값

처리 순서   :
    ① 관절별 이동 표준편차(Moving STD) 계산
    ② Moving STD 값을 관절별로 독립적으로 Min-Max 정규화
    ③ 동적 임계값 미달 관절(정지 관절)의 출력을 0.0 으로 마스킹

노이즈 억제 로직 (3-시그마 규칙 개념 적용):
    ① 에피소드 시작 RESTING_FRAMES 프레임(정지 구간)의 관절별 STD = resting_std_i
    ② 에피소드 전체 Moving STD 의 최댓값 = peak_moving_std_i
    ③ 조건 : peak_moving_std_i  ≤  NOISE_MULTIPLIER × resting_std_i
       → 해당 관절은 실질적 움직임 없음으로 판단 → 정규화 결과를 0.0 으로 마스킹

- 각 관절의 움직임 강도를 0~1 로 스케일링
- 정지 관절은 노이즈 증폭 없이 0.0 으로 억제됨
─────────────────────────────────────────────────────────────────
"""
import numpy as np
import pandas as pd

# ─── 하이퍼파라미터 ──────────────────────────────────────────────────────────
STD_WINDOW       = 21    # Moving STD 슬라이딩 윈도우 크기 (프레임). 30fps 기준 ≈ 0.7초
EPSILON          = 1e-6  # max == min 인 관절에서의 0 나눗셈 방지

NOISE_MULTIPLIER = 3.0   # K 값 (3-시그마 규칙).
                          # 에피소드 전체 peak Moving STD 가 K × resting_std 이하이면 정지로 간주.
                          # 값이 클수록 임계값이 높아져 더 많은 관절이 정지 처리됨.
                          # 권장 범위: 2.0 ~ 5.0

RESTING_FRAMES   = 30    # 정지 구간으로 간주할 초기 프레임 수.
                          # 30fps × 1초 = 30프레임 (stop_state_std_per_each_joint.py 와 동일)

# 그래프 y축 레이블
Y_LABEL = "moving-std min-max (0~1)"


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
    n = min(resting_frames, state_matrix.shape[0])
    return state_matrix[:n].std(axis=0)  # (N_joints,)


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
    std_window: int = STD_WINDOW,
    eps: float = EPSILON,
    noise_multiplier: float = NOISE_MULTIPLIER,
    resting_frames: int = RESTING_FRAMES,
    **kwargs,
) -> np.ndarray:
    """
    동적 노이즈 임계값이 적용된 Moving STD → Min-Max 정규화.

    처리 순서:
        ① 정지 구간(초기 resting_frames 프레임) STD 계산 → resting_std
        ② 에피소드 전체 Moving STD 계산
        ③ peak_moving_std(에피소드 내 최대 Moving STD) 계산 → per joint
        ④ peak_moving_std ≤ noise_multiplier × resting_std 인 관절 → 정지 판정
        ⑤ Moving STD 에 Min-Max 정규화 수행
        ⑥ 정지 관절의 출력을 0.0 으로 마스킹

    Args:
        state_matrix     : (N_frames, N_joints) — 필터링된 관절각 행렬
        std_window       : Moving STD 윈도우 크기
        eps              : 0 나눗셈 방지 epsilon
        noise_multiplier : 노이즈 임계값 배수 K
        resting_frames   : 정지 구간 프레임 수
    Returns:
        (N_frames, N_joints) — [0, 1] 범위 정규화 결과 (정지 관절은 0.0)
    """
    # ① 정지 구간 STD 계산 (관절별 resting noise 기준)
    resting_std = _compute_resting_std(state_matrix, resting_frames)  # (N_joints,)

    # ② 에피소드 전체 Moving STD 계산
    moving_std = _compute_moving_std(state_matrix, window=std_window)  # (N_frames, N_joints)

    # ③ 관절별 에피소드 내 최대 Moving STD
    peak_moving_std = moving_std.max(axis=0)  # (N_joints,)

    # ④ 동적 임계값: peak_moving_std > K × resting_std 인 관절만 실질적 움직임으로 판정
    threshold   = noise_multiplier * resting_std  # (N_joints,)
    active_mask = peak_moving_std > threshold      # (N_joints,) bool

    # ⑤ Moving STD 에 Min-Max 정규화
    min_val = moving_std.min(axis=0)  # (N_joints,)
    max_val = moving_std.max(axis=0)  # (N_joints,)
    normalized = (moving_std - min_val[np.newaxis, :]) / (
        (max_val - min_val)[np.newaxis, :] + eps
    )

    # ⑥ 정지 관절(임계값 미달)의 출력을 0.0 으로 마스킹
    normalized[:, ~active_mask] = 0.0

    return normalized
