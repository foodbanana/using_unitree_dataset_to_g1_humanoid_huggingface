import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt

# 1. 데이터 로드
ds = load_dataset("unitreerobotics/G1_Dex3_PickBottle_Dataset")
train_data = ds['train']

# 2. 전체 데이터를 넘파이 배열로 변환
states = np.array(train_data['observation.state']) # (176774, 28)

print("=== 데이터 통계 분석 ===")
all_stds = []
for i in range(28):
    col_min = np.min(states[:, i])
    col_max = np.max(states[:, i])
    col_std = np.std(states[:, i])
    all_stds.append(col_std) # 나중에 정렬하기 위해 저장
    
    status = "Active" if col_std > 0.01 else "Fixed/Static"
    print(f"Joint {i:02d} | Min: {col_min:6.3f} | Max: {col_max:6.3f} | Std: {col_std:6.3f} | [{status}]")

# --- 추가된 부분: 상위 5개 관절 출력 ---
print("\n" + "="*40)
print("🎯 가장 역동적인 상위 5개 관절 (Top 5 Active Joints)")
print("="*40)

# std 값 기준 내림차순 정렬하여 인덱스 추출
top_5_indices = np.argsort(all_stds)[-5:][::-1]

for rank, idx in enumerate(top_5_indices, 1):
    print(f"순위 {rank} | Joint {idx:02d} | Std: {all_stds[idx]:.4f}")
print("="*40)
# ----------------------------------------

# 3. 특정 에피소드 시각화 (0~27번 전체)
plt.figure(figsize=(12, 6))
for i in range(28): 
    # 상위 5개는 더 굵게 표시해서 눈에 띄게 할 수도 있습니다.
    linewidth = 3.0 if i in top_5_indices else 1.0
    alpha = 1.0 if i in top_5_indices else 0.4
    plt.plot(states[:500, i], label=f'Joint {i}', linewidth=linewidth, alpha=alpha)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2) # 범례를 옆으로 밀어 가독성 향상
plt.title("Trajectory of First 500 frames (Highlighted Top 5)")
plt.xlabel("Frame")
plt.ylabel("Value (Radian)")
plt.tight_layout()
plt.show()