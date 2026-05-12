#### 각 에피소드가 시작하고 1초동안(정지, jitter상태)일때 각 관절의 std값
```text
터미널 상황 : 
# 가상환경 활성화
cd ~/g1_datasets_huggingface && source g1_datasets/bin/activate


<사용법>
# Brainco (기본값)
python scripts/data_analysis/stop_state_std_per_each_joint.py

# Dex3
python scripts/data_analysis/stop_state_std_per_each_joint.py \
  --folder-name UnifoLM_G1_Dex3_Dataset \
  --dataset-name datasets--unitreerobotics--G1_Dex3_PickBottle_Dataset

# WBT Brainco
python scripts/data_analysis/stop_state_std_per_each_joint.py \
  --folder-name UnifoLM_WBT_Dataset \
  --dataset-name datasets--unitreerobotics--G1_WBT_Brainco_Pickup_Pillow

# WBT Inspire
python scripts/data_analysis/stop_state_std_per_each_joint.py \
  --folder-name UnifoLM_WBT_Dataset \
  --dataset-name datasets--unitreerobotics--G1_WBT_Inspire_Put_Clothes_Into_Basket

# 지원 데이터셋 목록
python scripts/data_analysis/stop_state_std_per_each_joint.py --list-supported-datasets

- 각 에피소드가 시작하고 처음 1초동안 jitter하는 관절별 std값들 출력 결과

📁 UnifoLM_G1_Brainco_Dataset

분석 데이터셋명   : datasets--unitreerobotics--G1_Dex3_PickBottle_Dataset
제어 타입    : Upper_body_control
#### Stop-State Jitter STD per Joint (first 1 second of each episode)

This document shows the per-joint standard deviation (STD) during the first 1 second
of each episode (the initial stop / jitter period). It includes usage examples,
short interpretation notes, and the raw output snippets produced by
`scripts/data_analysis/stop_state_std_per_each_joint.py`.

**Prerequisites**

- Activate the virtualenv from the repository root:

```bash
cd ~/g1_datasets_huggingface
source g1_datasets/bin/activate
```

**Usage examples**

- Default (Brainco upper-body datasets):

```bash
python scripts/data_analysis/stop_state_std_per_each_joint.py
```

- Dex3 example:

```bash
python scripts/data_analysis/stop_state_std_per_each_joint.py \
  --folder-name UnifoLM_G1_Dex3_Dataset \
  --dataset-name datasets--unitreerobotics--G1_Dex3_PickBottle_Dataset
```

- WBT Brainco example:

```bash
python scripts/data_analysis/stop_state_std_per_each_joint.py \
  --folder-name UnifoLM_WBT_Dataset \
  --dataset-name datasets--unitreerobotics--G1_WBT_Brainco_Pickup_Pillow
```

- WBT Inspire example:

```bash
python scripts/data_analysis/stop_state_std_per_each_joint.py \
  --folder-name UnifoLM_WBT_Dataset \
  --dataset-name datasets--unitreerobotics--G1_WBT_Inspire_Put_Clothes_Into_Basket
```

Use `--list-supported-datasets` to print supported dataset names.

**What this script reads**

- For Upper-body control datasets: reads `observation.state`, `frame_index`, `episode_index`.
- For WBT datasets: reads `observation.state.robot_q_current`, `observation.state.hand_state`, `frame_index`, `episode_index`.

The script computes per-joint std across the first second of each episode and
reports the median (across episodes) per joint.

---

**Output summary (selected datasets)**

- UnifoLM_G1_Brainco_Dataset (example: Dex3 PickBottle)

  - Control type : Upper_body_control
  - Hand type    : g1_with_brainco
  - Joint count  : 26
  - DataFrame columns: `['observation.state', 'frame_index', 'episode_index']`

  Median STD (per-joint, shape: 26):

```text
[0.00113495 0.00124103 0.00270395 0.0013799  0.00331397 0.00331996
 0.00413881 0.00128122 0.00137115 0.00380997 0.00103477 0.00399419
 0.0043128  0.00526503 0.0052808  0.00547178 0.00286867 0.00378358
 0.00409931 0.00642297 0.00043478 0.00657833 0.00436445 0.00443848
 0.0046077  0.00553197]
```

- UnifoLM_G1_Dex3_Dataset (PickBottle)

  - Control type : Upper_body_control
  - Hand type    : g1_with_dex3
  - Joint count  : 28
  - DataFrame columns: `['observation.state', 'frame_index', 'episode_index']`

  Median STD (per-joint, shape: 28) — shows very small hand-joint jitter for Dex3:

```text
[0.00381698 0.00164441 0.0022783  0.00597711 0.00450935 0.00666674
 0.00594614 0.00347298 0.00136542 0.00222396 0.0053138  0.0024319
 0.00519755 0.00542086 0.00000242 0.00000447 0.00179376 0.00028976
 0.00001308 0.00005023 0.00000505 0.00000257 0.00000498 0.00298049
 0.00022385 0.00079448 0.00000721 0.00032891]
```

- UnifoLM_WBT_Dataset — G1_WBT_Brainco_Pickup_Pillow

  - Control type : WBT
  - Hand type    : g1_with_brainco
  - Joint count  : 41
  - DataFrame columns: `['observation.state.robot_q_current', 'observation.state.hand_state', 'frame_index', 'episode_index']`

  Median STD (per-joint, shape: 41):

```text
[0.00370005 0.00104861 0.00242181 0.00067034 0.00089338 0.00087221
 0.00341337 0.00099145 0.0023847  0.00050223 0.00094479 0.00076809
 0.00047786 0.00294925 0.00414514 0.00381158 0.00102095 0.00088618
 0.00103296 0.00303243 0.00497727 0.00672888 0.00389246 0.00108886
 0.00074784 0.00051551 0.00124884 0.00157129 0.00370965 0.
 0.00000018 0.         0.         0.         0.         0.
 0.00000018 0.         0.         0.         0.        ]
```

- UnifoLM_WBT_Dataset — G1_WBT_Inspire_Put_Clothes_Into_Basket

  - Control type : WBT
  - Hand type    : g1_with_inspire
  - Joint count  : 41

  Median STD (per-joint, shape: 41) — shows larger wrist/arm jitter for some Inspire tasks:

```text
[0.0048766  0.00156028 0.00716376 0.00014931 0.00178724 0.00145991
 0.00444391 0.00152943 0.00677649 0.00047291 0.00148122 0.0012803
 0.00617846 0.00431598 0.00805455 0.00525989 0.00633424 0.00634309
 0.01039863 0.01649976 0.01026226 0.01672916 0.01130471 0.01119345
 0.01629861 0.02567718 0.05515273 0.02049035 0.03429979 0.00000012
 0.00000024 0.00000024 0.00000018 0.00000006 0.         0.00000012
 0.00000024 0.00000012 0.00000024 0.         0.        ]
```

---

**Interpretation notes**

- For WBT datasets the script concatenates the body joint values (from
  `observation.state.robot_q_current`, excluding the root pose/quaternion) and the
  `observation.state.hand_state`. That produces 41 values for Brainco/Inspire
  (29 body joints + 12 hand joints).
- Small or zero median STD in hand joints often indicates the hand stayed static
  during the initial 1-second window for that dataset/task.

**Raw outputs**

The raw arrays and full per-episode matrices are produced by the script and can
be large; the snippets above are the median-per-joint summaries. Use the
script's output files for complete raw matrices.

```
  21  kLeftWristYaw                       0.00672888
  22  kRightShoulderPitch                 0.00389246
  23  kRightShoulderRoll                  0.00108886
  24  kRightShoulderYaw                   0.00074784
  25  kRightElbow                         0.00051551
  26  kRightWristRoll                     0.00124884
  27  kRightWristPitch                    0.00157129
  28  kRightWristYaw                      0.00370965
  29  kLeftHandThumb                      0.00000000
  30  kLeftHandThumbAux                   0.00000018
  31  kLeftHandIndex                      0.00000000
  32  kLeftHandMiddle                     0.00000000
  33  kLeftHandRing                       0.00000000
  34  kLeftHandPinky                      0.00000000
  35  kRightHandThumb                     0.00000000
  36  kRightHandThumbAux                  0.00000018
  37  kRightHandIndex                     0.00000000
  38  kRightHandMiddle                    0.00000000
  39  kRightHandRing                      0.00000000
  40  kRightHandPinky                     0.00000000
=================================================================

[Raw] median std numpy 배열:
[0.00370005 0.00104861 0.00242181 0.00067034 0.00089338 0.00087221 0.00341337 0.00099145 0.0023847  0.00050223
 0.00094479 0.00076809 0.00047786 0.00294925 0.00414514 0.00381158 0.00102095 0.00088618 0.00103296 0.00303243
 0.00497727 0.00672888 0.00389246 0.00108886 0.00074784 0.00051551 0.00124884 0.00157129 0.00370965 0.
 0.00000018 0.         0.         0.         0.         0.         0.00000018 0.         0.         0.
 0.        ]

[Raw] 전체 에피소드 std 행렬 shape: (300, 41)



#2.G1_WBT_Inspire

분석 데이터셋명   : datasets--unitreerobotics--G1_WBT_Inspire_Put_Clothes_Into_Basket
제어 타입    : WBT
핸드 타입    : g1_with_inspire
관절 수      : 41

=================================================================
  정지 구간(0~1초) 관절별 Jitter 분석 결과
  (에피소드 수: 245, 관절 수: 41)
=================================================================
   #  관절 이름                               Median STD
-----------------------------------------------------------------
   0  kLeftHipPitch                       0.00487660
   1  kLeftHipRoll                        0.00156028
   2  kLeftHipYaw                         0.00716376
   3  kLeftKnee                           0.00014931
   4  kLeftAnklePitch                     0.00178724
   5  kLeftAnkleRoll                      0.00145991
   6  kRightHipPitch                      0.00444391
   7  kRightHipRoll                       0.00152943
   8  kRightHipYaw                        0.00677649
   9  kRightKnee                          0.00047291
  10  kRightAnklePitch                    0.00148122
  11  kRightAnkleRoll                     0.00128030
  12  kWaistYaw                           0.00617846
  13  kWaistRoll                          0.00431598
  14  kWaistPitch                         0.00805455
  15  kLeftShoulderPitch                  0.00525989
  16  kLeftShoulderRoll                   0.00633424
  17  kLeftShoulderYaw                    0.00634309
  18  kLeftElbow                          0.01039863
  19  kLeftWristRoll                      0.01649976
  20  kLeftWristPitch                     0.01026226
  21  kLeftWristYaw                       0.01672916
  22  kRightShoulderPitch                 0.01130471
  23  kRightShoulderRoll                  0.01119345
  24  kRightShoulderYaw                   0.01629861
  25  kRightElbow                         0.02567718
  26  kRightWristRoll                     0.05515273
  27  kRightWristPitch                    0.02049035
  28  kRightWristYaw                      0.03429979
  29  kLeftHandIndex                      0.00000012
  30  kLeftHandMiddle                     0.00000024
  31  kLeftHandRing                       0.00000024
  32  kLeftHandPinky                      0.00000018
  33  kLeftHandThumb                      0.00000006
  34  kLeftHandThumbAux                   0.00000000
  35  kRightHandIndex                     0.00000012
  36  kRightHandMiddle                    0.00000024
  37  kRightHandRing                      0.00000012
  38  kRightHandPinky                     0.00000024
  39  kRightHandThumb                     0.00000000
  40  kRightHandThumbAux                  0.00000000
=================================================================

[Raw] median std numpy 배열:
[0.0048766  0.00156028 0.00716376 0.00014931 0.00178724 0.00145991 0.00444391 0.00152943 0.00677649 0.00047291
 0.00148122 0.0012803  0.00617846 0.00431598 0.00805455 0.00525989 0.00633424 0.00634309 0.01039863 0.01649976
 0.01026226 0.01672916 0.01130471 0.01119345 0.01629861 0.02567718 0.05515273 0.02049035 0.03429979 0.00000012
 0.00000024 0.00000024 0.00000018 0.00000006 0.         0.00000012 0.00000024 0.00000012 0.00000024 0.
 0.        ]

[Raw] 전체 에피소드 std 행렬 shape: (245, 41)





```
