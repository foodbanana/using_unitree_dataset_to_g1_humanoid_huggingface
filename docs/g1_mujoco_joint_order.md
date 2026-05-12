# MuJoCo G1 + Hand Joint Order

이 문서는 MuJoCo 모델에서 G1 본체와 손(Brainco / Dex3 / Dex1 / Inspire)이 결합된 경우의 관절 순서를 정리한다.  
기본 본체 순서는 모든 변형에서 동일하며, 손 관절이 뒤에 추가된다.

## 목차

- [G1 Base (29 DOF)](#1-g1-base-29-dof)
- [G1 + Brainco Hand (29 + 12 DOF)](#2-g1--brainco-hand-29--12-dof)
- [G1 + Dex3 Hand (29 + 14 DOF)](#3-g1--dex3-hand-29--14-dof)
- [G1 + Dex1 Hand (29 + 4 DOF)](#4-g1--dex1-hand-29--4-dof)
- [G1 + Inspire Hand (29 + 12 DOF)](#5-g1--inspire-hand-29--12-dof)

## 변형 요약

| 모델 | Base DOF | Hand DOF | 총 DOF |
|------|:--------:|:--------:|:------:|
| G1 (base only) | 29 | — | **29** |
| G1 + Brainco | 29 | 12 | **41** |
| G1 + Dex3 | 29 | 14 | **43** |
| G1 + Dex1 | 29 | 4 | **33** |
| G1 + Inspire | 29 | 12 | **41** |

---

## 1) G1 Base (29 DOF)

```text
 0  left_hip_pitch_joint
 1  left_hip_roll_joint
 2  left_hip_yaw_joint
 3  left_knee_joint
 4  left_ankle_pitch_joint
 5  left_ankle_roll_joint
 6  right_hip_pitch_joint
 7  right_hip_roll_joint
 8  right_hip_yaw_joint
 9  right_knee_joint
10  right_ankle_pitch_joint
11  right_ankle_roll_joint
12  waist_yaw_joint
13  waist_roll_joint
14  waist_pitch_joint
15  left_shoulder_pitch_joint
16  left_shoulder_roll_joint
17  left_shoulder_yaw_joint
18  left_elbow_joint
19  left_wrist_roll_joint
20  left_wrist_pitch_joint
21  left_wrist_yaw_joint
22  right_shoulder_pitch_joint
23  right_shoulder_roll_joint
24  right_shoulder_yaw_joint
25  right_elbow_joint
26  right_wrist_roll_joint
27  right_wrist_pitch_joint
28  right_wrist_yaw_joint
```

---

## 2) G1 + Brainco Hand (29 + 12 DOF)

Base(0–28)는 [G1 Base](#1-g1-base-29-dof) 순서와 동일하다.

### Left Hand (29–34)

```text
29  left_thumb_metacarpal_joint
30  left_thumb_proximal_joint
31  left_index_proximal_joint
32  left_middle_proximal_joint
33  left_ring_proximal_joint
34  left_pinky_proximal_joint
```

### Right Hand (35–40)

```text
35  right_thumb_metacarpal_joint
36  right_thumb_proximal_joint
37  right_index_proximal_joint
38  right_middle_proximal_joint
39  right_ring_proximal_joint
40  right_pinky_proximal_joint
```

---

## 3) G1 + Dex3 Hand (29 + 14 DOF)

Base(0–28)는 [G1 Base](#1-g1-base-29-dof) 순서와 동일하다.

### Left Hand (29–35)

```text
29  left_hand_thumb_0_joint
30  left_hand_thumb_1_joint
31  left_hand_thumb_2_joint
32  left_hand_middle_0_joint
33  left_hand_middle_1_joint
34  left_hand_index_0_joint
35  left_hand_index_1_joint
```

### Right Hand (36–42)

```text
36  right_hand_thumb_0_joint
37  right_hand_thumb_1_joint
38  right_hand_thumb_2_joint
39  right_hand_index_0_joint
40  right_hand_index_1_joint
41  right_hand_middle_0_joint
42  right_hand_middle_1_joint
```

---

## 4) G1 + Dex1 Hand (29 + 4 DOF)

Base(0–28)는 [G1 Base](#1-g1-base-29-dof) 순서와 동일하다.

> Dex1은 각 손에 슬라이드 조인트 2개씩, 총 4 DOF.

### Left Hand (29–30)

```text
29  left_dex1_finger_joint_1
30  left_dex1_finger_joint_2
```

### Right Hand (31–32)

```text
31  right_dex1_finger_joint_1
32  right_dex1_finger_joint_2
```

---

## 5) G1 + Inspire Hand (29 + 12 DOF)

Base(0–28)는 [G1 Base](#1-g1-base-29-dof) 순서와 동일하다.

### Left Hand (29–40)

```text
29  L_thumb_proximal_yaw_joint
30  L_thumb_proximal_pitch_joint
31  L_thumb_intermediate_joint
32  L_thumb_distal_joint
33  L_index_proximal_joint
34  L_index_intermediate_joint
35  L_middle_proximal_joint
36  L_middle_intermediate_joint
37  L_ring_proximal_joint
38  L_ring_intermediate_joint
39  L_pinky_proximal_joint
40  L_pinky_intermediate_joint
```

### Right Hand (41–52)

```text
41  R_thumb_proximal_yaw_joint
42  R_thumb_proximal_pitch_joint
43  R_thumb_intermediate_joint
44  R_thumb_distal_joint
45  R_index_proximal_joint
46  R_index_intermediate_joint
47  R_middle_proximal_joint
48  R_middle_intermediate_joint
49  R_ring_proximal_joint
50  R_ring_intermediate_joint
51  R_pinky_proximal_joint
52  R_pinky_intermediate_joint
```
