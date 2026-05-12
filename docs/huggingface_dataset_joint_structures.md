# Hugging Face dataset joint structures

## 1. 상체만 제어할 때 Hugging Face 관절 이름

데이터셋의 1열: `observation.state` (g1 + Dex1 hand의 대다수 데이터셋 제외)

데이터셋의 2열 



### g1 + Dex3 hand [28]

```
kLeftShoulderPitch
kLeftShoulderRoll
kLeftShoulderYaw
kLeftElbow
kLeftWristRoll
kLeftWristPitch
kLeftWristYaw
kRightShoulderPitch
kRightShoulderRoll
kRightShoulderYaw
kRightElbow
kRightWristRoll
kRightWristPitch
kRightWristYaw
# 14 DOF for both hands
kLeftHandThumb0
kLeftHandThumb1
kLeftHandThumb2
kLeftHandMiddle0
kLeftHandMiddle1
kLeftHandIndex0
kLeftHandIndex1
kRightHandThumb0
kRightHandThumb1
kRightHandThumb2
kRightHandIndex0
kRightHandIndex1
kRightHandMiddle0
kRightHandMiddle1
```

### g1 + Brainco hand [26]

```
kLeftShoulderPitch
kLeftShoulderRoll
kLeftShoulderYaw
kLeftElbow
kLeftWristRoll
kLeftWristPitch
kLeftWristYaw
kRightShoulderPitch
kRightShoulderRoll
kRightShoulderYaw
kRightElbow
kRightWristRoll
kRightWristPitch
kRightWristYaw
# 12 DOF for both hands
kLeftHandThumb
kLeftHandThumbAux
kLeftHandIndex
kLeftHandMiddle
kLeftHandRing
kLeftHandPinky
kRightHandThumb
kRightHandThumbAux
kRightHandIndex
kRightHandMiddle
kRightHandRing
kRightHandPinky
```

### g1 + Dex1 hand [16]

상황 1: 2개의 데이터셋에 해당
- `G1_Dex1_MountCamera_Dataset`
- `G1_Dex1_MountCameraRedGripper_Dataset`

제일 왼쪽 열이 [16]

```
kLeftShoulderPitch
kLeftShoulderRoll
kLeftShoulderYaw
kLeftElbow
kLeftWristRoll
kLeftWristPitch
kLeftWristYaw
kRightShoulderPitch
kRightShoulderRoll
kRightShoulderYaw
kRightElbow
kRightWristRoll
kRightWristPitch
kRightWristYaw
# 2 DOF for both hands
kLeftGripper
kRightGripper
```

상황 2: 나머지 데이터셋에 해당

제일 왼쪽 열부터 `observation.left_arm`, `observation.right_arm`

총 DOF: 57

`observation.left_arm` [7]

```
kLeftShoulderPitch
kLeftShoulderRoll
kLeftShoulderYaw
kLeftElbow
kLeftWristRoll
kLeftWristPitch
kLeftWristYaw
```

`observation.right_arm` [7]

```
kRightShoulderPitch
kRightShoulderRoll
kRightShoulderYaw
kRightElbow
kRightWristRoll
kRightWristPitch
kRightWristYaw
```

`observation.left_gripper` [1]

```
kLeftGripper
```

`observation.right_gripper` [1]

```
kRightGripper
```

`observation.left_ee` [6]

```
kLeftEEX
kLeftEEY
kLeftEEZ
kLeftEER
kLeftEEP
kLeftEEY
```

`observation.right_ee` [6]

```
kRightEEX
kRightEEY
kRightEEZ
kRightEER
kRightEEP
kRightEEY
```

`observation.body` [29] (waist 포함)

```
kLeftHipPitch
kLeftHipRoll
kLeftHipYaw
kLeftKnee
kLeftAnklePitch
kLeftAnkleRoll
kRightHipPitch
kRightHipRoll
kRightHipYaw
kRightKnee
kRightAnklePitch
kRightAnkleRoll
kWaistYaw
kWaistRoll
kWaistPitch
kLeftShoulderPitch
kLeftShoulderRoll
kLeftShoulderYaw
kLeftElbow
kLeftWristRoll
kLeftWristPitch
kLeftWristYaw
kRightShoulderPitch
kRightShoulderRoll
kRightShoulderYaw
kRightElbow
kRightWristRoll
kRightWristPitch
kRightWristYaw
```

### g1 + Dex1 diversemap [16]

```
kLeftShoulderPitch
kLeftShoulderRoll
kLeftShoulderYaw
kLeftElbow
kLeftWristRoll
kLeftWristPitch
kLeftWristYaw
kRightShoulderPitch
kRightShoulderRoll
kRightShoulderYaw
kRightElbow
kRightWristRoll
kRightWristPitch
kRightWristYaw
kLeftGripper
kRightGripper
```

## 2. WBT일때 Hugging Face에서 관절 이름

### unitreerobotics/G1_WBT_Inspire

제일 왼쪽 열: `observation.state.ee_state` [12]

Computed via forward kinematics (FK) from the root link to the left and right end-effectors.
Represented as concatenated poses of both end-effectors.

왼쪽에서 2번째 열: `observation.state.hand_state` [12]

Inspire Hand (range: 0.0 - 1.0, open -> close)

각 손마다 있는 관절:
- Index finger
- Middle finger
- Ring finger
- Little finger
- Thumb (open/close)
- Thumb (lateral tilt) # 측면 회전

왼쪽에서 3번째 열: `observation.state.robot_q_current` [36]

Root position: (x, y, z) [3]
Root orientation: quaternion (w, x, y, z) [4]
Robot joint positions. [29]

### unitreerobotics/G1_WBT_Brainco

제일 왼쪽 열: `observation.state.ee_state` [12]

Computed via forward kinematics (FK) from the root link to the left and right end-effectors.
Represented as concatenated poses of both end-effectors.

왼쪽에서 2번째 열: `observation.state.hand_state` [12]

Brainco Hand (range: 0.0 - 1.0, open -> close)

각 손마다 있는 관절:
- Thumb (open/close)
- Thumb (lateral tilt)
- Index finger
- Middle finger
- Ring finger
- Little finger

왼쪽에서 3번째 열: `observation.state.robot_q_current` [36]

Root position: (x, y, z) [3]
Root orientation: quaternion (w, x, y, z) [4]
Robot joint positions [29]

### unitreerobotics/G1_WBT_Dex1

제일 왼쪽 열: `observation.state.ee_state` [12]

Computed via forward kinematics (FK) from the root link to the left and right end-effectors.
Represented as concatenated poses of both end-effectors.

왼쪽에서 2번째 열: `observation.state.hand_state` [2]

Dex1 Hand (range: 5.5 - 0.0, open -> close)

Per hand:
- Open/close

왼쪽에서 3번째 열: `observation.state.robot_q_current` [36]

Root position: (x, y, z) [3]
Root orientation: quaternion (w, x, y, z) [4]
Robot joint positions [29]
