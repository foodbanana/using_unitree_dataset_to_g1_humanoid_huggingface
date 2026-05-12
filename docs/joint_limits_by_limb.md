# Joint Limits by Limb (Model Files)

Source files:
- assets/g1_base/g1_29dof_rev_1_0.urdf
- assets/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf
- assets/g1_with_dex3_hand/g1_with_dex3_hand.xml
- assets/g1_with_inspire_hand/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf
- assets/g1_with_inspire_hand/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf
- mujoco_menagerie/unitree_g1/g1_with_hands.xml --> more accurate about dex3 hand joint limit 

Units: radians. Values reflect what is written in the model files.

## Dataset Coverage and Joint Name Mapping

The tables below list all joints that appear in the URDF/XML files. The actual Hugging Face dataset typically includes only a subset of these joints. For example, the BrainCo hand has 6 actuated DOF per hand, and the dataset outputs 6 joint values per hand accordingly.

The dataset-to-MuJoCo joint name mappings used in this project are defined in [scripts/mujoco_visualization/hf2mujoco_joint_mapping/hf_dataset_to_mujoco_mapping.py](scripts/mujoco_visualization/hf2mujoco_joint_mapping/hf_dataset_to_mujoco_mapping.py).

### HF to MuJoCo Mapping (BrainCo hand)

| HF dataset key | MuJoCo joint |
| --- | --- |
| `kLeftHandThumb` | `left_thumb_proximal_joint` |
| `kLeftHandThumbAux` | `left_thumb_metacarpal_joint` |
| `kLeftHandIndex` | `left_index_proximal_joint` |
| `kLeftHandMiddle` | `left_middle_proximal_joint` |
| `kLeftHandRing` | `left_ring_proximal_joint` |
| `kLeftHandPinky` | `left_pinky_proximal_joint` |
| `kRightHandThumb` | `right_thumb_proximal_joint` |
| `kRightHandThumbAux` | `right_thumb_metacarpal_joint` |
| `kRightHandIndex` | `right_index_proximal_joint` |
| `kRightHandMiddle` | `right_middle_proximal_joint` |
| `kRightHandRing` | `right_ring_proximal_joint` |
| `kRightHandPinky` | `right_pinky_proximal_joint` |

### HF to MuJoCo Mapping (Dex3-1 hand)

| HF dataset key | MuJoCo joint |
| --- | --- |
| `kLeftHandThumb0` | `left_hand_thumb_0_joint` |
| `kLeftHandThumb1` | `left_hand_thumb_1_joint` |
| `kLeftHandThumb2` | `left_hand_thumb_2_joint` |
| `kLeftHandMiddle0` | `left_hand_middle_0_joint` |
| `kLeftHandMiddle1` | `left_hand_middle_1_joint` |
| `kLeftHandIndex0` | `left_hand_index_0_joint` |
| `kLeftHandIndex1` | `left_hand_index_1_joint` |
| `kRightHandThumb0` | `right_hand_thumb_0_joint` |
| `kRightHandThumb1` | `right_hand_thumb_1_joint` |
| `kRightHandThumb2` | `right_hand_thumb_2_joint` |
| `kRightHandIndex0` | `right_hand_index_0_joint` |
| `kRightHandIndex1` | `right_hand_index_1_joint` |
| `kRightHandMiddle0` | `right_hand_middle_0_joint` |
| `kRightHandMiddle1` | `right_hand_middle_1_joint` |

### HF to MuJoCo Mapping (Inspire hand)

| HF dataset key | MuJoCo joint |
| --- | --- |
| `kLeftHandIndex` | `L_index_proximal_joint` |
| `kLeftHandMiddle` | `L_middle_proximal_joint` |
| `kLeftHandRing` | `L_ring_proximal_joint` |
| `kLeftHandPinky` | `L_pinky_proximal_joint` |
| `kLeftHandThumb` | `L_thumb_proximal_pitch_joint` |
| `kLeftHandThumbAux` | `L_thumb_proximal_yaw_joint` |
| `kRightHandIndex` | `R_index_proximal_joint` |
| `kRightHandMiddle` | `R_middle_proximal_joint` |
| `kRightHandRing` | `R_ring_proximal_joint` |
| `kRightHandPinky` | `R_pinky_proximal_joint` |
| `kRightHandThumb` | `R_thumb_proximal_pitch_joint` |
| `kRightHandThumbAux` | `R_thumb_proximal_yaw_joint` |

## Unitree G1 (base)

### Legs

| Joint | Lower | Upper |
| --- | ---: | ---: |
| left_hip_pitch_joint | -2.5307 | 2.8798 |
| left_hip_roll_joint | -0.5236 | 2.9671 |
| left_hip_yaw_joint | -2.7576 | 2.7576 |
| left_knee_joint | -0.087267 | 2.8798 |
| left_ankle_pitch_joint | -0.87267 | 0.5236 |
| left_ankle_roll_joint | -0.2618 | 0.2618 |
| right_hip_pitch_joint | -2.5307 | 2.8798 |
| right_hip_roll_joint | -2.9671 | 0.5236 |
| right_hip_yaw_joint | -2.7576 | 2.7576 |
| right_knee_joint | -0.087267 | 2.8798 |
| right_ankle_pitch_joint | -0.87267 | 0.5236 |
| right_ankle_roll_joint | -0.2618 | 0.2618 |

### Arms

| Joint | Lower | Upper |
| --- | ---: | ---: |
| left_shoulder_pitch_joint | -3.0892 | 2.6704 |
| left_shoulder_roll_joint | -1.5882 | 2.2515 |
| left_shoulder_yaw_joint | -2.618 | 2.618 |
| left_elbow_joint | -1.0472 | 2.0944 |
| left_wrist_roll_joint | -1.972222054 | 1.972222054 |
| left_wrist_pitch_joint | -1.614429558 | 1.614429558 |
| left_wrist_yaw_joint | -1.614429558 | 1.614429558 |
| right_shoulder_pitch_joint | -3.0892 | 2.6704 |
| right_shoulder_roll_joint | -2.2515 | 1.5882 |
| right_shoulder_yaw_joint | -2.618 | 2.618 |
| right_elbow_joint | -1.0472 | 2.0944 |
| right_wrist_roll_joint | -1.972222054 | 1.972222054 |
| right_wrist_pitch_joint | -1.614429558 | 1.614429558 |
| right_wrist_yaw_joint | -1.614429558 | 1.614429558 |

### Hands

No hand joints in this model.

## G1 + BrainCo hand

### Legs

Same as Unitree G1 (base).

### Arms

Same as Unitree G1 (base).

### Hands

| Joint | Lower | Upper |
| --- | ---: | ---: |
| left_thumb_metacarpal_joint | 0.0 | 1.5184 |
| left_thumb_proximal_joint | 0.0 | 1.0472 |
| left_thumb_distal_joint | 0.0 | 1.0472 |
| left_thumb_tip_joint | 0.0 | 0.0 |
| left_index_proximal_joint | 0.0 | 1.4661 |
| left_index_distal_joint | 0.0 | 1.693 |
| left_index_tip_joint | 1.0 | 1.0 |
| left_middle_proximal_joint | 0.0 | 1.4661 |
| left_middle_distal_joint | 0.0 | 1.693 |
| left_middle_tip_joint | 1.0 | 1.0 |
| left_ring_proximal_joint | 0.0 | 1.4661 |
| left_ring_distal_joint | 0.0 | 1.693 |
| left_ring_tip_joint | 1.0 | 1.0 |
| left_pinky_proximal_joint | 0.0 | 1.4661 |
| left_pinky_distal_joint | 0.0 | 1.693 |
| left_pinky_tip_joint | 1.0 | 1.0 |
| right_thumb_metacarpal_joint | 0.0 | 1.5184 |
| right_thumb_proximal_joint | 0.0 | 1.0472 |
| right_thumb_distal_joint | 0.0 | 1.0472 |
| right_thumb_tip | 0.0 | 0.0 |
| right_index_proximal_joint | 0.0 | 1.4661 |
| right_index_distal_joint | 0.0 | 1.693 |
| right_index_tip_joint | 0.0 | 0.0 |
| right_middle_proximal_joint | 0.0 | 1.4661 |
| right_middle_distal_joint | 0.0 | 1.693 |
| right_middle_tip_joint | 0.0 | 0.0 |
| right_ring_proximal_joint | 0.0 | 1.4661 |
| right_ring_distal_joint | 0.0 | 1.693 |
| right_ring_tip_joint | 0.0 | 0.0 |
| right_pinky_proximal_joint | 0.0 | 1.4661 |
| right_pinky_distal_joint | 0.0 | 1.693 |
| right_pinky_tip_joint | 0.0 | 0.0 |

## G1 + Dex3-1 hand (MuJoCo)

### Legs

Same as Unitree G1 (base).

### Arms

Same as Unitree G1 (base), with minor rounding differences in the MuJoCo file:
- left/right wrist_roll_joint: -1.97222 to 1.97222
- left/right wrist_pitch_joint: -1.61443 to 1.61443
- left/right wrist_yaw_joint: -1.61443 to 1.61443

### Hands

| Joint | Lower | Upper |
| --- | ---: | ---: |
| left_hand_thumb_0_joint | -1.0472 | 1.0472 |
| left_hand_thumb_1_joint | -0.724312 | 1.0472 |
| left_hand_thumb_2_joint | 0.0 | 1.74533 |
| left_hand_middle_0_joint | -1.5708 | 0.0 |
| left_hand_middle_1_joint | -1.74533 | 0.0 |
| left_hand_index_0_joint | -1.5708 | 0.0 |
| left_hand_index_1_joint | -1.74533 | 0.0 |
| right_hand_thumb_0_joint | -1.0472 | 1.0472 |
| right_hand_thumb_1_joint | -1.0472 | 0.724312 |
| right_hand_thumb_2_joint | -1.74533 | 0.0 |
| right_hand_middle_0_joint | 0.0 | 1.5708 |
| right_hand_middle_1_joint | 0.0 | 1.74533 |
| right_hand_index_0_joint | 0.0 | 1.5708 |
| right_hand_index_1_joint | 0.0 | 1.74533 |

## G1 + Dex3-1 hand (MuJoCo menagerie)

### Legs

| Joint | Lower | Upper |
| --- | ---: | ---: |
| left_hip_pitch_joint | -2.5307 | 2.8798 |
| left_hip_roll_joint | -0.5236 | 2.9671 |
| left_hip_yaw_joint | -2.7576 | 2.7576 |
| left_knee_joint | -0.087267 | 2.8798 |
| left_ankle_pitch_joint | -0.87267 | 0.5236 |
| left_ankle_roll_joint | -0.2618 | 0.2618 |
| right_hip_pitch_joint | -2.5307 | 2.8798 |
| right_hip_roll_joint | -2.9671 | 0.5236 |
| right_hip_yaw_joint | -2.7576 | 2.7576 |
| right_knee_joint | -0.087267 | 2.8798 |
| right_ankle_pitch_joint | -0.87267 | 0.5236 |
| right_ankle_roll_joint | -0.2618 | 0.2618 |

### Arms

| Joint | Lower | Upper |
| --- | ---: | ---: |
| waist_yaw_joint | -2.618 | 2.618 |
| waist_roll_joint | -0.52 | 0.52 |
| waist_pitch_joint | -0.52 | 0.52 |
| left_shoulder_pitch_joint | -3.0892 | 2.6704 |
| left_shoulder_roll_joint | -1.5882 | 2.2515 |
| left_shoulder_yaw_joint | -2.618 | 2.618 |
| left_elbow_joint | -1.0472 | 2.0944 |
| left_wrist_roll_joint | -1.97222 | 1.97222 |
| left_wrist_pitch_joint | -1.61443 | 1.61443 |
| left_wrist_yaw_joint | -1.61443 | 1.61443 |
| right_shoulder_pitch_joint | -3.0892 | 2.6704 |
| right_shoulder_roll_joint | -2.2515 | 1.5882 |
| right_shoulder_yaw_joint | -2.618 | 2.618 |
| right_elbow_joint | -1.0472 | 2.0944 |
| right_wrist_roll_joint | -1.97222 | 1.97222 |
| right_wrist_pitch_joint | -1.61443 | 1.61443 |
| right_wrist_yaw_joint | -1.61443 | 1.61443 |

### Hands

| Joint | Lower | Upper |
| --- | ---: | ---: |
| left_hand_thumb_0_joint | -1.0472 | 1.0472 |
| left_hand_thumb_1_joint | -0.724312 | 1.0472 |
| left_hand_thumb_2_joint | 0.0 | 1.74533 |
| left_hand_middle_0_joint | -1.5708 | 0.0 |
| left_hand_middle_1_joint | -1.74533 | 0.0 |
| left_hand_index_0_joint | -1.5708 | 0.0 |
| left_hand_index_1_joint | -1.74533 | 0.0 |
| right_hand_thumb_0_joint | -1.0472 | 1.0472 |
| right_hand_thumb_1_joint | -1.0472 | 0.724312 |
| right_hand_thumb_2_joint | -1.74533 | 0.0 |
| right_hand_middle_0_joint | 0.0 | 1.5708 |
| right_hand_middle_1_joint | 0.0 | 1.74533 |
| right_hand_index_0_joint | 0.0 | 1.5708 |
| right_hand_index_1_joint | 0.0 | 1.74533 |

## G1 + Inspire hand (DFQ URDF)

### Legs

Same as Unitree G1 (base).

### Arms

Same as Unitree G1 (base).

### Hands

| Joint | Lower | Upper |
| --- | ---: | ---: |
| L_thumb_proximal_yaw_joint | -0.1 | 1.3 |
| L_thumb_proximal_pitch_joint | -0.1 | 0.6 |
| L_thumb_intermediate_joint | 0.0 | 0.8 |
| L_thumb_distal_joint | 0.0 | 1.2 |
| L_index_proximal_joint | 0.0 | 1.7 |
| L_index_intermediate_joint | 0.0 | 1.7 |
| L_middle_proximal_joint | 0.0 | 1.7 |
| L_middle_intermediate_joint | 0.0 | 1.7 |
| L_ring_proximal_joint | 0.0 | 1.7 |
| L_ring_intermediate_joint | 0.0 | 1.7 |
| L_pinky_proximal_joint | 0.0 | 1.7 |
| L_pinky_intermediate_joint | 0.0 | 1.7 |
| R_thumb_proximal_yaw_joint | -0.1 | 1.3 |
| R_thumb_proximal_pitch_joint | -0.1 | 0.6 |
| R_thumb_intermediate_joint | 0.0 | 0.8 |
| R_thumb_distal_joint | 0.0 | 1.2 |
| R_index_proximal_joint | 0.0 | 1.7 |
| R_index_intermediate_joint | 0.0 | 1.7 |
| R_middle_proximal_joint | 0.0 | 1.7 |
| R_middle_intermediate_joint | 0.0 | 1.7 |
| R_ring_proximal_joint | 0.0 | 1.7 |
| R_ring_intermediate_joint | 0.0 | 1.7 |
| R_pinky_proximal_joint | 0.0 | 1.7 |
| R_pinky_intermediate_joint | 0.0 | 1.7 |

## G1 + Inspire hand (FTP URDF)

### Legs

Same as Unitree G1 (base).

### Arms

Same as Unitree G1 (base).

### Hands

No Inspire hand joints found in this file.
