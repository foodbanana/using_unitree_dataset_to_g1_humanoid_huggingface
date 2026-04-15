import mujoco
import mujoco.viewer
import time
import numpy as np
import pinocchio as pin
from pathlib import Path

# ==============================================================================
# Step 1. 두 개의 세계(Physics & Math) 초기화
# ==============================================================================
current_dir = Path(__file__).resolve().parent

# 1-1. MuJoCo 초기화 (물리 엔진)
xml_path = str(current_dir / "ur5e_with_gripper_rigid.xml")
model_mj = mujoco.MjModel.from_xml_path(xml_path)
data_mj = mujoco.MjData(model_mj)

# 1-2. Pinocchio 초기화 (수학 엔진 - URDF 필요!)
# 1-2. MuJoCo 모델을 URDF로 변환하는 파이썬 파일을 외부에서 실행하고여기 특정 경로에 저장해야 한다 (중요!)
urdf_path = str(current_dir / "ur5e_generated.urdf")
model_pin = pin.buildModelFromUrdf(urdf_path)
data_pin = model_pin.createData()

# ==============================================================================
# Step 2. 관절 매핑 및 제어 프레임 설정
# ==============================================================================
# 제어할 액츄에이터 (MuJoCo용)
actuator_names = [
    "ur5e_shoulder_pan", "ur5e_shoulder_lift", "ur5e_elbow",
    "ur5e_wrist_1", "ur5e_wrist_2", "ur5e_wrist_3"
]
actuator_ids = [mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names]

# Pinocchio에서 추적할 End-Effector 프레임 이름 (URDF에 정의된 이름)
# 보통 'wrist_3_link', 'tool0', 혹은 그리퍼 attachment 링크 이름입니다.
ee_frame_name = "rq_gripper_tcp_link"
ee_frame_id = model_pin.getFrameId(ee_frame_name)
site_id = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_SITE, 'rq_gripper_tcp_site')

print(f"✅ Pinocchio 모델 로드 완료! 제어 프레임 ID: {ee_frame_id}")

# ==============================================================================
# Step 3. 목표 Pose(위치+회전) 및 파라미터 설정
# ==============================================================================
# Pinocchio는 SE(3) 객체를 사용하여 위치(Translation)와 회전(Rotation)을 한 번에 다룹니다.
target_pos = np.array([0.4, 0.0, 0.5])

# 회전은 현재 기본값(단위 행렬)을 유지한다고 가정
target_rot = np.eye(3) 
oMdes = pin.SE3(target_rot, target_pos) # oMdes: Origin to Desired frame

dt = 0.002        # 시뮬레이션 타임스텝
damp = 1e-4       # DLS Damping factor (Pinocchio에서도 필수!)
step_size = 0.5   # Pinocchio IK의 수렴 속도 (0 ~ 1.0)
tol = 1e-3        # 허용 오차 (1mm)

# ==============================================================================
# Step 4. 시뮬레이션 및 CLIK 루프
# ==============================================================================
with mujoco.viewer.launch_passive(model_mj, data_mj) as viewer:
    # print(f"🚀 Pinocchio 기반 IK 제어 시작!")
    # print(f"📊 MuJoCo 제어 가능 관절 수 (nq): {model_mj.nq}")
    # print(f"📊 Pinocchio 필요 관절 수 (nq): {model_pin.nq}")
    # print(f"🔗 Pinocchio 관절 이름들: {[model_pin.names[i] for i in range(model_pin.njoints)]}")
    # print(f"Frame 76 name: {model_pin.frames[76].name}")
    # print("\n--- 📝 Pinocchio 모든 프레임 리스트 ---")
    # for i, f in enumerate(model_pin.frames):
    #    print(f"Index {i}: {f.name} (Type: {f.type})")
    # print("-" * 40)

    # 2. 우리가 찾는 이름이 있는지 확인하는 안전한 코드
    if model_pin.existFrame(ee_frame_name):
        ee_frame_id = model_pin.getFrameId(ee_frame_name)
        print(f"✅ 찾았다! {ee_frame_name}의 ID는 {ee_frame_id}입니다.")
    else:
        print(f"❌ 못 찾았다: '{ee_frame_name}'이라는 프레임은 존재하지 않습니다.")
        # 대안으로 가장 마지막 프레임이라도 일단 잡아봅니다.
        ee_frame_id = model_pin.nframes - 1 
        print(f"⚠️ 임시로 마지막 프레임({model_pin.frames[ee_frame_id].name})을 사용합니다.")


    while viewer.is_running():
        step_start = time.time()

        # ---------------------------------------------------------
        # [A] 상태 동기화: MuJoCo의 현실을 Pinocchio의 뇌로 복사
        # ---------------------------------------------------------
        # (주의: 이 예제는 MuJoCo와 Pinocchio의 관절 순서가 같다고 가정합니다)
        q_curr = data_mj.qpos # MuJoCo의 전체 qpos(14개)를 그대로 가져옵니다.
        
        # ---------------------------------------------------------
        # [B] Pinocchio의 정기구학(FK) 및 야코비안 업데이트
        # ---------------------------------------------------------
        pin.forwardKinematics(model_pin, data_pin, q_curr)
        pin.updateFramePlacements(model_pin, data_pin)
        
        # ---------------------------------------------------------
        # [C] 6D 공간(Spatial) 오차 계산 (이것이 Pinocchio의 꽃입니다!)
        # ---------------------------------------------------------
        # 현재 EE의 Pose (SE3 객체)
        current_pose = data_pin.oMf[ee_frame_id] 
        
        # 원하는 Pose와 현재 Pose 사이의 변환 행렬 차이 계산
        dMi = oMdes.actInv(current_pose) 
        
        # Lie 대수(log6)를 이용해 6차원 에러 벡터 [선형오차 3, 각도오차 3] 추출
        err = pin.log6(dMi).vector 
        
        # 위치 오차의 크기만 추출 (종료 조건용)
        err_norm = np.linalg.norm(err[:3]) 
        
        

        
        
        if err_norm > tol:
            # ---------------------------------------------------------
            # [D] Pinocchio Jacobian 계산 및 DLS 역행렬 풀이
            # ---------------------------------------------------------
            # 1. 6x14 전체 야코비안 추출
            J_full = pin.computeFrameJacobian(model_pin, data_pin, q_curr, ee_frame_id, pin.ReferenceFrame.LOCAL)
            
            # 2. 핵심: 팔 6축에 해당하는 열만 잘라내기 (6x6 행렬 생성)
            J = J_full[:, :6]
            
            # 3. 6x6 행렬에 대해 DLS 역행렬 계산
            # DLS (J^T * (J * J^T + damp * I)^-1) 계산
            J_T = J.T
            J_inv = J_T @ np.linalg.inv(J @ J_T + damp * np.eye(6))
            
            # 4. 6차원 속도 벡터 계산
            # 각도 변화량 (속도) 계산 -> v = - J_inv * err
            # (오차 방향이 oMdes 기준으로 역방향이므로 -를 붙임)
            v = -J_inv @ err 
            
            # ---------------------------------------------------------
            # [E] MuJoCo로 명령 전송
            # ---------------------------------------------------------
            # 계산된 변화량 v를 step_size(비례 게인)를 곱해 제어기에 누적
            # 5. 이제 v[0]~v[5]는 "팔 6개만으로 목표를 달성하기 위한 최적값"이 됩니다.
            for i, act_id in enumerate(actuator_ids):
                data_mj.ctrl[act_id] += v[i] * step_size * dt
        
        
        # [EE pos 출력부]
        # 1. 사이트 ID를 이용해 실제 세계(MuJoCo)의 좌표 가져오기
        # 만약 사이트 이름을 못 찾으면 site_id가 -1이 되므로 안전장치를 둡니다.
        #if site_id != -1:
        #    ee_pos = data_mj.site_xpos[site_id]
        #    # 위치와 함께 오차(err_norm)도 같이 찍어주면 훨씬 보기 좋습니다!
        #    print(f"\r[POS] X:{ee_pos[0]:.3f}, Y:{ee_pos[1]:.3f}, Z:{ee_pos[2]:.3f} | Error: {err_norm:.4f}", end="", flush=True)
        
        
        
        
        
        # 물리 엔진 1스텝 진행
        mujoco.mj_step(model_mj, data_mj)

        # 상태 출력 (Pinocchio가 계산한 EE 위치)
        print(f"\rError Norm: {err_norm:.4f} ,  EE Pos: {current_pose.translation}", end="")
        # 화면 및 시간 동기화
        viewer.sync()
        time_until_next_step = model_mj.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
