import mujoco
import mujoco.viewer
import time
import numpy as np
from pathlib import Path

# step 1. 파일 경로 및 모델 초기화
xml_path = str(Path(__file__).resolve().parent / "ur5e_with_gripper_rigid.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)



# step 2. 움직이려는 관절과 모터 이름 정의 (XML 파일에 맞게 써야 함)
# 여기서는 그리퍼를 제외한 팔(Arm)을 움직이는 딱 6개의 축만 제어할것이다. 그래서 6개만 IK 계산에 사용하는 계산입니다
joint_names = [
    "ur5e_shoulder_pan_joint", "ur5e_shoulder_lift_joint", "ur5e_elbow_joint",
    "ur5e_wrist_1_joint", "ur5e_wrist_2_joint", "ur5e_wrist_3_joint"
]
actuator_names = [
    "ur5e_shoulder_pan", "ur5e_shoulder_lift", "ur5e_elbow",
    "ur5e_wrist_1", "ur5e_wrist_2", "ur5e_wrist_3"
]



# step 3. 다자유도 관절(Ball Joint 등)까지 완벽 포함하는 범용 DOF 인덱스 추출
dof_ids = []
for name in joint_names:
    # 1. 데이터 시작 주소 (Address)
    start_addr = model.joint(name).dofadr[0] # model.jnt_dofadr는 각 관절이 시작하는 DOF 인덱스를 알려줌 (예: Revolute Joint = 1, Ball Joint = 3)
    
    # 2. 해당 관절이 가진 자유도 개수 파악
    # 관절의 속도(qvel) 배열 길이 == 자유도(DOF) 개수
    # ex. num_dof --> Ball Joint = 3, Revolute Joint = 1
    num_dof = len(data.joint(name).qvel)
    
    # 3. 시작 주소부터 자유도 개수만큼 인덱스를 모두 긁어모음
    for i in range(num_dof):
        dof_ids.append(start_addr + i)

print(dof_ids)

# step 4. 액츄에이터 ID 추출 - mujoco.mj_name2id(model, type, name)
actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names]
print(actuator_ids)

# step 5. [중요 안전장치] 추출된 자유도(DOF) 개수와 모터(Actuator) 개수가 일치하는지 확인!
# assert는 조건이 False일 때 프로그램을 멈춤
assert len(dof_ids) == len(actuator_ids), f"경고: 제어할 DOF 개수({len(dof_ids)})와 모터 개수({len(actuator_ids)})가 다릅니다!"
# -------------------------------------------------------------------------

site_id = model.site('ur5e_attachment_site').id

# 확인용 출력
print(f"DOF IDs: {dof_ids}") 
print(f"Actuator IDs: {actuator_ids}") 
print(f"Site ID: {site_id}") 
print("✅ 로봇 팔 범용 관절 매핑 완료! (다자유도 대응 완비)")
# -------------------------------------------------------------------------

# step 6. 목표 좌표 설정 (x, y, z)
target_pos = np.array([-0.5, -0.3, 0.4]) 

# IK 제어 파라미터
step_size = 0.08  # 한 번에 이동할 비율 (Learning Rate)
tol = 0.006       # 허용 오차 - 6mm 이내로 들어오면 성공으로 간주

# step 7. 시뮬레이션 실행
with mujoco.viewer.launch_passive(model, data) as viewer:
    print(f"🚀 IK 제어 시작! 목표 좌표: {target_pos}")
    
    while viewer.is_running():
        step_start = time.time()

        # 목표지점과 현재의 벡터 차이 측정 -> error = target_pos - 현재 site(보통 EE 근처에 찍음) 위치
        current_pos = data.site(site_id).xpos
        error = target_pos - current_pos  
        
        # 목표 지점에 아직 도달하지 않았다면 - np.linalg.norm(error) : 벡터의 크기(거리)
        if np.linalg.norm(error) > tol:
            
            # 1. 빈 Jacobian 행렬 준비 - 3(x,y,z) x nv(DOF개수)
            jacp = np.zeros((3, model.nv))  
            
            # 2. Jacobian 계산 - mujoco.mj_jacSite(model, data, jacp, None, site_id) -> jacp에 위치 야코비안 계산 결과가 채워짐
            # jacp (Position Jacobian) - 관절을 움직였을 때, 손끝의 위치(x,y,z)가 어떻게 변하는지 적혀있음
            # None : jacr은 Orientation Jacobian (회전) 계산이 필요 없어서 None으로 넘김(손목을 특정 각도로 회전은 여기서는 사용x)
            # site_id : 움직일 것
            # v_site ​= J_p​*q_dot
            # mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            mujoco.mj_jacSite(model, data, jacp, None, site_id)
            
            # 3. 팔(Arm) 축에 해당하는 열만 쏙 빼옴 (범용 dof_ids 사용!)
            # jacp = 3(x,y,z) * 전체관절DOF
            # : -> x,y,z 전체 가져옴
            # dof_ids -> 우리가 제어하려는 관절들의 DOF 인덱스 리스트
            J_arm = jacp[:, dof_ids]
            
            # 4. IK 계산
            # 역자코비안 공식: delta_q = J_inv * error
            # inv()는 정방행렬에만 적용 가능 -> 역야코비안에서는 일반적으로 유사역행렬(pseudo-inverse)을 사용
            # J_inv = np.linalg.inv(J_arm)  # 정방행렬이 아니므로 오류 발생 가능
            # J_inv에는 3차원 공간의 오차를 6개 관절의 움직임으로 번역해주는 6×3 행렬이 반환됨 (행 = 관절 DOF 개수, 열 = x,y,z)
            # pinv == numpy의 ik풀어주는 것들 중 하나
            J_inv = np.linalg.pinv(J_arm) 
            
            # 5. 각도 변화량 계산 및 클리핑(너무 큰 변화량은 한 번에 적용하지 않도록 제한)

            delta_q = J_inv @ error # @ : 행렬 곱 연산자 (Python 3.5+)
            delta_q = np.clip(delta_q, -0.05, 0.05) # 최대 속도 제한 # 클리핑:한 번에 너무 큰 움직임이 발생하지 않도록 -0.05 ~ 0.05 라디안으로 제한 (약 2.8도)
            
            # 6. 제어 명령 누적 (DOF와 Actuator의 매핑이 1:1이라고 가정)
            for i, act_id in enumerate(actuator_ids):
                data.ctrl[act_id] += step_size * delta_q[i]

        # 물리 엔진 1스텝 전진
        mujoco.mj_step(model, data)

        # 실시간 EE 좌표 출력
        print(f"\rEE Position (X, Y, Z): {data.site(site_id).xpos[0]:.4f}, {data.site(site_id).xpos[1]:.4f}, {data.site(site_id).xpos[2]:.4f}", end="")

        # 화면 및 시간 동기화
        viewer.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)