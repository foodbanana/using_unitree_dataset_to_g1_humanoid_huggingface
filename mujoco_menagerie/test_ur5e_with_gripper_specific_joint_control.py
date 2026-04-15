import mujoco
import mujoco.viewer
import time
import numpy as np
from pathlib import Path

# 1. 파일 경로 및 모델 초기화
xml_path = str(Path(__file__).resolve().parent / "ur5e_with_gripper_rigid.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 2. 제어 대상 ID 찾기 (어깨 모터 & 그리퍼 모터 & 손목2 모터)
# 주의: XML 파일에 정의된 실제 모터 이름을 확인하고 필요시 수정하세요.
shoulder_name = "ur5e_shoulder_pan" 
gripper_name = "ur5e_rq_fingers_actuator" 
wrist2_name = "ur5e_wrist_2"

shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, shoulder_name)
gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_name)
wrist2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, wrist2_name)

# ID 검색 실패 시 안전 종료
if shoulder_id == -1 or gripper_id == -1 or wrist2_id == -1:
    print(f"❌ 모터 이름을 찾을 수 없습니다. XML을 확인하세요.")
    print(f"Shoulder ID: {shoulder_id}, Gripper ID: {gripper_id}, Wrist2 ID: {wrist2_id}")
    exit()

print(f"✅ 제어 준비 완료! (어깨 ID: {shoulder_id}, 그리퍼 ID: {gripper_id}, 손목2 ID: {wrist2_id})")

# 3. 시뮬레이션 뷰어 실행
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("🚀 3개 모터 제어 시작: 어깨/손목2는 사인파, 그리퍼는 잼잼!")
    
    while viewer.is_running():
        step_start = time.time() 

        # -------------------------------------------------------
        # [핵심 로직 1] 부드러운 어깨 관절 제어 (Sine Wave)
        # -------------------------------------------------------
        amplitude = 1.0     # 진폭: 최대 움직이는 각도 (라디안, 1.0 rad는 약 57.3도)
        frequency = 0.5     # 주파수: 1초에 몇 바퀴 돌 것인가? (0.5Hz = 2초에 1왕복)
        
        # 공식: A * sin(2 * pi * f * t)
        target_angle = amplitude * np.sin(2 * np.pi * frequency * data.time)
        
        # 어깨 모터에 목표 각도 명령 전달
        data.ctrl[shoulder_id] = target_angle

        # -------------------------------------------------------
        # [핵심 로직 2] 시간 기반 그리퍼 토글 제어 (1단계 복습)
        # -------------------------------------------------------
        if (data.time % 4) < 2.0:
            data.ctrl[gripper_id] = 255.0  # 꽉 쥐기
        else:
            data.ctrl[gripper_id] = 0.0    # 펴기

        # -------------------------------------------------------
        # [핵심 로직 3] 손목2 관절 제어 (Sine Wave)
        # -------------------------------------------------------
        wrist2_amplitude = 0.5
        wrist2_frequency = 1.5
        wrist2_target_angle = wrist2_amplitude * np.sin(2 * np.pi * wrist2_frequency * data.time)

        # 손목2 모터에 목표 각도 명령 전달
        data.ctrl[wrist2_id] = wrist2_target_angle

        # 4. 물리 엔진 연산 및 화면 업데이트
        mujoco.mj_step(model, data)

        # 위치 방향 읽기

        
        ee_pos = data.site('ur5e_attachment_site').xpos       # 위치 (Position)
        ee_orient = data.site('ur5e_attachment_site').xmat    # 방향 (Orientation - 3x3 Matrix)



        # \r로 줄을 초기화하고, end=""로 줄바꿈을 막아 실시간 갱신 효과를 줍니다.
        print(f"\r[POS] X:{ee_pos[0]:.3f}, Y:{ee_pos[1]:.3f}, Z:{ee_pos[2]:.3f}", end="", flush=True)


        viewer.sync()

        # 5. 실시간 동기화 (Real-time Sync)
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)




