import mujoco
import mujoco.viewer
import time
import numpy as np
from pathlib import Path

xml_path = str(Path(__file__).resolve().parent / "ur5e_with_gripper_rigid.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 그리퍼 액추에이터 ID 찾기 (이름은 XML 파일의 <actuator> 섹션 확인 필요)
# 보통 'fingers_actuator' 또는 'gripper' 등으로 되어 있습니다.
gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ur5e_rq_fingers_actuator")
if gripper_id != -1:
    print(f"성공! 그리퍼의 ID는 {gripper_id}번입니다.")
else:
    print("여전히 이름을 찾을 수 없습니다. 오타를 확인해 주세요.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("그리퍼 제어 테스트를 시작합니다. (2초 간격)")
    
    while viewer.is_running():
        step_start = time.time()

        # [그리퍼 제어 로직]
        # 시뮬레이션 시간을 4초로 나눈 나머지가 2보다 작으면 닫고, 크면 엽니다.
        if (data.time % 4) < 2.0:
            # 꽉 쥐기 (보통 최대값은 255 또는 1.0, XML 설정을 따름)
            data.ctrl[gripper_id] = 255.0 
        else:
            # 완전히 펴기
            data.ctrl[gripper_id] = 0.0

        mujoco.mj_step(model, data)
        viewer.sync()

        # 시간 동기화
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


