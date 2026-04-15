# /home/taeung/g1_datasets_huggingface/mujoco_menagerie/test_ur5e_with_gripper_v1.py
import mujoco
import mujoco.viewer
import time

# 1. 실행할 XML 파일 경로 지정 (태웅님이 만드신 파일 경로)
xml_path = "/home/taeung/g1_datasets_huggingface/mujoco_menagerie/ur5e_with_gripper_rigid.xml"

# 2. 모델(Model)과 데이터(Data) 불러오기
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("MuJoCo 시뮬레이션 환경이 성공적으로 로드되었습니다!")

# 3. 뷰어(화면) 띄우기 (launch_passive 사용)
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # 뷰어 창이 닫히기 전까지 무한 루프
    while viewer.is_running():
        
        # ---------------------------------------------------
        # [제어 파트] 나중에 여기에 로봇을 움직이는 코드가 들어갑니다.
        # 예: data.ctrl[0] = 1.0 (1번 모터에 힘 주기)
        # ---------------------------------------------------

        # 4. 물리 엔진 1스텝 진행 (가장 중요한 함수!)
        mujoco.mj_step(model, data)

        # 5. 변경된 데이터 상태를 화면에 동기화(업데이트)
        viewer.sync()

        # 시뮬레이터 속도 조절 (너무 빨리 도는 것을 방지)
        time.sleep(model.opt.timestep)