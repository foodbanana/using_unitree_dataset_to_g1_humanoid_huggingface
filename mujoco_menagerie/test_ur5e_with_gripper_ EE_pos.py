import mujoco
import mujoco.viewer
import time
from pathlib import Path

# 1. 스크립트 기준 상대경로로 XML 위치를 계산해 이식성을 높임
xml_path = str(Path(__file__).resolve().parent / "ur5e_with_gripper_rigid.xml")

try:
    # 2. 모델 로드
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    print("MuJoCo 환경 로드 성공!")
except Exception as e:
    print(f"모델 로드 중 에러 발생: {e}")
    exit()

# 3. 시뮬레이션 및 뷰어 실행
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("\n[알림] 시뮬레이션이 시작되었습니다.")
    print("터미널 하단에 실시간 EE(End-Effector) 좌표가 표시됩니다.")
    
    while viewer.is_running():
        step_start = time.time()

        # 물리 연산 수행
        mujoco.mj_step(model, data)

        # -------------------------------------------------------
        # [실시간 EE 좌표 추출 파트]
        # ur5e_attachment_site의 xpos(전역 좌표)를 가져옵니다.
        # -------------------------------------------------------
        ee_pos = data.site('ur5e_attachment_site').xpos
        
        # \r을 사용해 한 줄에서 계속 업데이트 (깔끔하게 보기 위함)
        print(f"\rEE Position (X, Y, Z): {ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}", end="")

        viewer.sync()

        # 시뮬레이션 속도 유지
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)