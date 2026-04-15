import mujoco
from pathlib import Path

# 1. XML 경로 설정 (현재 파일과 같은 위치의 XML 로드)
xml_path = str(Path(__file__).resolve().parent / "ur5e_with_gripper_rigid.xml")

# 2. 모델 로드 (설계도면인 MjModel만 필요)
model = mujoco.MjModel.from_xml_path(xml_path)

print("  MuJoCo Actuator List")

# 3. 모든 액추에이터(모터) 이름과 ID 출력
# model.nu: XML에 정의된 총 액추에이터 개수
# model.nu: XML에 정의된 <actuator> 태그 총 개수 ex) 7개

# name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
# mujoco.mj_id2name(model, object_type, name)
# mj_id2name은 이름을 보고 id를 알려줌
# data.ctrl[6] --> ok , data_ctrl["gripper_motor"] --> X 이기에 필요
 
# mjt0bj == mujoco type of object 
# mjOBJ_ACTUATOR == <actuator>
# mjOBJ_JOINT == <joint>
# mjOBJ_BODY == <body name="..."> # 링크, 몸체
# mjOBJ_SITE == <site> # EE끝점 등등
# mjOBJ_SENSOR == <sensor>

for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"ID: {i} | Name: {name}")


# 4. 특정 이름으로 ID 찾기 테스트
target_name = "ur5e_rq_fingers_actuator" 
idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, target_name)

if idx != -1:
    print(f"결과: '{target_name}'의 ID는 {idx}번입니다.")
else:
    print(f"결과: '{target_name}'을(를) 찾을 수 없습니다. (오타 확인 ㄱㄱ)")
