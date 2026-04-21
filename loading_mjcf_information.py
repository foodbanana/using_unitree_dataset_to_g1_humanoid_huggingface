import mujoco

# 생성하신 파일 경로
xml_path = "/home/taeung/g1_datasets_huggingface/assets/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.xml"

try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    print(f"모델 로드 성공!")
    print(f"전체 자유도(nv): {model.nv}")
    print(f"관절(Joint) 개수: {model.njnt}")
    print(f"구동기(Actuator) 개수: {model.nu}")
    
    # 각 관절 이름과 ID 출력 (29개가 맞는지 확인)
    print("\n[관절 리스트]")
    for i in range(model.njnt):
        print(f"ID {i}: {model.joint(i).name}")
        
except Exception as e:
    print(f"오류 발생: {e}")