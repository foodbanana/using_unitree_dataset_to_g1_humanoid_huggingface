from mjcf_urdf_simple_converter import convert
from pathlib import Path

# 1. 경로 설정
current_dir = Path(__file__).resolve().parent
xml_path = str(current_dir / "ur5e_with_gripper_rigid.xml")
urdf_save_path = str(current_dir / "ur5e_generated.urdf")

# 2. 변환 실행
# 이 라이브러리가 MuJoCo 내부 엔진을 이용해 기구학 구조를 파싱한 뒤 URDF로 구워줍니다.
try:
    print(f"⏳ 변환 시작: {xml_path}...")
    convert(xml_path, urdf_save_path)
    print("-" * 50)
    print(f"✅ 변환 성공! 파일이 생성되었습니다: {urdf_save_path}")
    print("-" * 50)
except Exception as e:
    print(f"❌ 변환 실패: {e}")