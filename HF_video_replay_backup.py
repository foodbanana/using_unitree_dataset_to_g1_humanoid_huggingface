import cv2
import imageio
import numpy as np
import sys
from huggingface_hub import hf_hub_download, list_repo_files

WINDOW_NAME = 'Unitree G1 Multi-View (Top: High / Bottom: Wrist)'

# 1. 설정
repo_id = "unitreerobotics/G1_Dex3_GraspSquare_Dataset"
file_index = "file-000"  # 분석하고 싶은 파일 번호
camera_names = [
    "cam_left_high", "cam_right_high",
    "cam_left_wrist", "cam_right_wrist"
]

print(f"1. '{file_index}'에 해당하는 모든 카메라 영상을 찾는 중...")
try:
    all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
except Exception as e:
    print(f"❌ 파일 목록을 가져오지 못했습니다: {e}")
    sys.exit()

# 2. 존재하는 카메라 영상 경로 매핑 및 다운로드
video_paths = {}
for cam in camera_names:
    match = [f for f in all_files if cam in f and f"{file_index}.mp4" in f]
    if match:
        print(f"   [V] 찾음: {cam}")
        local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=match[0])
        video_paths[cam] = local_path
    else:
        print(f"   [X] 없음: {cam}")

# 3. 비디오 리더 준비
readers = {}
for cam, path in video_paths.items():
    try:
        readers[cam] = imageio.get_reader(path)
    except Exception as e:
        print(f"❌ {cam} 리더 생성 오류: {e}")

print("\n🚀 멀티뷰 재생 시작! (종료하려면 'q'를 누르세요)")

# 일부 OpenCV 백엔드에서 X 버튼 이벤트 처리를 안정화하기 위해 창을 명시적으로 생성
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

try:
    while True:
        frames = []
        for cam in camera_names:
            if cam in readers:
                try:
                    # 💡 수정된 부분: get_next_data()를 사용하여 프레임을 가져옵니다.
                    frame = readers[cam].get_next_data()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except (EOFError, IndexError, StopIteration):
                    # 영상이 끝났을 경우
                    print(f"\n🎥 {cam} 영상이 종료되었습니다.")
                    # 한쪽이라도 끝나면 전체 종료를 위해 루프 탈출
                    raise StopIteration
            else:
                # 없는 카메라는 검은색 화면 처리
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"{cam} (No Data)", (150, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frames.append(frame)

        # 4. 2x2 격자 구성
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        combined_frame = np.vstack((top_row, bottom_row))

        # 5. 화면 표시 (원본이 너무 크면 0.8배로 조절)
        display_frame = cv2.resize(combined_frame, (0, 0), fx=0.8, fy=0.8)
        cv2.imshow(WINDOW_NAME, display_frame)

        # 'q' 또는 'ESC' 키를 누르거나, 창의 닫기(X) 버튼을 감지했을 때 종료
        key = cv2.waitKey(30) & 0xFF
        try:
            visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            # X 버튼으로 닫히면 백엔드에 따라 예외가 발생할 수 있음
            visible = -1

        # 27은 ESC 키의 아스키 코드입니다.
        if key == ord('q') or key == 27 or visible < 1:
            print("\n🛑 재생을 중단합니다.")
            break

except StopIteration:
    print("🏁 모든 영상 재생이 끝났습니다.")
except Exception as e:
    print(f"\n❌ 실행 중 오류 발생: {e}")

# 리소스 정리
for r in readers.values():
    r.close()
cv2.destroyAllWindows()