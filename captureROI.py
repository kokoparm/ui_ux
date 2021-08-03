import cv2
import os
from glob import glob

cap = cv2.VideoCapture(0)

count = 0
# 저장 폴더
base_dir = "./captures"
if not os.path.exists(base_dir):
    os.mkdir(base_dir)


def saveROI(base, target, frame):
    """인수로 받은 폴더에 프레임 저장

    Args:
        base (str): 루트 폴더
        target (str): 저장하고자하는 폴더
        frame (numpy.ndarray): 저장할 이미지
    """
    if not os.path.exists(os.path.join(base, target)):
        os.mkdir(os.path.join(base, target))

    # 폴더 내 png 파일의 수로 파일 이름 지정
    print(os.path.join(base, target, "*.png"))
    imglist = glob(os.path.join(base, target, "*.png"))
    count = len(imglist)

    w = cv2.imwrite(os.path.join(base, target, f"{count + 1}.png"), frame)  # 경로에 한글 포함 시 저장 x
    # status
    print(w)
    print(os.path.join(base, target, f"{count + 1}.png saved"))


while True:
    ret, frame = cap.read()

    # 관심 영역 추출 및 표시
    roi = frame[100:300, 200:400].copy()
    frame = cv2.rectangle(frame, (200, 100), (400, 300), (255, 0, 0), 5, cv2.LINE_8)

    # 관심 영역 크기 조절
    roi = cv2.resize(roi, dsize=(100, 100), interpolation=cv2.INTER_LINEAR)

    # 25fps로 화면에 출력
    cv2.imshow("frame", frame)
    # cv2.imshow("roi", roi)
    key = cv2.waitKey(40)

    # esc 입력 시 종료
    if key == 27:
        break

    elif (key >= 97 and key <= 122) or (key >= 48 and key <= 57):
        saveROI(base_dir, chr(key), roi)

cap.release()
cv2.destroyAllWindows()
