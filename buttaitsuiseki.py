from ultralytics import YOLO
import cv2
import numpy as np

# YOLOモデルロード
model = YOLO("yolov8x.pt")

# 動画を開く
cap = cv2.VideoCapture("barca.mp4")

# 動画の幅・高さ・FPS取得
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# 出力動画設定
output_path = "output_barca_team_detected.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# HSV色空間の黄色と白の範囲
lower_yellow = np.array([28, 180, 180])
upper_yellow = np.array([36, 255, 255])

lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 60, 255])

def is_target_area(hsv_img, x1, y1, x2, y2):
    if x2 <= x1 or y2 <= y1:
        return False
    cropped_img = hsv_img[y1:y2, x1:x2]
    if cropped_img.size == 0:
        return False
    
    # 黄色領域のマスク
    mask_yellow = cv2.inRange(cropped_img, lower_yellow, upper_yellow)
    yellow_ratio = cv2.countNonZero(mask_yellow) / (cropped_img.shape[0] * cropped_img.shape[1])

    # 白色領域のマスク
    mask_white = cv2.inRange(cropped_img, lower_white, upper_white)
    white_ratio = cv2.countNonZero(mask_white) / (cropped_img.shape[0] * cropped_img.shape[1])

    # 判定：黄色か白の割合が一定以上ならTrue
    return yellow_ratio > 0.05 or white_ratio > 0.05

# フレームごとに処理
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # YOLOで人物検出
    results = model.predict(frame, classes=[0], conf=0.2)

    # HSV変換
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        if is_target_area(hsv_image, x1, y1, x2, y2):
            color = (0, 0, 255)  # ドルトムント（黄＋白）→赤枠
        else:
            color = (0, 255, 0)  # バルサ→緑枠
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    out.write(frame)
    cv2.imshow("チーム別枠線表示", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 後処理
cap.release()
out.release()
cv2.destroyAllWindows()
