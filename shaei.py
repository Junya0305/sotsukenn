import cv2
import numpy as np

# 入力画像の読み込み（例: サッカーコートの1フレーム）
img = cv2.imread("frame_0100.jpg")  # 適切な画像ファイル名に変更

# クリックされた座標を格納するリスト
src_points = []

# マウスクリックで4点取得
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(src_points) < 4:
        src_points.append([x, y])
        print(f"選択された点: ({x}, {y})")
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 Points", img)

        if len(src_points) == 4:
            do_perspective_transform()



def do_perspective_transform():
    src = np.array(src_points, dtype=np.float32)

    # 仮想2Dフィールド（例: 横600×縦400のフィールド）
    dst = np.array([
        [0, 0],
        [600, 0],
        [600, 400],
        [0, 400]
    ], dtype=np.float32)

    # 射影変換行列を求める
    matrix = cv2.getPerspectiveTransform(src, dst)

    # 射影変換を適用
    warped = cv2.warpPerspective(img, matrix, (600, 400))
    cv2.imshow("Warped Field", warped)
    cv2.imwrite("warped_output.jpg", warped)
    print("射影変換が完了しました（warped_output.jpg に保存）")

# ウィンドウ表示
cv2.imshow("Select 4 Points", img)
cv2.setMouseCallback("Select 4 Points", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
