"""原始版单目测距，不考虑相机畸变,最好把相机正对物体"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # 视频对象
high = np.array([35, 255, 255])
low = np.array([10, 50, 100])  # 颜色识别阈值

def PlayVideo(cap, low, high):
    while cap.isOpened():
        ret, frame = cap.read()
        video = GetMask1(frame, low, high)  # 处理视频
        GetOutline(video, np.array([5, 50, 100]), np.array([50, 255, 255]))
        cv2.imshow("video", video)  # 可视化视频
        # out.write(video)  # 输出视频
        c = cv2.waitKey(5)
        if c == 27:
            break    # 每5毫秒等一次，直到按下Esc键后退出
    return 0


def GetMask1(img, low, high):
    img = cv2.GaussianBlur(img, (5, 5), 0, None)  # 高斯模糊
    h, s, v = cv2.split(img)
    v = cv2.equalizeHist(v)
    hsvimg = cv2.merge((h, s, v))  # 直方图均衡化
    mask = cv2.inRange(hsvimg, low, high)  # 生成掩膜

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)  # 对掩膜进行腐蚀和膨胀处理

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_l = list(contours)
    new_contours_l = [contour for contour in contours_l if cv2.contourArea(contour) > 500]  # 筛选面积大于500的轮廓

    centers = []
    apexes = []

    for cnt in new_contours_l:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        centers.append((cx, cy))  # 计算轮廓中心

        x1 = cx - w
        y1 = cy - h
        w1 = 2 * w
        h1 = 2 * h
        apexes.append([x1, y1, w1, h1])  # 计算较大边框

    area = np.zeros_like(img)
    for i in range(len(apexes)):
        cv2.rectangle(area, (apexes[i][0], apexes[i][1]), (apexes[i][0] + apexes[i][2], apexes[i][1] + apexes[i][3]),
                      (255, 255, 255), -1)  # 获得新的区域

    result = cv2.bitwise_and(img, area)
    return result


def GetOutline(img, low, high):
    mask = cv2.inRange(img, low, high)  # 生成掩膜
    print(mask.shape)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=5)  # 对掩膜进行腐蚀和膨胀处理

    for i in [img, mask]:
        print(i.shape)
    img = cv2.bitwise_and(img, mask)
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    edges_img = cv2.Canny(gray, 80, 180, apertureSize=3)
    Lines = cv2.HoughLinesP(edges_img, 1, np.pi/180, 30, 300, 5, 300)
    print(Lines)
    return None



PlayVideo(cap, low, high)