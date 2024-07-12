
import numpy as np
import cv2

cap = cv2.VideoCapture(0)  # 视频对象
fourcc = cv2.VideoWriter.fourcc(*"XVID")  # 编码器
out = cv2.VideoWriter("C:\\Users\\yb028028\\Desktop\\my_video.avi", fourcc, 30.00, (640, 480))  # 视频输出对象
high = np.array([35, 255, 255])
low = np.array([10, 50, 100])  # 颜色识别阈值


def PlayVideo(cap, low, high):
    while cap.isOpened():
        ret, frame = cap.read()
        video = GetMask(frame, low, high)  # 处理视频
        cv2.imshow("frame", video)  # 可视化视频
        out.write(frame)  # 输出视频
        c = cv2.waitKey(5)
        if c == 27:
            break    # 每5毫秒等一次，直到按下Esc键后退出
    return 0


def VideoInformation(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高和宽
    print("fps:%f"%fps, "width:%d"%width, "height:%d"%height)
    return 0


def GetMask(img, low, high):
    """new_width = 300
    new_height = 200
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)"""
    img = cv2.GaussianBlur(img, (5, 5), 0, None)  # 高斯模糊
    h, s, v = cv2.split(img)
    v = cv2.equalizeHist(v)
    hsvimg = cv2.merge((h, s, v))  # 直方图均衡化
    mask = cv2.inRange(hsvimg, low, high)  # 生成掩膜

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)  # 对掩膜进行腐蚀和膨胀处理

    img_values = img[mask > 0]
    h_mean = np.mean(img_values[:, 0])
    s_mean = np.mean(img_values[:, 1])
    v_mean = np.mean(img_values[:, 2])
    print(h_mean, s_mean, v_mean)  # 计算目标范围的平均HSV

    img, centers = DrawContour(img, mask)
    print(centers)  # 目标轮廓可视化，并计算轮廓中心

    """centers, apexes = GetRectangle(img, mask)
    for i in range(len(apexes)):
        cv2.rectangle(img, (apexes[i][0], apexes[i][1]), (apexes[i][0] + apexes[i][2], apexes[i][1] + apexes[i][3]),
                    (0, 255, 0), 2)  # 绘制矩形方框框住目标"""

    return img


def GetRectangle(img, mask):
    # 寻找轮廓
    new_contours_l = GetContour(mask)  # 筛选面积大于500的轮廓

    centers = []
    apexes = []
    MeanHSV = []

    for cnt in new_contours_l:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        centers.append((cx, cy))  # 计算轮廓中心

        x1 = cx - w
        y1 = cy - h
        w1 = 2*w
        h1 = 2*h
        apexes.append([x1, y1, w1, h1])  # 计算较大边框

        area = img[y: y+h, x: x+w]
        h_mean, s_mean, v_mean = cv2.mean(area)[:3]
        MeanHSV.append([h_mean, s_mean, v_mean])  # 计算目标范围的平均HSV
    print("centers:", centers, "apexes:", apexes, "Mean_HSV:", MeanHSV)
    return centers, apexes


def GetContour(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_l = list(contours)
    new_contours_l = [contour for contour in contours_l if cv2.contourArea(contour) > 500]  # 筛选面积大于500的轮廓
    return new_contours_l


def DrawContour(img, mask):
    new_contours_l = GetContour(mask)
    centers = []
    for i in new_contours_l:
        cv2.drawContours(img, i, -1, (255, 255, 255), 2)  # 边框可视化
        x, y, w, h = cv2.boundingRect(i)
        cx = x + w // 2
        cy = y + h // 2
        centers.append((cx, cy))  # 计算轮廓中心
    for center in centers:
        cv2.circle(img, center, 3, (0, 0, 255), -1)  # 绘制轮廓中心
    return img, centers





PlayVideo(cap, low, high)
VideoInformation(cap)
cap.release()
out.release()
cv2.destroyAllWindows()