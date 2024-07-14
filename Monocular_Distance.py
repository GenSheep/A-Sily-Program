"""单目测距
利用相机内参数矩阵去畸变，然后单目测距。
认为看见的方块轮廓面积为一个面面积的两倍"""

import numpy as np
import cv2
import math

cap = cv2.VideoCapture(0)  # 视频对象
fourcc = cv2.VideoWriter.fourcc(*"XVID")  # 编码器
out = cv2.VideoWriter("C:\\Users\\yb028028\\Desktop\\my_video.avi", fourcc, 30.00, (640, 480))  # 视频输出对象
high = np.array([32, 255, 255])
low = np.array([3, 72, 80])  # 颜色识别阈值
L = 1.55
S = L*L*2  # 1.5个面的面积，单位平方厘米


class CamSet(object):
    def __init__(self):
        self.CameraMatrix = np.array([[543.2352805344083, 0.0, 335.6325913504112],
                                      [0, 540.0090338378359, 237.52912533756933],
                                      [0, 0, 1]])
        self.DistCoeffs = np.array([-0.041107079237683884, 0.09952660331229725,
                                    -0.002757530806781375, 0.0024557170640417568,
                                    0.1088243671433803])

    def CorrectImage(self, img):
        h, w = img.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(self.CameraMatrix, self.DistCoeffs,(w, h),
                                                             1, (w, h), 0)
        # 计算无畸变和修正转换关系
        mapx, mapy = cv2.initUndistortRectifyMap(self.CameraMatrix, self.DistCoeffs, None, newCameraMatrix, (w, h),
                                                 cv2.CV_16SC2)
        # 重映射 输入是矫正后的图像
        CorrectImg = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        return CorrectImg


def GetMask(img, low, high):
    """new_width = 300
    new_height = 200
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.GaussianBlur(img, (5, 5), 0, None)  # 高斯模糊
    h, s, v = cv2.split(img)
    v = cv2.equalizeHist(v)
    hsvimg = cv2.merge((h, s, v))  # 直方图均衡化
    mask = cv2.inRange(hsvimg, low, high)  # 生成掩膜

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)  # 对掩膜进行腐蚀和膨胀处理

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = list()
    area = [cv2.contourArea(contours[i]) for i in range(len(contours))]
    if len(area) > 0:
        contour.append(contours[area.index(max(area))])
        img = cv2.drawContours(img, contour, -1, (0, 0, 255), 2)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = CamSet().CorrectImage(img)
        return img, max(area)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = CamSet().CorrectImage(img)
        return img, None



def PlayVideo(cap, low, high):
    while cap.isOpened():
        ret, frame = cap.read()
        video, area = GetMask(frame, low, high)  # 处理视频
        distance = OneEyeDistance(543.2352805344083, 540.0090338378359, S, area)
        print("area:", area, "distance:", distance)
        cv2.imshow("video", video)  # 可视化视频
        out.write(video)  # 输出视频
        c = cv2.waitKey(5)
        if c == 27:
            break    # 每5毫秒等一次，直到按下Esc键后退出
    return 0


def OneEyeDistance(fx, fy, S1, S2):
    if S2 is not None:
        d1 = math.sqrt(fx * fx * S1 / S2)
        d2 = math.sqrt(fy * fy * S1 / S2)
        return (d1 + d2) / 2
    else:
        return None

PlayVideo(cap, low, high)
