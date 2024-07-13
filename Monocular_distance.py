"""单目测距算法
包括：相机内参数矩阵的标定、单目测距算法"""

import numpy as np
import cv2
import csv

l = 1.0
w = 9
h = 6

def My_camera(w, h, l, fnames):
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objp = objp * l
    objectPoints = []  # 世界坐标系中的角点
    imgPoints = []  # 相机坐标系中的角点

    patternSize = (w, h)  # 棋盘格的尺寸
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE  # 使用自适应阈值来查找棋盘格; 在查找角点之前对图像进行归一化
    for fname in fnames:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
        height, width = gray.shape[:2]
        size = (width, height)  # 获取图片大小
        ret, corners = cv2.findChessboardCorners(gray, patternSize, flags=flags)  # 提取图像中的棋盘角点

        if ret:
            objectPoints.append(objp)
            imgPoints.append(corners)  # 如果提取成功，记录世界坐标系和相机坐标系的角点位置
            cv2.drawChessboardCorners(img, (w, h), corners, ret)
            cv2.imshow("corners", img)
            cv2.waitKey(5000)  # 可视化角点
    cv2.destroyAllWindows()
    # 完成两个坐标系下角点位置的确定

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 选择迭代优化算法
    camera_matrix = np.zeros((3, 3))
    dist_coeffs = np.zeros(5)  #初始化内参矩阵、畸变参数
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints,
                                                                        imgPoints,
                                                                        size,
                                                                        camera_matrix,
                                                                        dist_coeffs,
                                                                        criteria=criteria)  # 标定内参
    if ret:
        print("Camera Matrix:\n", camera_matrix, "\nDist_Coeffs:\n", dist_coeffs)
        return camera_matrix, dist_coeffs
    else:
        print("failed to get camera_matrix")
        return np.zeros((3, 3)), np.zeros(5)


def GetImgs(path):
    cap = cv2.VideoCapture(0)
    index = 1
    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ret:
            cv2.imshow("ChessBoard", gray)
            c = cv2.waitKey(5)
            if c == 27:
                break
            elif c == 32:
                cv2.imwrite(path+"\\ChessBoard%d.jpg"%index, gray)
                index += 1
                print("Get a Picture!")
    return index


def WriteData(camera_matrix, dist_coeffs, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in camera_matrix:
            writer.writerow(row)
        writer.writerow(dist_coeffs)
        print("Have Writen Camera Data")
    return None


numChessBoards = GetImgs("E:\\WareHouse\\RoboGame\\Monocular_distance")
fnames = ["E:\\WareHouse\\RoboGame\\Monocular_distance\\ChessBoard%d.jpg"%index for index in range(1, numChessBoards, 1)]
camera_matrix, dist_coeffs = My_camera(9, 6, 1.0,fnames)
WriteData(camera_matrix, dist_coeffs, "E:\\WareHouse\\RoboGame\\Monocular_distance\\CameraData.csv")
