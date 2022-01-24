'''
图像分割（将前景物体从背景物体中分离出来）
1，传统的图像分割方法（1，分水岭，2，GrabCut法，3，MeanShift法，4，背景抠出）
2，基于深度学习的图像分割方法
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 传统的图像分割方法之背景抠出法实现

if __name__ == "__main__":
    cap = cv2.VideoCapture("../video/vtest.avi")
    # 以混合高斯模型为基础的前景和背景分割算法
    # 创建BackgroundSubtractorMOG对象，用于去除静态物品，相关参数
    # history（表示以多少帧为基准来判断物体是否是动态物体，数值越大计算越精准）
    # nmixtures（高斯范围值，默认5）
    # backgroundRatio（背景比率，默认值0.7就是图像%70是背景）
    # noiseSigma（默认0，自动降噪）
    #mog = cv2.createBackgroundSubtractorMOG()

    # 创建BackgroundSubtractorMOG2对象，用于去除静态物品，这个算法是对MOG进行改进，但是这个算法会产生很多噪点
    # history（表示以多少帧为基准来判断物体是否是动态物体，数值越大计算越精准），detectShadows（是否检测阴影）
    mog = cv2.createBackgroundSubtractorMOG2()

    # 创建createBackgroundSubtractorKNN对象，用于去除静态物品，这个算法是对MOG进行改进，而且噪点更低
    # history（表示以多少帧为基准来判断物体是否是动态物体，数值越大计算越精准），detectShadows（是否检测阴影）
    knn = cv2.createBackgroundSubtractorKNN()
    while True:
        # 读取视频帧
        ret,frame = cap.read()
        if ret == True:
            # 背景分割
            mgmask = mog.apply(frame)
            knnmask = knn.apply(frame)
            cv2.imshow("mog",mgmask)
            cv2.imshow("knn", knnmask)

        key = cv2.waitKey(10)
        if key == 27 or ret == False:
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


