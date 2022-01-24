'''
图像分割（将前景物体从背景物体中分离出来）
1，传统的图像分割方法（1，分水岭，2，GrabCut法，3，MeanShift法，4，背景抠出）
2，基于深度学习的图像分割方法
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 传统的图像分割方法之MeanShift法实现
# 1，严格来说该方法并并不是用来对图像分割的，而是在色彩层面的平滑滤波处理
# 2，它会中和色彩分布相近的颜色，平滑色彩细节，侵蚀掉面积较小的颜色区域
# 3，它以图像上一任点P为圆心，半径为sp，色彩副值（就是两个颜色的值相差多少表示相近颜色）为sr进行不断的迭代

if __name__ == "__main__":
    img = cv2.imread("../images/flower.png")
    # sp半径（双精度的半径，值越大图像模糊程度就越高，值越小图像模糊程度就越低）
    sp = 20
    # sr色彩副值（两个颜色的值相差多少表示相近颜色，值越大图像里面块连成一片区域的可能性就大，值越小图像里面块连成一片区域的可能性就小）
    sr = 30
    # 平滑处理图像
    new = cv2.pyrMeanShiftFiltering(img,sp,sr)
    # 查找图像边缘
    new_img = cv2.Canny(new,150,300)
    # 查找轮廓
    contours,_ = cv2.findContours(new_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    cv2.drawContours(img,contours,-1,(0,0,255,2))

    cv2.imshow("img",img)
    cv2.imshow("new",new)
    cv2.imshow("new_img", new_img)

    cv2.waitKey(-1)
