'''
图像分割（将前景物体从背景物体中分离出来）
1，传统的图像分割方法（1，分水岭，2，GrabCut法，3，MeanShift法，4，背景抠出）
2，基于深度学习的图像分割方法
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 传统的图像分割方法之分水岭法实现
def watershed():
    # 标识背景
    # 标记前景
    # 标识未知域
    # 进行分割
    img = cv2.imread("../images/mmm.jpeg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图像二值化
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 开运算
    kernel = np.ones((3, 3), np.int8)
    open1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # 膨胀
    bg = cv2.dilate(open1, kernel, iterations=1)
    # 获取前景物体
    # 计算非0值到最近0值的距离
    # distanceType=cv2.DIST_L1（绝对值的计算，非0值坐标减去最近0值坐标），cv2.DIST_L2（勾股定理计算））
    # maskSize（扫描时卷积核大小）（distanceType=cv2.DIST_L1取3，distanceType=cv2.DIST_L2取5）
    dist = cv2.distanceTransform(open1, cv2.DIST_L2, 5)
    # 前景二值化
    ret1, fg = cv2.threshold(dist, 0.7 * dist.max(), 255, cv2.THRESH_BINARY)
    # 测试显示代码
    # plt.imshow(dist,cmap='gray')
    # plt.show()
    # exit()

    # 获取未知区域（膨胀图形减去前景）
    fg = np.uint8(fg)
    unknow = cv2.subtract(bg, fg)
    # 获取前景图像的连通域
    # 计算图像所有非0元素的连通域（像素与像素之间有连接而且值相同就会变成连通域）
    # connectivity（根据几个像素来计算连通域一般取4或8（默认值））
    ret3, marker = cv2.connectedComponents(fg)
    # 将前景和背景设置不同值
    # 对marker所有元素都加1
    marker = marker + 1
    # marker的unknow位置的像素是255的就让其等于0
    marker[unknow == 255] = 0
    # 进行图像分割
    # marker一定是前景和背景设置不同值，否者无法区分它们
    res = cv2.watershed(img, marker)
    # 找打res等于-1的所有像素坐标，在img图像里面将这写坐标的值设置成 [0,0,255] 也就是红色
    img[res == -1] = [0, 0, 255]

    cv2.imshow("new", img)
    cv2.imshow("old", cv2.imread("../images/mmm.jpeg"))
    cv2.waitKey(-1)

# 传统的图像分割方法之GrabCut法实现（通过交互的方式获取前景物体（就是手动指定要分割图像的区域））
def grab_cut():
    cv2.imread("../images/mmm.jpeg")

if __name__ == "__main__":
    # 传统的图像分割方法之分水岭法分割图像
    watershed()
