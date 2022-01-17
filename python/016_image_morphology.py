'''
图像的形态学
1，基于图像形态进行处理的一些基本方法
2，这些处理方法基本是对二进制图像进行处理
3，卷积核决定着图像处理后的效果

图像形态处理方法
1，腐蚀（变小就像一块石头被岁月腐蚀得越来越小）和膨胀（变大）
2，开运算（对图像先做腐蚀再做膨胀）
3，闭运算（对图像先做膨胀再做腐蚀）
4，顶帽（插值运算就是给原有图像添加其它图像）
5，黑帽（插值运算）
'''

import cv2
import numpy as np

if __name__ == "__main__":
    # 创建一个显示窗口,窗口的名字是窗口的唯一ID（cv2.WINDOW_NORMAL 创建的窗口可以改变大小）
    cv2.namedWindow("旧图片",cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("新图片", cv2.WINDOW_AUTOSIZE)
    mat = cv2.imread("../images/fff.png",cv2.IMREAD_ANYCOLOR)

    # 图像全局二值化简单使用
    # 阀值（图像二值化的过程当中，源图像像素值大于100就取最大值，小于100就取0）
    thresh = 100
    # 最大值
    maxVal = 255
    # 类型：
    # cv2.THRESH_BINARY（表示源图像像素值大于100就取最大值，小于100就取0）（二值化常用API）
    # cv2.THRESH_BINARY_INV（表示源图像像素值大于100就取0，小于100就取最大值）（二值化常用API）
    # cv2.THRESH_TRUNC（表示源图像像素值大于100就取最大值，小于100就取实际值）（非二值化API主要用于削峰，不常用）
    # cv2.THRESH_TOZERO（表示源图像像素值大于100就取实际值，小于100就取0）（非二值化API主要用于削峰，不常用）
    # cv2.THRESH_TOZERO_INV（表示源图像像素值大于100就取最大值，小于100就取实际值）（非二值化API主要用于削峰，不常用）
    threshType = cv2.THRESH_BINARY
    # 先将图像转成灰度图像
    mat1 = cv2.cvtColor(mat,cv2.COLOR_BGR2GRAY)
    # 图像全局二值化（就是将整个图像变成黑白图像，要注意的是 threshold 函数只能处理灰度图像，所以要使用 threshold 函数需要先将图像转成灰度图）
    ret,new = cv2.threshold(mat1,thresh,maxVal,threshType)

    # 图像全局二值化自适应阀值简单使用（推荐二值化使用）
    # 计算阀值的方法：
    # cv2.ADAPTIVE_THRESH_MEAN_C（计算临近区域的平均值）
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C（高斯窗口加权平均值，就是临近就是中心点权重值较大，越边缘越小）（推荐使用这个方法）
    adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    # 临近区域的大小（要想计算更精准可以将blockSize设置小一点，要是不想那么精准就可以设置大一些（计算速度就比较快）
    # 3表示表示3*3大小的卷积核
    blockSize = 11
    # 常量，应从计算出的平均值或加权平均值中减去这个值（就是自适应计算出的阀值最后会减去这个值，最终得到阀值）
    C = 0
    # 类型：
    # cv2.THRESH_BINARY（表示源图像像素值大于100就取最大值，小于100就取0）（二值化常用API）
    # cv2.THRESH_BINARY_INV（表示源图像像素值大于100就取0，小于100就取最大值）（二值化常用API）
    threshType = cv2.THRESH_BINARY
    # 注意: adaptiveThreshold 函数只能处理灰度图像，所以要使用 adaptiveThreshold 函数需要先将图像转成灰度图
    new = cv2.adaptiveThreshold(mat1,maxVal,adaptiveMethod,threshType,blockSize,C)

    # 图像腐蚀简单使用
    # 构建一个3*3的像素，每个像素的值是1（如下图）的卷积核。用这个卷积核去扫描图像（和原始图像进行计算）
    # 1 1 1
    # 1 1 1
    # 1 1 1
    kernel = np.ones((3, 3), np.uint8)
    # 腐蚀次数
    iterations = 1
    # 腐蚀图像
    new = cv2.erode(mat,kernel,iterations)



    # 显示图片窗口
    cv2.imshow("旧图片",mat)
    cv2.imshow("新图片",new)
    # 窗口显示多长时候后窗口自动关闭，如果不调用该函数窗口创建后会立即自动关闭（waitKey表示在等待的过程当中可以接收鼠标和键盘事件； 0 表示无限等待）
    key = cv2.waitKey(0)
    # 如果按了q键就退出（ord函数获取q键的Ascii码）
    if key & 0xFF == ord("q"):
        # 退出
        #exit()
        # 释放所有资源并且自动退出
        cv2.destroyAllWindows()