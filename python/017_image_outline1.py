'''
图像轮廓查找与计算（具有相同颜色或强度的连续点的曲线）
注意：为了保证图像轮廓检测的准确性，需要先对图像做二值化或Canny操作
'''

import cv2
import numpy as np

# 绘制边框
def drawShape(img,points):
    i = 0
    while i < len(points):
        if(i == len(points)- 1):
            x,y = points[i][0]
            x1,y1 = points[0][0]
            # 画线（img=图像，(x,y)=第一个点的坐标，(x1,y1)=第二个点的坐标，(0,0,255)=线的颜色，1=线宽）
            cv2.line(img,(x,y),(x1,y1),(0,0,255),1)
        else:
            x,y = points[i][0]
            x1,y1 = points[i+1][0]
            # 画线（img=图像，(x,y)=第一个点的坐标，(x1,y1)=第二个点的坐标，(0,0,255)=线的颜色，1=线宽）
            cv2.line(img,(x,y),(x1,y1),(0,0,255),1)
        i= i + 1

if __name__ == "__main__":
    # 创建一个显示窗口,窗口的名字是窗口的唯一ID（cv2.WINDOW_NORMAL 创建的窗口可以改变大小）
    cv2.namedWindow("画轮廓图片",cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("二值化图片", cv2.WINDOW_AUTOSIZE)


    # 查找轮廓简单使用
    # 加载原图
    mat = cv2.imread("../images/jjj.png", cv2.IMREAD_ANYCOLOR)
    # 先将图像转成灰度图（将图像转变成单通道）
    gray = cv2.cvtColor(mat,cv2.COLOR_BGR2GRAY)
    # 图像全局二值化（就是将整个图像变成黑白图像，要注意的是 threshold 函数只能处理灰度图像，所以要使用 threshold 函数需要先将图像转成灰度图）
    ret,binary = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
    # 查找模式
    # cv2.RETR_EXTERNAL = 表示只检测最外圈的大轮廓，里面的小轮廓不检测（效率高，适用于只检测大物体的轮廓）（实际使用较多）
    # cv2.RETR_LIST = 表示检测所有轮廓但不建立等级关系（适用于检测图片里面的所有轮廓，然后将其放到一个列表里面）
    # cv2.RETR_CCOMP = 表示检测所有轮廓并建立等级关系（每层最多建立两级关系）
    # cv2.RETR_TREE = 表示检测所有轮廓并按树形存储等级关系（实际使用较多）
    mode = cv2.RETR_EXTERNAL
    # 数据保留模式
    # cv2.CHAIN_APPROX_NONE = 保留所有轮廓上的点数据
    # cv2.CHAIN_APPROX_SIMPLE = 只保留角点数据
    AppMode = cv2.CHAIN_APPROX_SIMPLE
    # 查找轮廓（contours=查找到的所有的轮廓，hierarchy=查找的轮廓有没有层级关系）
    contours,hierarchy = cv2.findContours(binary,mode,AppMode)

    # 将轮廓数据转换成多边形坐标点简单使用
    e = 20
    approxCurve = True
    # 将轮廓数据转换成多边形坐标点（利用这些坐标点我们可以做一些自定义的东西，比如手动将轮廓画出来，这样画轮廓效率比较高，因为轮廓数据没有原始轮廓数据那么大）（contours[0]=第0个轮廓数据，e=精度，approxCurve=轮廓是否闭合（就是轮廓是否矩形或圆形之类的））
    approx = cv2.approxPolyDP(contours[0],e,approxCurve)
    # 绘制自定义多变行轮廓
    #drawShape(mat, approx)

    # 将轮廓数据转换成包围形坐标点简单使用（利用这些坐标点我们可以做一些自定义的东西，比如手动将轮廓画出来，这样画轮廓效率比较高，因为轮廓数据没有原始轮廓数据那么大）
    hull = cv2.convexHull(contours[0])
    # 绘制自定义包围行轮廓
    drawShape(mat, hull)

    # 绘制原始轮廓
    contourIdx = -1
    color = (0,0,255)
    thickness = 1
    # 绘制原始轮廓（mat=轮廓要绘制在哪个图像上，contours=轮廓数据，contourIdx=要绘制的轮廓序号（下标）轮廓数据是个List集合嘛-1表示绘制所有轮廓,color=轮廓颜色,thickness=轮廓线宽-1表示填充整个轮廓）
    # 注意：如果将轮廓颜色是彩色的绘制在二值化以后的图像上（比如 binary 图像）轮廓是显示不出来的，因为二值化以后的图像只有黑白，显示不了其它颜色
    #cv2.drawContours(mat,contours,contourIdx,color,thickness)

    # 计算轮廓面积
    # 计算轮廓面积（参数是轮廓数据）
    contourArea = cv2.contourArea(contours[0])
    print(f'第0个轮廓的面积={contourArea}')

    # 计算轮廓周长
    # 轮廓数据
    curve = contours[0]
    # 轮廓数据是否闭合（就是轮廓是否矩形或圆形之类的）
    closed = True
    perimeter = cv2.arcLength(curve,closed)
    print(f'第0个轮廓的周长={perimeter}')

    # 显示图片窗口
    cv2.imshow("画轮廓图片",mat)
    cv2.imshow("二值化图片",binary)
    # 窗口显示多长时候后窗口自动关闭，如果不调用该函数窗口创建后会立即自动关闭（waitKey表示在等待的过程当中可以接收鼠标和键盘事件； 0 表示无限等待）
    key = cv2.waitKey(0)
    # 如果按了q键就退出（ord函数获取q键的Ascii码）
    if key & 0xFF == ord("q"):
        # 退出
        #exit()
        # 释放所有资源并且自动退出
        cv2.destroyAllWindows()