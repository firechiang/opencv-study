'''
查找轮廓的最小外接矩形和最大外接矩形（就是查找最小或最大矩形轮廓，适用场景将一副图像里面的某个小图像的角度纠正，或者是旋转图像里面的小图像）
'''

import cv2
import numpy as np

if __name__ == "__main__":
    # 创建一个显示窗口,窗口的名字是窗口的唯一ID（cv2.WINDOW_NORMAL 创建的窗口可以改变大小）
    cv2.namedWindow("图片1",cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("图片2", cv2.WINDOW_AUTOSIZE)
    # 加载原图
    mat = cv2.imread("../images/kkk.jpeg", cv2.IMREAD_ANYCOLOR)
    # 先将图像转成灰度图（将图像转变成单通道）
    gray = cv2.cvtColor(mat,cv2.COLOR_BGR2GRAY)
    # 图像全局二值化（就是将整个图像变成黑白图像，要注意的是 threshold 函数只能处理灰度图像，所以要使用 threshold 函数需要先将图像转成灰度图）
    ret,binary = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
    # 查找模式
    # cv2.RETR_EXTERNAL = 表示只检测最外圈的大轮廓，里面的小轮廓不检测（效率高，适用于只检测大物体的轮廓）（实际使用较多）
    # cv2.RETR_LIST = 表示检测所有轮廓但不建立等级关系（适用于检测图片里面的所有轮廓，然后将其放到一个列表里面）
    # cv2.RETR_CCOMP = 表示检测所有轮廓并建立等级关系（每层最多建立两级关系）
    # cv2.RETR_TREE = 表示检测所有轮廓并按树形存储等级关系（实际使用较多）
    mode = cv2.RETR_TREE
    # 数据保留模式
    # cv2.CHAIN_APPROX_NONE = 保留所有轮廓上的点数据
    # cv2.CHAIN_APPROX_SIMPLE = 只保留角点数据
    AppMode = cv2.CHAIN_APPROX_SIMPLE
    # 查找轮廓（contours=查找到的所有的轮廓，hierarchy=查找的轮廓有没有层级关系）
    contours,hierarchy = cv2.findContours(binary,mode,AppMode)

    # 查找最小外接矩形（参数是传一个轮廓数据，它会自动找到最小外接矩形）
    minArea = cv2.minAreaRect(contours[1])
    print(f'起始点坐标={minArea[0]},宽高={minArea[1]},角度={minArea[2]}')
    # 将查找到的最小外接矩形数据转换成坐标点数据（其实就是只要起始点坐标和宽高数据）
    box = cv2.boxPoints(minArea)
    # 将浮点型数据转换成整型
    box = np.int0(box)
    # 在原图上画最小矩形也就是画最小矩形轮廓
    cv2.drawContours(mat,[box],0,(0,0,255),1)

    # 查找最大外接矩形（参数传一个轮廓数据，它会自动找到最大外接矩形）
    maxArea = cv2.boundingRect(contours[1])
    x,y,w,h = cv2.boundingRect(contours[1])
    print(f'起始点坐标={maxArea[0]},宽高={maxArea[1]}')
    # 在原图上画最大矩形也就是画最大矩形轮廓
    cv2.rectangle(mat,(x,y),(x+w,y+h),(0,0,255),1)

    # 显示图片窗口
    cv2.imshow("图片1",mat)
    #cv2.imshow("图片2",binary)
    # 窗口显示多长时候后窗口自动关闭，如果不调用该函数窗口创建后会立即自动关闭（waitKey表示在等待的过程当中可以接收鼠标和键盘事件； 0 表示无限等待）
    key = cv2.waitKey(0)
    # 如果按了q键就退出（ord函数获取q键的Ascii码）
    if key & 0xFF == ord("q"):
        # 退出
        #exit()
        # 释放所有资源并且自动退出
        cv2.destroyAllWindows()