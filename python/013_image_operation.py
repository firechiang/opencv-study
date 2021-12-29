'''
图像运算以及融合
'''

import cv2
import numpy as np

if __name__ == "__main__":
    # 创建一个显示窗口,窗口的名字是窗口的唯一ID（cv2.WINDOW_NORMAL 创建的窗口可以改变大小）
    #cv2.namedWindow("原图",cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("加法新图", cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("减法新图", cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("乘法新图", cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("除法新图", cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("融合新图", cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("非（位）运算原图", cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("非（位）运算新图", cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("与（位）运算原图", cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("与（位）运算新图", cv2.WINDOW_AUTOSIZE)

    #cv2.namedWindow("或（位）运算原图", cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("或（位）运算新图", cv2.WINDOW_AUTOSIZE)

    cv2.namedWindow("异或（位）运算原图", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("异或（位）运算新图", cv2.WINDOW_AUTOSIZE)

    img1 = cv2.imread("/home/chiangfire/images/aaa.jpeg",cv2.IMREAD_ANYCOLOR)
    shape = img1.shape
    print(f'宽度={shape[0]},高度={shape[1]}')
    # 创建一张图，宽高以及层次和img1一致。* 100 表示这张图的每一个像素的值都乘以100，也就是单个像素的值由1变成了100
    img2 = np.ones((shape[0],shape[1],shape[2]),np.uint8) * 100
    # 两张图相加（图的加法运算就是矩阵的加法运算也是单个像素的加法运算（所以两张图相加，两张图的宽度和高度必须一致，否则无法相加））
    # 加法一般使图像更亮一些
    img3 = cv2.add(img1,img2)
    # 两张图相减（图的加法运算就是矩阵的减法运算也是单个像素的减法运算（所以两张图相加，两张图的宽度和高度必须一致，否则无法相减））
    # 减法一般使图像更暗一些
    img4 = cv2.subtract(img3, img2)

    # 两张图相乘（图的加法运算就是矩阵的乘法运算也是单个像素的乘法运算（所以两张图相乘，两张图的宽度和高度必须一致，否则无法相乘））
    img5 = cv2.multiply(img1,img2)

    # 两张图相除（图的加法运算就是矩阵的除法运算也是单个像素的除法运算（所以两张图相乘，两张图的宽度和高度必须一致，否则无法相除））
    img6 = cv2.divide(img5,img1)

    # 两张图融合（图的加法运算就是矩阵的融合运算也是单个像素的融合运算（所以两张图融合，两张图的宽度和高度必须一致，否则无法融合））
    # 0.3表示img1融合后显示权重占比（越大显示越清晰），0.8表示img2融合后显示权重占比，0 表示融合后所有的像素都会加上这个值
    img7 = cv2.addWeighted(img1,0.3,img2,0.8,0)

    img8 = np.zeros((480,640),np.uint8)
    # [y1:y2,x1:x2]（检索roi矩阵 y 50 到 y 150 和 x 50 到 x 150 这一块的子矩阵数据让其值等于 255 也就是白色）
    img8[50:150,50:150] = 255
    # 对img8做非（位）运算（也就是将每个像素的值取反）
    #img9 = cv2.bitwise_not(img8)

    img10 = np.zeros((480, 640), np.uint8)
    # [y1:y2,x1:x2]（检索roi矩阵 y 100 到 y 250 和 x 100 到 x 250 这一块的子矩阵数据让其值等于 255 也就是白色）
    img10[100:250,100:250] = 255
    # 对img8与img10做与（位）运算（两张图做运算前提是两张图的宽度和高度必须一致，否则无法运算）
    # 与运算就是找出交叉块（只显示交叉块位置图像）
    #img11 = cv2.bitwise_and(img8,img10)

    # 对img8与img10做或（位）运算（两张图做运算前提是两张图的宽度和高度必须一致，否则无法运算）
    # 或运算就是重叠交叉块（也就是显示两张图的所有像素）
    #img12 = cv2.bitwise_or(img8,img10)

    # 对img8与img10做异或（位）运算（两张图做运算前提是两张图的宽度和高度必须一致，否则无法运算）
    # 异或运算就是不显示交叉块
    img13 = cv2.bitwise_xor(img8,img10)


    # 显示图片窗口
    #cv2.imshow("原图", img1)
    #cv2.imshow("加法新图",img3)
    #cv2.imshow("减法新图", img4)
    #cv2.imshow("乘法新图", img5)
    #cv2.imshow("除法新图", img6)
    #cv2.imshow("融合新图", img7)
    #cv2.imshow("非（位）运算原图", img8)
    #cv2.imshow("非（位）运算新图", img9)

    #cv2.imshow("与（位）运算原图", img8)
    #cv2.imshow("与（位）运算新图", img11)

    #cv2.imshow("或（位）运算原图", img8)
    #cv2.imshow("或（位）运算新图", img12)

    cv2.imshow("异或（位）运算原图", img8)
    cv2.imshow("异或（位）运算新图", img13)


    while True:
        # 窗口显示多长时候后窗口自动关闭，如果不调用该函数窗口创建后会立即自动关闭（waitKey表示在等待的过程当中可以接收鼠标和键盘事件； 0 表示无限等待）
        key = cv2.waitKey(0)
        # 如果按了q键就退出（ord函数获取q键的Ascii码）
        if key & 0xFF == ord("q"):
            break
            # 退出
            #exit()

# 释放所有资源并且自动退出
cv2.destroyAllWindows()
