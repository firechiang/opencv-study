'''
色彩转换
'''

import cv2
import numpy as np

# 进度条回调函数
def callback():
    pass

if __name__ == "__main__":
    # 创建一个显示窗口,窗口的名字是窗口的唯一ID（cv2.WINDOW_NORMAL 创建的窗口可以改变大小）
    cv2.namedWindow("trackbar",cv2.WINDOW_AUTOSIZE)

    # 色彩转换类型数组
    colorsspaces = [cv2.COLOR_BGR2RGBA,cv2.COLOR_BGR2BGRA,
                    cv2.COLOR_BGR2GRAY,cv2.COLOR_BGR2HSV_FULL,
                    cv2.COLOR_BGR2YUV]

    # 创建滑块进度条（注意：R是进度条的名称，trackbar是图像窗口的名称）
    cv2.createTrackbar("R",'trackbar',0,len(colorsspaces) -1,callback)

    # 改变窗口大小
    cv2.resizeWindow("trackbar",800,500)

    # cv2.IMREAD_ANYCOLOR 图片原始颜色，cv2.IMREAD_GRAYSCALE 转灰色图片
    mat = cv2.imread("/home/chiangfire/images/aaa.jpeg",cv2.IMREAD_ANYCOLOR)
    while True:
        # 获取进度条的值
        index = cv2.getTrackbarPos("R","trackbar")
        # 色彩转换
        cvt_mat = cv2.cvtColor(mat,colorsspaces[index])
        # 将进度条的值当成RGB颜色值
        #mat[:] = [b,g,r]

        # 显示转换后图像
        cv2.imshow("trackbar",cvt_mat)

        # 窗口显示多长时候后窗口自动关闭，如果不调用该函数窗口创建后会立即自动关闭（waitKey表示在等待的过程当中可以接收鼠标和键盘事件； 0 表示无限等待）
        key = cv2.waitKey(1)
        # 如果按了q键就退出（ord函数获取q键的Ascii码）
        if key & 0xFF == ord("q"):
            break

# 释放所有资源并且自动退出
cv2.destroyAllWindows()