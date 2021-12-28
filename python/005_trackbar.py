'''
滑块和进度条
'''

import cv2
import numpy as np

# 进度条回调函数
def callback():
    pass

if __name__ == "__main__":
    # 创建一个显示窗口,窗口的名字是窗口的唯一ID（cv2.WINDOW_NORMAL 创建的窗口可以改变大小）
    cv2.namedWindow("trackbar",cv2.WINDOW_AUTOSIZE)

    # 创建滑块进度条（注意：R是进度条的名称，trackbar是图像窗口的名称）
    cv2.createTrackbar("R",'trackbar',0,255,callback)
    # 创建滑块进度条
    cv2.createTrackbar("G", 'trackbar', 0, 255, callback)
    # 创建滑块进度条
    cv2.createTrackbar("B", 'trackbar', 0, 255, callback)

    # 改变窗口大小
    cv2.resizeWindow("trackbar",800,500)

    # 创建全黑图像（3个360乘640面积，也可以说是3层像素）
    # 行的个数480，列的个数640，通道数/层数3，每一个元素的类型是uint8值是0，也就是黑色
    # 可以想象成创建3副画，然后跌在一起。也就是一张画有三层
    mat = np.zeros((480,640,3),np.uint8)
    while True:
        # 获取进度条的值
        r = cv2.getTrackbarPos("R","trackbar")
        g = cv2.getTrackbarPos("G", "trackbar")
        b = cv2.getTrackbarPos("B", "trackbar")
        # 将进度条的值当成RGB颜色值
        mat[:] = [b,g,r]

        # 显示图像窗口
        cv2.imshow("trackbar",mat)

        # 窗口显示多长时候后窗口自动关闭，如果不调用该函数窗口创建后会立即自动关闭（waitKey表示在等待的过程当中可以接收鼠标和键盘事件； 0 表示无限等待）
        key = cv2.waitKey(1)
        # 如果按了q键就退出（ord函数获取q键的Ascii码）
        if key & 0xFF == ord("q"):
            break

# 释放所有资源并且自动退出
cv2.destroyAllWindows()