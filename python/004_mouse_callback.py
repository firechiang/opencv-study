'''
鼠标事件处理
'''

import cv2
import numpy as np
import numpy as nt

# 鼠标回调函数
def mouse_callback(event,x,y,flags,userdata):
    # 打印事件信息
    print(f'event={event},x={x},y={y},flags={flags},userdata={userdata}')

if __name__ == "__main__":
    # 创建一个显示窗口,窗口的名字是窗口的唯一ID（cv2.WINDOW_NORMAL 创建的窗口可以改变大小）
    cv2.namedWindow("mouse",cv2.WINDOW_AUTOSIZE)
    # 改变窗口大小
    cv2.resizeWindow("mouse",800,500)
    # 设置鼠标回调函数（注意：123是自定义参数也就是回调函数里面的userdata参数的值）
    cv2.setMouseCallback("mouse",mouse_callback,"123")
    # 创建全黑图像（3个360乘640面积，也可以说是3层像素）
    mat = np.zeros((360,640,3),np.uint8)
    while True:
        # 显示图片窗口
        cv2.imshow("mouse",mat)
        # 窗口显示多长时候后窗口自动关闭，如果不调用该函数窗口创建后会立即自动关闭（waitKey表示在等待的过程当中可以接收鼠标和键盘事件； 0 表示无限等待）
        key = cv2.waitKey(1)
        # 如果按了q键就退出（ord函数获取q键的Ascii码）
        if key & 0xFF == ord("q"):
            break

# 释放所有资源并且自动退出
cv2.destroyAllWindows()

