'''
用鼠标绘画
1，按下A键，滑动鼠标画线
2，按下S键，即可画矩形
3，按下D键，即可画圆
'''
import cv2
import numpy as np

# 鼠标回调函数
def mouse_callback(event,x,y,flags,userdata):
    # 表示 startpos为全局变量
    global startpos
    # 鼠标按下左键
    if event & cv2.EVENT_LBUTTONDOWN == cv2.EVENT_LBUTTONDOWN:
        startpos = (x,y)
    # 鼠标左键抬起
    elif event & cv2.EVENT_LBUTTONUP == cv2.EVENT_LBUTTONUP:
        if curshape == 0:
            # 画线
            cv2.line(mat,startpos,(x,y),(0,0,255))
        if curshape == 1:
            cv2.rectangle(mat,startpos,(x,y),(0,0,255))
        if curshape == 2:
            a = (x - startpos[0])
            b = (y - startpos[1])
            r = int((a**2+b**2)**0.5)
            cv2.circle(mat,startpos,r,(0,0,255))

if __name__ == "__main__":
    curshape = 0
    # 画线起始点
    startpos = (0, 0)
    # 创建全黑图像（3个360乘640面积，也可以说是3层像素）
    mat = np.zeros((480,640,3),np.uint8)
    # 创建一个显示窗口,窗口的名字是窗口的唯一ID（cv2.WINDOW_NORMAL 创建的窗口可以改变大小）
    cv2.namedWindow("mouse",cv2.WINDOW_AUTOSIZE)
    # 设置鼠标回调函数（注意：123是自定义参数也就是回调函数里面的userdata参数的值）
    cv2.setMouseCallback("mouse",mouse_callback,"123")

    while True:
        # 显示图片窗口
        cv2.imshow("mouse",mat)
        # 窗口显示多长时候后窗口自动关闭，如果不调用该函数窗口创建后会立即自动关闭（waitKey表示在等待的过程当中可以接收鼠标和键盘事件； 0 表示无限等待）
        key = cv2.waitKey(1) & 0xFF
        # 如果按了q键就退出（ord函数获取q键的Ascii码）
        if key == ord("q"):
            break
        if key == ord("a"):
            curshape = 0
        if key == ord("s"):
            curshape = 1
        if key == ord("d"):
            curshape = 2

# 释放所有资源并且自动退出
cv2.destroyAllWindows()