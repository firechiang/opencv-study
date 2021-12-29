'''
通道分割和合并
'''

import cv2
import numpy as np

if __name__ == "__main__":
    img = np.zeros((480,640,3),np.uint8)

    # 通道分割（也就是将已经组合好的BGR，分割成单独的 B,G,R 通道（画布））
    b,g,r = cv2.split(img)

    # [y1:y2,x1:x2]（检索b矩阵 y 10 到 y 100 和 x 10 到 x 100 这一块的子矩阵数据让其值等于 255 也就是白色）
    b[10:100,10:100] = 255
    g[10:100, 10:100] = 255

    # 通道合并（合并纯色画布）
    img2 = cv2.merge((b,g,r))

    # 显示图片窗口
    cv2.imshow("img",b)
    cv2.imshow("img2",img2)

    # 窗口显示多长时候后窗口自动关闭，如果不调用该函数窗口创建后会立即自动关闭（waitKey表示在等待的过程当中可以接收鼠标和键盘事件； 0 表示无限等待）
    key = cv2.waitKey(0)
    # 如果按了q键就退出（ord函数获取q键的Ascii码）
    if key & 0xFF == ord("q"):
        # 退出
        #exit()
        # 释放所有资源并且自动退出
        cv2.destroyAllWindows()