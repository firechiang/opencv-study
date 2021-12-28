'''
读取视频
'''

import cv2

if __name__ == "__main__":
    cv2.namedWindow("video",cv2.WINDOW_AUTOSIZE)
    # 读取视频文件
    video = cv2.VideoCapture("/home/chiangfire/视频/asaa.mp4")
    while True:
        # 读取视频帧
        ret,frame = video.read()
        # 显示图像
        cv2.imshow("video",frame)
        # 窗口显示多长时候后窗口自动关闭，如果不调用该函数窗口创建后会立即自动关闭（waitKey表示在等待的过程当中可以接收鼠标和键盘事件； 1 表示等待1毫秒）
        key = cv2.waitKey(1)
        # 如果按了q键就退出（ord函数获取q键的Ascii码）
        if key & 0xFF == ord("q"):
            break

    # 释放摄像头资源
    video.release()
    # 释放所有资源并且自动退出
    cv2.destroyAllWindows()
