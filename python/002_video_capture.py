'''
视频采集和录制
'''

import cv2

if __name__ == "__main__":
    cv2.namedWindow("video",cv2.WINDOW_AUTOSIZE)
    # 获取0号摄像头
    #cap = cv2.VideoCapture(0)
    # 打开远程摄像头
    cap = cv2.VideoCapture("http://192.168.3.130:4747/video")
    # 获取摄像头帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取视频宽
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # 获取视频高度
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # 设置写入视频格式
    foucc = cv2.VideoWriter.fourcc(*"MJPG")
    print(f'手机摄像头是否已经打开: {cap.isOpened()},宽:{width},高:{height},帧率: {fps}')
    # 创建要写入的视频文件
    video1 = cv2.VideoWriter("/home/chiangfire/视频/asaa1.mp4",foucc,int(fps),(int(width),int(height)))

    while cap.isOpened():
        # 读取视频帧
        ret,frame = cap.read()
        # 视频帧读取成功
        if ret == True:
            # 显示图像
            cv2.imshow("video",frame)
            #重新设置窗口大小
            #cv2.resizeWindow("video",640,320)
            # 写入视频帧到视频文件
            video1.write(frame)
            # 窗口显示多长时候后窗口自动关闭，如果不调用该函数窗口创建后会立即自动关闭（waitKey表示在等待的过程当中可以接收鼠标和键盘事件； 1 表示等待1毫秒）
            key = cv2.waitKey(1)
            # 如果按了q键就退出（ord函数获取q键的Ascii码）
            if key & 0xFF == ord("q"):
                break

    # 释放是您写入相关资源
    video1.release()
    # 释放摄像头资源
    cap.release()
    # 释放所有资源并且自动退出
    cv2.destroyAllWindows()