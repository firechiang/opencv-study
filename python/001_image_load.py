import cv2

if __name__ == "__main__":
    # 创建一个显示窗口,窗口的名字是窗口的唯一ID（cv2.WINDOW_NORMAL 创建的窗口可以改变大小）
    cv2.namedWindow("图片",cv2.WINDOW_AUTOSIZE)
    # 改变窗口大小
    #cv2.resizeWindow("new",800,500)
    #cv2.imshow("new",0)
    # cv2.IMREAD_ANYCOLOR 图片原始颜色，cv2.IMREAD_GRAYSCALE 转灰色图片
    mat = cv2.imread("../images/aaa.jpeg",cv2.IMREAD_ANYCOLOR)
    # 显示图片窗口
    cv2.imshow("图片",mat)
    # 保存图片（将图片写入磁盘）
    #cv2.imwrite("/home/chiangfire/images/aaa.png",mat)
    # 窗口显示多长时候后窗口自动关闭，如果不调用该函数窗口创建后会立即自动关闭（waitKey表示在等待的过程当中可以接收鼠标和键盘事件； 0 表示无限等待）
    key = cv2.waitKey(0)
    # 如果按了q键就退出（ord函数获取q键的Ascii码）
    if key & 0xFF == ord("q"):
        # 退出
        #exit()
        # 释放所有资源并且自动退出
        cv2.destroyAllWindows()
