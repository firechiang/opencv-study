'''
Mat（矩阵）深拷贝和浅拷贝以及属性访问简单示例
'''

import cv2

if __name__ == "__main__":
    # cv2.IMREAD_ANYCOLOR 图片原始颜色，cv2.IMREAD_GRAYSCALE 转灰色图片
    mat = cv2.imread("../images/aaa.jpeg", cv2.IMREAD_ANYCOLOR)
    shape = mat.shape
    print(f'高度={shape[0]},宽度={shape[1]},通道数={shape[2]}')

    # 图像大小 = 高度 * 宽度 * 通道数
    print(f'图像大小={mat.size}')

    # 图像中每个元素的位深（就是每个元素值的类型）
    print(f'type={mat.dtype}')

    # 这种就是浅拷贝（两个mat的头信息内存位置不一样（就是拷贝了一份头信息），两个mat的实际数据的内存位置是一样的，也就是说数据是共享的）
    mat2 = mat

    # 这个就是深拷贝也就是说将整个数据从新拷贝了一份（包括头信息和数据体）
    mat3 = mat.copy()

    # 显示图片窗口
    cv2.imshow("img",mat)

    # 窗口显示多长时候后窗口自动关闭，如果不调用该函数窗口创建后会立即自动关闭（waitKey表示在等待的过程当中可以接收鼠标和键盘事件； 0 表示无限等待）
    key = cv2.waitKey(0)
    # 如果按了q键就退出（ord函数获取q键的Ascii码）
    if key & 0xFF == ord("q"):
        # 退出
        #exit()
        # 释放所有资源并且自动退出
        cv2.destroyAllWindows()

