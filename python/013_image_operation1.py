'''
图片添加水印
1，导入图片
2，需要一张Logo图片
3，计算图片在什么位置添加，在添加的地方变成黑色
4，利用add函数将Logo与图片叠加在一起
'''

import cv2
import numpy as np

if __name__ == "__main__":
    img = cv2.imread("/home/chiangfire/images/aaa.jpeg",cv2.IMREAD_ANYCOLOR)
    shape = img.shape
    print(f'宽度={shape[0]},高度={shape[1]}')
    # Logo图片
    logo = np.zeros((200,200,3),np.uint8)
    # 掩码图片（没指定通道，默认就是单通道）
    mask = np.zeros((200,200),np.uint8)
    # [y1:y2,x1:x2]（检索logo矩阵 y 20 到 y 120 和 x 20 到 x 120 这一块的子矩阵数据让其值等于 [0,0,255] 也就是红色）
    logo[20:120,20:120] = [0,0,255]
    # [y1:y2,x1:x2]（检索logo矩阵 y 80 到 y 180 和 x 80 到 x 180 这一块的子矩阵数据让其值等于 [0,255,0] 也就是绿色）
    logo[80:180,80:180] = [0,255,0]
    # [y1:y2,x1:x2]（检索logo矩阵 y 20 到 y 120 和 x 20 到 x 120 这一块的子矩阵数据让其值等于 255 也就是白色）
    mask[20:120,20:120] = 255
    # [y1:y2,x1:x2]（检索logo矩阵 y 80 到 y 180 和 x 80 到 x 180 这一块的子矩阵数据让其值等于 255 也就是白色）
    mask[80:180,80:180] = 255
    # 对mask做非（位）运算（也就是将每个像素的值取反）
    m = cv2.bitwise_not(mask)

    # 选择位置添加水印（注意：添加水印位置的宽高要和logo图片的宽高一致）
    # [y1:y2,x1:x2]（检索出 y 0 到 y 200 和 x 0 到 x 200 这一块的子矩阵数据）
    roi = img[0:200,0:200]
    # 对roi与m做与（位）运算（两张图做运算前提是两张图的宽度和高度必须一致，否则无法运算）
    # 因为roi与m（单通道），通道数不一致，所以使用下面写法，如果通道数一致，可直接这样写：cv2.bitwise_and(roi,m)
    # 与运算就是找出交叉块（只显示交叉块位置图像）
    tmp = cv2.bitwise_and(roi,roi,mask=m)
    # 叠加水印
    dst = cv2.add(tmp,logo)
    # 将叠加好的水印数据赋值给图片的指定区域
    img[0:200, 0:200] = dst

    cv2.imshow("img",img)

    while True:
        # 窗口显示多长时候后窗口自动关闭，如果不调用该函数窗口创建后会立即自动关闭（waitKey表示在等待的过程当中可以接收鼠标和键盘事件； 0 表示无限等待）
        key = cv2.waitKey(0)
        # 如果按了q键就退出（ord函数获取q键的Ascii码）
        if key & 0xFF == ord("q"):
            break
            # 退出
            # exit()

# 释放所有资源并且自动退出
cv2.destroyAllWindows()