'''
图像变换
'''

import cv2
import numpy as np


if __name__ == "__main__":
    img = cv2.imread("/home/chiangfire/images/aaa.jpeg", cv2.IMREAD_ANYCOLOR)
    # 图线缩放
    # (400,400)缩放大小
    #new = cv2.resize(img,(400,400))

    # 图片缩放算法举例
    #INTER_NEAREST（临近插值，速度快，效果差）
    #INTER_LINEAR(双线插值，原图中的4个点，速度快，效果还行)
    #INTER_CUBIC（三次插值，原图中的16个点，速度还行，效果更好）
    #INTER_AREA（效果最好，速度最慢）
    # 宽高设置成None，整个图片的大小设置成 0.3 * 0.3,interpolation 为图片缩放算法
    new  = cv2.resize(img,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_AREA)

    # 图像翻转
    # flipCode（0：上下翻转，大于0：左右翻转，小于0：上下和左右翻转）
    flip = cv2.flip(img,flipCode=1)

    # 图线旋转（rotateCode=旋转角度）
    rotate = cv2.rotate(img,rotateCode=cv2.ROTATE_180)

    height,width,ch = img.shape

    # 图像的旋转，缩放，平移，总称仿射变换

    # 生成图像向右平移矩阵（就是用下面的内容来填充图像左边的内容）
    # x轴数据为 1和0，y轴的数据为 0和1（数组的第三个值为偏移量，就是x轴平移100也就是向右平移100个像素，y轴平移0也就是向下平移0个像素）（注意：负值可以向相反的方向平移）
    m = np.float32([[1,0,100],[0,1,10]])

    # 生成图像旋转矩阵
    # 旋转角度为逆时针旋转（(width/2,height/2)旋转中心点，30是旋转角度，0.9是缩放比例）
    m = cv2.getRotationMatrix2D((width/2,height/2),30,0.9)

    # 生成图像3D变换矩阵
    # 横的两点确定一条直线，竖的两点确定一条直线
    # 源三个点的位置（第一个点的位置 x=400，y=300）（第二个点的位置 x=800，y=300）（第三个点的位置 x=400，y=1000）
    src = np.float32([[400,300],[800,300],[400,1000]])
    # 变换后三个点的位置（第一个点的位置 x=200，y=400）（第二个点的位置 x=600，y=500）（第三个点的位置 x=150，y=1100）
    dst = np.float32([[200,400],[600,500],[150,1100]])
    # src 图像源三点，dst变换后三点
    m = cv2.getAffineTransform(src,dst)

    # 生成图像透视变换矩阵（就是可以将3D倾斜图像变换成方方正正的正面图片 方便看清里面的内容。原理其实是利用源图像的四个角将图像扣出来，形成一张新的图像）
    img = cv2.imread("/home/chiangfire/images/ccc.jpg", cv2.IMREAD_ANYCOLOR)
    # 从源图像的哪四个角将图像扣出来，形成一张新的图像
    # 源图像四个角（点）的位置（第一个点的位置 x=100，y=1100）（第二个点的位置 x=2100，y=1100）（第三个点的位置 x=0，y=4000）（第四个点的位置 x=2500，y=3900）
    src = np.float32([[100,1100],[2100,1100],[0,4000],[2500,3900]])
    # 源图像四个角（点）的位置（第一个点的位置 x=0，y=0）（第二个点的位置 x=2300，y=0）（第三个点的位置 x=0，y=2300）（第四个点的位置 x=2300，y=3000）
    dst = np.float32([[0,0],[2300,0],[0,3000],[2300,3000]])
    m = cv2.getPerspectiveTransform(src,dst)

    dsize = (width,height)
    # 旋转或平移变换
    # m（变换矩阵），dsize（变换后的图像输出大小），flag（缩放算法），mode（边界外推法标志），value（填充边界的值）
    #warp = cv2.warpAffine(img,m,dsize)

    # 透视变换
    # m（变换矩阵），dsize（变换后的图像输出大小），flag（缩放算法），mode（边界外推法标志），value（填充边界的值）
    warp = cv2.warpPerspective(img,m,dsize)

    cv2.imshow("img", warp)

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