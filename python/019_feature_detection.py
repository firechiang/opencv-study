'''
图像特征检测
1,图像平坦部分很难找到它在原图中的位置
2，边缘相比平坦部分要好找找一些，但也不能一下确定
3，角点可以一下就能找到其在原图的位置

什么是特征
1，图像特征就是指有意义的图像区域，具有独立独特性，易于识别性，比如角点，斑点以及密度区
2，在特征中最重要的就是角点

什么是角点
1，灰度梯度的最大值对应的像素
2，两条线的交点
3，极值点（一阶导数最大值，但二阶导数为0）

角点包含的信息
1，位置，大小，方向
2，描述子（记录了角点周围对其有贡献的像素点的一组向量值，其不受仿射变换，关照变换等影响）

Harris角点说明
1，光滑地区（无论向哪里移动，衡量系数不变）
2，边缘地区（垂直边缘移动时，衡量系数变化剧烈）
3，在交点处（无论往那个方向移动，衡量系数都变化剧烈）

Harris角点检测缺点
1，Harris角点具有旋转不变的特性
2，图像在缩放后，原来的角点有可能就不是角点了
'''

import cv2
import numpy as np

if __name__ == "__main__":
    # 创建一个显示窗口,窗口的名字是窗口的唯一ID（cv2.WINDOW_NORMAL 创建的窗口可以改变大小）
    cv2.namedWindow("原图片", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("角图片", cv2.WINDOW_AUTOSIZE)
    # 加载原图
    mat = cv2.imread("../images/lll.png", cv2.IMREAD_ANYCOLOR)

    # Harris角点检测简单使用
    # 先将图像转成灰度图像（将图像转变成单通道）
    gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    # Harris角点检测（img（图像）,blockSize（检测窗口大小），ksize（Sobel卷积核），k（权重系数，经验值，一般取 0.02 - 0.04 之间））
    # 注意：Harris角点检测的原图必须是灰度图像
    dst = cv2.cornerHarris(gray,blockSize = 2,ksize = 3,k = 0.04)
    # 显示Harris角点（将大于角点最大值百分之一的位置的像素变成红色）
    #mat[dst>0.01*dst.max()] = [0,0,255]

    # Shi-Tomasi角点检测简单使用
    # 检测Shi-Tomasi角点
    # maxCorners（角点的最大数，值为0表示无限制）
    # qualityLevel（需要过滤掉的角点值，取小于1.0的正数，一般在0.01-0.1之间）
    # minDistance（角点间最小距离，忽略小于此距离的角点（就是过滤掉太近的角点）））
    # mask（感兴趣的区域，对哪个区域进行角点检测，没设置就对整幅图像进行检测）
    # blockSize（检测窗口大小）
    # useHarrisDetector（是否使用 Harris 算法）
    # k（Harris算法，权重系数，经验值，一般取 0.02 - 0.04 之间，如果不使用Harris 算法该参数无效）
    # 注意：Shi-Tomasi角点检测的原图必须是灰度图像
    corners = cv2.goodFeaturesToTrack(gray,maxCorners=1000,qualityLevel=0.01,minDistance=10,mask=None,blockSize=None,useHarrisDetector=False,k=None)
    # 因为Shi-Tomasi角点数据是浮点型的所有要将其转换为整型
    corners = np.int0(corners)
    # 遍历角点并绘制出来
    for i in corners:
        # 将多维数组转为一维数组
        x,y = i.ravel()
        # 画圆（将角点位置绘制出来）
        #cv2.circle(mat,(x,y),3,(0,0,255),-1)

    # SIFT关键点检测简单使用（该算法速度慢，准确性高）
    # 注意：SIFT关键点检测是可以规避，Harris角点检测缺点的（比如：Harris角点具有旋转不变的特性，Harris角点检测在图像缩放或放大后，原来的角点有可能就不是角点了）
    # 创建SIFT算法检测对象
    sift = cv2.SIFT_create()
    # 使用SIFT算法对图像进行关键点检测（第二个参数是检测范围，传None表示检测整张图像）
    # 注意：SIFT关键点检测的原图必须是灰度图像
    #kps = sift.detect(gray,None)
    # 使用SIFT算法对图像进行关键点和描述子（关键点描述信息，可用于特征匹配）检测（第二个参数是检测范围，传None表示检测整张图像）
    # 注意：SIFT关键点检测的原图必须是灰度图像
    kps, desc = sift.detectAndCompute(gray, None)
    # 根据原图像和关键点来计算角点（关键点）的描述子（描述信息），这写描述信息可作为特征匹配（注意：上一个API是将关键点和描述子同时都计算出来的，推荐使用上一个API）
    #kps,desc = sift.compute(gray,kps)
    # 绘制检测到的角点（image检测原图像，keypoints检测到的点，outImage点要绘制在哪张图像上，color点的颜色）
    #cv2.drawKeypoints(image=gray,keypoints=kps,outImage=mat,color=(0,0,255))
    # 遍历绘制检测到的角点
    for kp in kps:
        x, y = kp.pt
        #cv2.circle(mat, (int(x), int(y)), 3, (0,0,255),-1)

    # SURF关键点检测简单使用（该算法速度快，准确性高）（注意：SURF算法已申请专利，需要手动编译安装OpenCV，否则无法使用该算法）
    # 创建SURF算法检测对象
    #surf = cv2.xfeatures2d.SURF_create()
    # 使用SURF算法对图像进行关键点检测（第二个参数是检测范围，传None表示检测整张图像）
    # 注意：SURF关键点检测的原图必须是灰度图像
    #kps = surf.detect(gray,None)
    # 使用SURF算法对图像进行关键点和描述子（关键点描述信息，可用于特征匹配）检测（第二个参数是检测范围，传None表示检测整张图像）
    # 注意：SURF关键点检测的原图必须是灰度图像
    #kps, desc = surf.detectAndCompute(gray, None)
    # 根据原图像和关键点来计算角点（关键点）的描述子（描述信息），这写描述信息可作为特征匹配（注意：上一个API是将关键点和描述子同时都计算出来的，推荐使用上一个API）
    #kps,desc = surf.compute(gray,kps)
    # 绘制检测到的角点（image检测原图像，keypoints检测到的点，outImage点要绘制在哪张图像上，color点的颜色）
    #cv2.drawKeypoints(image=gray,keypoints=kps,outImage=mat,color=(0,0,255))
    # 遍历绘制检测到的角点
    for kp in kps:
        x, y = kp.pt
        cv2.circle(mat, (int(x), int(y)), 3, (0,0,255),-1)

    # 显示图片窗口
    cv2.imshow("原图片", gray)
    cv2.imshow("角图片", mat)
    # cv2.imshow("图片2",binary)
    # 窗口显示多长时候后窗口自动关闭，如果不调用该函数窗口创建后会立即自动关闭（waitKey表示在等待的过程当中可以接收鼠标和键盘事件； 0 表示无限等待）
    key = cv2.waitKey(0)
    # 如果按了q键就退出（ord函数获取q键的Ascii码）
    if key & 0xFF == ord("q"):
        # 退出
        # exit()
        # 释放所有资源并且自动退出
        cv2.destroyAllWindows()