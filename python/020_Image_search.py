'''
图像查找（主要用到两种技术，特征匹配和单应性矩阵）
图像查找是用一张图片去另一张图上查找是否存在第一张图片的内容
'''

import cv2
import numpy as np

if __name__ == "__main__":
    # 创建一个显示窗口,窗口的名字是窗口的唯一ID（cv2.WINDOW_NORMAL 创建的窗口可以改变大小）
    cv2.namedWindow("图片1", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("图片2(将查找到的图片标出来)", cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("图片3", cv2.WINDOW_AUTOSIZE)
    # 加载两张图片
    mat1 = cv2.imread("../images/opencv_search.png", cv2.IMREAD_ANYCOLOR)
    mat2 = cv2.imread("../images/opencv_orig.png", cv2.IMREAD_ANYCOLOR)
    # 先将图像转成灰度图像（将图像转变成单通道）
    gray1 = cv2.cvtColor(mat1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(mat2, cv2.COLOR_BGR2GRAY)

    # SIFT关键点检测简单使用（该算法速度慢，准确性高）
    # 注意：SIFT关键点检测是可以规避，Harris角点检测缺点的（比如：Harris角点具有旋转不变的特性，Harris角点检测在图像缩放或放大后，原来的角点有可能就不是角点了）
    # 创建SIFT算法检测对象
    sift = cv2.SIFT_create()

    # 使用SIFT算法对图像进行关键点和描述子（关键点描述信息，可用于特征匹配）检测（第二个参数是检测范围，传None表示检测整张图像）
    kps1, desc1 = sift.detectAndCompute(gray1, None)
    kps2, desc2 = sift.detectAndCompute(gray2, None)

    # 使用FLANN算法对两张图像的关键点和描述子进行匹配（FLANN算法优点：在进行批量匹配特征点匹配时，FLANN速度更快；FLANN算法缺点：由于它使用的是临近近似值，所以精度较差，如果是要做精准匹配建议使用暴力匹配算法）
    # 注意：详细的FLANN算法使用请查看上一章节的内容
    index_params = dict(algorithm=1,trees=5)
    search_param = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params,search_param)
    # 匹配两个特征点的描述子（第一个参数是第一张图的特征点描述子，第二个参数是第二张图的特征点描述子，k表示查找最优的2个匹配点就是第一张图中任意一个描述子与第二张图中的所有描述子进行匹配取最优的前k个）
    match = flann.knnMatch(desc1,desc2,k=2)
    # 用来装较好的匹配点
    good = []
    # 遍历优化所有匹配点（d1=第一张图的匹配点,d2=第二张图的匹配点)
    for i,(d1,d2) in enumerate(match):
        # distance（表示描述子之间的距离，值越低越好），queryIdx（第一张图的描述子索引值），trainIdx（第二张图的描述子索引值），imgIdx（第二张图的索引值）
        if(d1.distance < 0.7 * d2.distance):
            good.append(d1)


    # 好的点大于4才作单应性矩阵查找就是图片查找（因为单应性矩阵查找匹配点必须大于等于4否则无法查找）
    if len(good) >= 4:
        # 查找单应性矩阵简单使用
        # 原pts从匹配点中获取（queryIdx（第一张图的描述子索引值））
        # 下面代码就是遍历good数组，获取每一个元素，再用元素的queryIdx属性从kps集合中获取值，最后将值添加到一个新数组里面
        srcPts = [kps1[m.queryIdx].pt for m in good]
        # 先将所有原始值转成flaot型再转换成有无数行的，每一行中有一个元素是由两个子元素组成的（x=-1(可以是任意值)，y=1,z=2）
        srcPts = np.float32(srcPts).reshape(-1,1,2)
        # 目的pts从匹配点中获取（trainIdx（第二张图的描述子索引值））
        # 下面代码就是遍历good数组，获取每一个元素，再用元素的trainIdx属性从kps集合中获取值，最后将值添加到一个新数组里面
        disPts = [kps2[m.trainIdx].pt for m in good]
        # 先将所有原始值转成flaot型再转换成有无数行的，每一行中有一个元素是由两个子元素组成的（x=-1(可以是任意值)，y=1,z=2）
        disPts = np.float32(disPts).reshape(-1,1,2)
        # 查找单应性矩就是用原图去目标图上查找是否存在原图内容（srcPoints=原pts，dstPoints=目的pts，method=错误匹配点的过滤方法（cv2.RANSAC表示随机抽样过滤）,ransacReprojThreshold=错误匹配点的过滤阀值（一般是1-10之间））
        # matrix=单应性矩，_=掩码
        matrix,_ = cv2.findHomography(srcPoints=srcPts,dstPoints=disPts,method=cv2.RANSAC,ransacReprojThreshold=5.0)
        # 获取第一张图的高和宽
        h,w = mat1.shape[:2]
        # 构建要搜索图的4个角点（[0,0]左上角坐标，[0,h-1]左下角坐标，[w-1,h-1]右下角坐标，[w-1,0]右上角坐标）
        # reshape函数是数据转换成有无数行的，每一行中有一个元素是由两个子元素组成的（x = -1(可以是任意值)，y = 1, z = 2）
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        # 透视变换（将查找到的单应性矩阵转换为实际的坐标位置）
        dst = cv2.perspectiveTransform(pts,matrix)
        # 将查找到的图像位置在目标图上画出来
        cv2.polylines(mat2,[np.int32(dst)],True,(0,0,255))

    # 将两张图的匹配结果画出来（img1=第一张图，keypoints1=第一张图的特征点描述子，img2=第二张图，keypoints2=第二张图的特征点描述子，matches1to2=两张图的匹配结果，outImg=输出图像传None输出图像直接返回）
    # 注意：这个画匹配结果的函数只针对与使用 knnMatch 函数匹配出的结果
    #mat = cv2.drawMatchesKnn(img1=mat1,keypoints1=kps1,img2=mat2,keypoints2=kps2,matches1to2=[good],outImg=None)

    # 显示图片窗口
    cv2.imshow("图片1", mat1)
    cv2.imshow("图片2(将查找到的图片标出来)", mat2)
    #cv2.imshow("图片3", mat)
    # cv2.imshow("图片2",binary)
    # 窗口显示多长时候后窗口自动关闭，如果不调用该函数窗口创建后会立即自动关闭（waitKey表示在等待的过程当中可以接收鼠标和键盘事件； 0 表示无限等待）
    key = cv2.waitKey(0)
    # 如果按了q键就退出（ord函数获取q键的Ascii码）
    if key & 0xFF == ord("q"):
        # 退出
        # exit()
        # 释放所有资源并且自动退出
        cv2.destroyAllWindows()