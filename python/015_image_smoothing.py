'''
图像滤波（卷积核就是一块自定义的像素）也就是图片通过卷积核扫描（图片原始像素点的值与卷积核像素点值相计算）以后生成一张新的图片
卷积概念：
1，卷积核可以看作是一块像素，然后用这块像素，对图像原始像素进行扫描，也就是进行计算（如果卷积核是一个像素，那么就是用这一个像素和整张图片的每一个像素进行计算，
如果卷积核是一个3*3的像素（入下图）那就是一次计算原始图像的9个像素，因为一次计算一块（一个卷积核）像素
123
254
964
）
卷积核大小（一般为奇数（比如 3*3，5*5，7*7），因为需要保证输出大小与原始大小保持一致还有如果是奇数的话锚点始终在卷积核的正中心，可以防止位置发生偏移
在深度学习中，卷积核越大，看到的信息（感受野）越多，因此提取的特征越好，同时计算量就会变大）

2，锚点（下面的数据就是一个3*3的卷积核，5就是中心锚点）
147
258
364

3，边界扩充（当卷积核大于1且不进行边界扩充，输出尺寸将相应缩小；当卷积核以标准方式进行边界扩充，则输出数据的空间尺寸将与输入相等）
边界扩充公式：
N（输出图像大小） = （W（源图大小） - F（卷积核大小） + 2P（扩充尺寸）） / S（步长大小） + 1

4，步长（每扫描一块相隔几个像素，扫描下一块）比如下图是原始图像，卷积核大小是1，步长是2那么第一次扫描是在原始图像1的位置，第二次扫描是在原始图像5的位置

12354532
65548421
21245185


卷积核可以是低通滤波或高通滤波概念
1，低通滤波可以去除噪音或平滑图像（美颜）
2，高通滤波可以帮助查找图像的边缘
'''

import cv2
import numpy as np

if __name__ == "__main__":
    # 创建一个显示窗口,窗口的名字是窗口的唯一ID（cv2.WINDOW_NORMAL 创建的窗口可以改变大小）
    cv2.namedWindow("旧图片",cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("新图片", cv2.WINDOW_AUTOSIZE)
    mat = cv2.imread("../images/aaa.jpeg",cv2.IMREAD_ANYCOLOR)

    # 卷积核简单使用
    # (ddpeth表示图像经过滤波之后它的位深，可取值cv2.CV_8F，cv2.CV_16F，cv2.CV_32F，cv2.CV_64F，一般我们设置-1就是原始图像的位深是多少输出图像的位深就是多少)
    ddpeth = -1
    # 卷积核（如果是低通滤波就是美颜，如果是高通滤波就是查找图像的边缘）（注意：卷积核越大计算越精准，也就是处理效果越好）
    # 构建一个5*5的像素，每个像素的值是1（如下图），最后每个像素值除以25得到一个新的像素块，也就是卷积核。用这个卷积核去扫描图像（和原始图像进行计算）
    # 1 1 1 1 1      0.04 0.04 0.04 0.04 0.04
    # 1 1 1 1 1      0.04 0.04 0.04 0.04 0.04
    # 1 1 1 1 1      0.04 0.04 0.04 0.04 0.04
    # 1 1 1 1 1      0.04 0.04 0.04 0.04 0.04
    # 1 1 1 1 1      0.04 0.04 0.04 0.04 0.04
    kernel = np.ones((5,5),np.float32) / 25
    # 锚点（取值-1表示根据卷积核的内容自动找到卷积核的中心点也就是锚点）
    anchor = -1
    # 经过卷积核计算以后像素点的值，还可以增加或减少一个指定的值，这个值就是下面的参数
    delta = 0
    # 边界类型（一般不传）
    borderType = cv2.BORDER_DEFAULT
    # 经过卷积核计算
    new = cv2.filter2D(mat,ddpeth,kernel)
    #new = cv2.filter2D(mat, ddpeth, kernel,borderType)

    # OpenCV自己实现的低通滤波算法如下（注意：低通滤波可以去除噪音或平滑图像（美颜））：

    #方盒均值滤波简单使用
    # normailze = True 那么 a = 1/（W（滤波器宽度） * H（滤波器高度））
    # 比如卷积核是3*3的那么a就等于9分之1（也就是当normailze=True时方盒滤波就等于平均滤波）
    normailze = True
    # normailze = False 那么 a = 1
    #normailze = False
    # 卷积核大小（下面表示5*5大小的卷积核）（注意：卷积核越大计算越精准，也就是处理效果越好）
    ksize = (5,5)
    # 这个函数一般不用
    #new = cv2.boxFilter(mat,ddpeth,ksize,anchor,normailze,borderType)
    # 一般用这个函数（注意：这个函数里面其实就是调用了上面的那个函数 normailze 传的是True）
    new = cv2.blur(mat,ksize)
    #new = cv2.blur(mat, ksize, anchor, borderType)

    # 高斯滤波简单使用（高斯滤波就是一条线中间高两边低，主要用于处理图像的高斯噪点，也就是降噪）如下图数据
    # 1 2 1
    # 2 3 2
    # 3 4 3
    # 卷积核大小（下面表示5*5大小的卷积核）（注意：如果没有设置sigma的值，数据处理主要以卷积核大小为基准）
    ksize = (5, 5)
    # X轴到中心点误差（也就是X轴的拓展长度）
    sigmaX = 1
    # Y轴到中心点误差（也就是Y轴的拓展长度）
    sigmaY = 1
    new = cv2.GaussianBlur(mat,ksize,sigmaX=sigmaX,sigmaY=sigmaY)

    # 中值滤波简单使用（卷积核每一个像素位的值都是一个数组，然后取其中的中间值作为卷积后的结果值），比如数组 [1,5,5,3,6,8,9] 我们就取3
    # 中值滤波的优点就是处理胡椒噪音（噪点）胡椒噪点就是一张图片满图都是噪点（像胡椒粉洒在上面一样），这种就是胡椒噪点
    # 卷积核大小（5表示表示5*5大小的卷积核）
    ksize = 5
    new  = cv2.medianBlur(mat,ksize)

    # 双边滤波简单使用（保留边缘同时对边缘内的区域进行平滑处理），主要作用就是美颜
    # 双边滤波算法详情：https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
    # 直径（两边边缘的距离）
    d = 50
    # 要忽略掉的颜色范围
    sigmaColor = 20
    # 要进行平滑处理的范围
    sigmaSpace = 100
    new = cv2.bilateralFilter(mat,d,sigmaColor,sigmaSpace)

    # OpenCV自己实现的高通滤波算法如下（注意：高通滤波可以帮助查找图像的边缘）：
    # Sobel（索贝尔）算法查找图像边缘，只能计算一个坐标轴的像素，要么是X要么是Y（主要特点就是抗噪点比较强，也就是如果图像噪点比较多的话要查找图线边缘可以使用Sobel（索贝尔）算法）
    # 注意：如果将卷积核参数传-1默认就是使用Scharr（沙尔）算法
    # ddpeth参数表示图像经过滤波之后它的位深，可取值cv2.CV_8F，cv2.CV_16F，cv2.CV_32F，cv2.CV_64F，一般我们设置-1就是原始图像的位深是多少输出图像的位深就是多少
    ddpeth = cv2.CV_64F
    # X设为1那就是求Y的边缘，Y必须设为0（因为它只能求一个方向的边缘）
    dx = 1
    # Y如果设为1那就是求X的边缘，X必须设为0（因为它只能求一个方向的边缘）
    dy = 0
    # 缩放比例
    scale = 1
    mat = cv2.imread("../images/ddd.jpeg", cv2.IMREAD_ANYCOLOR)
    #new =cv2.Sobel(mat,ddpeth,dx,dy,ksize,scale,delta,borderType)
    # 求Y轴边缘
    new1 = cv2.Sobel(mat, ddpeth, dx, dy, ksize)
    # 求X轴边缘
    new2 = cv2.Sobel(mat, ddpeth, dy,dx, ksize)
    # 两个边缘都求出来以后，再相加就是这张图片的完整边缘
    #new = new1 + new2
    new = cv2.add(new1,new2)

    # Scharr（沙尔）算法查找图线边缘，，只能计算一个坐标轴的像素，要么是X要么是Y（卷积核不可改变，尺寸固定就是3*3）（注意：如果Sobel（索贝尔）算法卷积核参数传-1默认就是使用Scharr（沙尔）算法）
    # 注意：Scharr（沙尔）算法不常用，一般都是用Sobel（索贝尔）算法
    # new =cv2.Scharr(mat,ddpeth,dx,dy,scale,delta,borderType)
    # 求Y轴边缘
    new1 = cv2.Scharr(mat, ddpeth, dx, dy)
    # 求X轴边缘
    new2 = cv2.Scharr(mat, ddpeth, dy,dx)
    # 两个边缘都求出来以后，再相加就是这张图片的完整边缘
    new = cv2.add(new1,new2)

    # Laplacian（拉普拉斯）同时计算整张图的像素的边缘，但是不能处理噪点多的图像，需要手动先给图像降噪，再调用该算法
    #new = cv2.Laplacian(mat,ddpeth,ksize,scale,borderType)
    new = cv2.Laplacian(mat, ddpeth, ksize)

    # 边缘检测终极大法Canny算法（原理：首先使用5*5高斯滤波消除噪点，再从4个角度（0、45，90，135）使用Sobel（索贝尔）算法计算最后用4个方向的局部最大值来计算边缘，最大值大于阀值就是边缘，小于最低阀值也是边缘，在两个阀值之间则需要计算）
    # 最大阀值（计算得到的值大于这个值就是边缘）
    maxVal = 100
    # 最小阀值（计算得到的值小于这个值就是边缘）
    minVal = 200
    new = cv2.Canny(mat,minVal,maxVal)

    # 显示图片窗口
    cv2.imshow("旧图片",mat)
    cv2.imshow("新图片",new)
    # 窗口显示多长时候后窗口自动关闭，如果不调用该函数窗口创建后会立即自动关闭（waitKey表示在等待的过程当中可以接收鼠标和键盘事件； 0 表示无限等待）
    key = cv2.waitKey(0)
    # 如果按了q键就退出（ord函数获取q键的Ascii码）
    if key & 0xFF == ord("q"):
        # 退出
        #exit()
        # 释放所有资源并且自动退出
        cv2.destroyAllWindows()