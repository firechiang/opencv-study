'''
图像拼接
图像拼接步骤：
1，读取图像并重置其尺寸，因为尺寸不一样不能拼接
2，根据特征点和计算描述子，得到单应性矩阵
3，图像旋转变换
4，将图像拼接并输出其结果
'''

import cv2
import numpy as np

# 读取图像并设置成相同尺寸（640 * 480）
def resize():
    img1 = cv2.imread("../images/map1.png")
    img2 = cv2.imread("../images/map2.png")
    imgSize = (640,480)
    img1 = cv2.resize(img1,imgSize)
    img2 = cv2.resize(img2,imgSize)
    return img1,img2

# 获取两张图像单应性矩阵
def get_homo(img1,img2):
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    # 创建特征转换对象
    sift = cv2.SIFT_create()
    # 通过特征转换对象获得特征点和描述子
    k1,d1 = sift.detectAndCompute(gray1,None)
    k2,d2 = sift.detectAndCompute(gray2,None)
    # 创建特征匹配器
    bf = cv2.BFMatcher_create(normType=cv2.NORM_L1,crossCheck=False)
    # 进行特征匹配
    matchs = bf.knnMatch(d1, d2, k=2)
    # 过滤无效特征点找出有效的特征匹配点
    verify_ratio = 0.8
    verify_matchs = []
    for i,(m1,m2) in enumerate(matchs):
        # 有效的匹配点
        if m1.distance < verify_ratio * m2.distance:
            verify_matchs.append(m1)

    if len(verify_matchs) < 4:
        print("匹配到的有效特征点太少，不能计算单应性矩阵")
        return

    # 计算单应性矩阵
    img1_pts = []
    img2_pts = []
    for m in verify_matchs:
        img1_pts.append(k1[m.queryIdx].pt)
        img2_pts.append(k2[m.trainIdx].pt)

    img1_pts = np.float32(img1_pts).reshape(-1,1,2)
    img2_pts = np.float32(img2_pts).reshape(-1,1,2)
    # 获取单应性矩阵
    h,_ = cv2.findHomography(img1_pts,img2_pts,cv2.RANSAC,5.0)
    return h

# 根据单应性矩阵对图像进行变换，然后平移输出最终结果
def stitch_image(img1,img2,h):
    # 获取每张图像的4个角点
    # 获取原始图像的高宽
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    # 构建要搜索图的4个角点（[0,0]左上角坐标，[0,h1]左下角坐标，[w1,h1]右下角坐标，[w1,0]右上角坐标）
    img1_pts = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    img2_pts = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # 对图片进行变换（使用单应性矩阵对图像进行旋转，平移）
    img1_transform = cv2.perspectiveTransform(img1_pts,h)
    # axis=0表示横向拼接
    res_dst = np.concatenate((img2_pts,img1_transform),axis=0)
    # 获取最小值（axis=0表示取x轴的最小值）（ravel函数是将二维数组转换为一维数组）,减去0.5是为了向下取整
    [x_min,y_min] = np.int32(res_dst.min(axis=0).ravel() - 0.5)
    # 获取最大值（axis=0表示取x轴的最大值）（ravel函数是将二维数组转换为一维数组）,加上0.5是为了向上取整
    [x_max,y_max] = np.int32(res_dst.max(axis=0).ravel() + 0.5)
    # 平移的距离（因为最小x和y是负值，所以加上了减号，让其等于正数）
    trandform_dist = [-x_min,-y_min]
    # 使用线性代数原理进行平移，其实就是乘以一个起始坐标，就可以达到平移的效果（注意：平移坐标的斜对角都是1，如下面的数据）
    # 构建平移数据矩阵（第一个参数是平移x轴的距离，第二个参数是平移y轴的距离）
    treansform_array = np.array([[1,0,trandform_dist[0]],
                               [0,1,trandform_dist[1]],
                               [0,0,1]])
    # 透视变换（最大宽度减去最小宽度就等于实际宽度）（treansform_array.dot(h)就是用平移数据矩阵乘以已经计算好了的单应性矩阵，以达到图像平移的效果）
    res_img = cv2.warpPerspective(img1,treansform_array.dot(h),(x_max - x_min,y_max - y_min))
    # 将指定位置的像素设置成第张二图，已完成拼接效果
    res_img[trandform_dist[1]:trandform_dist[1]+h2,
            trandform_dist[0]:trandform_dist[0]+w2] = img2
    return res_img


if __name__ == "__main__":
    # 创建一个显示窗口,窗口的名字是窗口的唯一ID（cv2.WINDOW_NORMAL 创建的窗口可以改变大小）
    cv2.namedWindow("原图片", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("拼接图片", cv2.WINDOW_AUTOSIZE)
    # 第一步：读取图像并设置成相同尺寸（640 * 480）
    img1,img2 = resize()
    # 旧的两张图
    old_image = np.hstack((img1,img2))

    # 第二步：找特征点，描述子，计算单应性矩阵
    h = get_homo(img1,img2)

    # 第三步：根据单应性矩阵对图像进行变换，然后平移输出最终结果
    new_img = stitch_image(img1,img2,h)

    # 显示图像
    cv2.imshow("原图片",old_image)
    cv2.imshow("拼接图片", new_img)
    # 阻塞（按任意键退出）
    key = cv2.waitKey(-1)