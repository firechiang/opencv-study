'''
传统车辆检测（不是很准确）
1，加载视频
2，通过形态学识别车辆
3，对车辆进行统计
4，显示车辆统计信息

视频中动态物体检测论文：https://link.springer.com/chapter/10.1007/978-1-4615-0913-4_11
'''

import cv2
import numpy as np

# 去噪点
def de_noise(frame):
    # 灰度化
    cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # 去噪（高斯算法）
    blur = cv2.GaussianBlur(frame,(3,3),5)
    return blur

# 去背景（去除静态物品）
def de_background(frame,bgsubmog):
    # 处理图像
    mask = bgsubmog.apply(frame)
    # 返回处理结果
    return mask

# 腐蚀（去掉一些小斑点）
def corrosion(frame,kernel):
    # iterations=表示腐蚀次数
    val = cv2.erode(frame,kernel,iterations=2)
    return val

# 膨胀（还原车辆原来大小）
def expand(frame,kernel):
    # iterations=表示膨胀次数
    dilate = cv2.dilate(frame,kernel,iterations=2)
    return dilate

# 计算轮廓中心点
def center(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx,cy

# 加载视频
def load_video():
    # 车辆检测分界线高
    lin_height = 550
    # 车辆检测分界线偏移量
    offset = 7
    # 过车辆检测分界线车辆计数
    count = 0
    # 创建一个显示窗口,窗口的名字是窗口的唯一ID（cv2.WINDOW_NORMAL 创建的窗口可以改变大小）
    cv2.namedWindow("video",cv2.WINDOW_AUTOSIZE)
    # 加载视频
    cap = cv2.VideoCapture("../video/vehicle.mp4")
    # 创建BackgroundSubtractorMOG对象，用于去除静态物品（history（表示以多少帧为基准来判断物体是否是动态物体，数值越大计算越精准），detectShadows（是否检测阴影））
    bgsubmog = cv2.createBackgroundSubtractorMOG2()
    # 该算法是对MOG算法的改进，而且噪点更低
    #bgsubmog = cv2.createBackgroundSubtractorKNN()
    #bgsubmog = cv2.bgsegm.createBackgroundSubtractorMOG(history = None)
    # 获取卷积核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    while True:
        ret,frame = cap.read()
        if(ret == True):
            # 去噪点
            blur = de_noise(frame)
            # 去背景（去除静态物品）
            mask = de_background(blur,bgsubmog)
            # 腐蚀（去掉一些小斑点）
            val = corrosion(mask, kernel)
            # 膨胀（还原车辆原来大小）
            dilate = expand(val, kernel)
            # 对图像做闭运算（去除车辆内的小斑点）
            closed = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
            # 获取车辆的轮廓
            conts,h = cv2.findContours(closed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            # 画车辆检测分界线
            cv2.rectangle(frame,(0,lin_height),(1300,lin_height),(0,0,255),1)
            # 遍历获取到的轮廓
            for(i,c) in enumerate(conts):
                # 查找最大外接矩形（参数传一个轮廓数据，它会自动找到最大外接矩形）
                (x,y,w,h) = cv2.boundingRect(c)
                # 轮廓宽度大于等于50并且轮廓高度大于等于50才算有效车辆轮廓
                if(w >= 50 and h >= 50):
                    # 将轮廓画到视频帧上
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
                    # 车辆轮廓中心点
                    cx,cy = center(x, y, w, h)
                    # 统计车辆是否过检测分界线
                    if((cy > lin_height - offset) and (cy < lin_height + offset)):
                        count+=1

            # 画一个车辆统计计数
            cv2.putText(frame,f'Vehicle Count: {count}',(400,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
            # 显示帧
            cv2.imshow("video",frame)
        key = cv2.waitKey(10)
        # 如果按了Esc键或读取视频失败退出循环
        if(key == 27 or ret == False):
            break

    # 释放所有资源
    cap.release()
    cv2.destroyAllWindows()
    exit(0)


if __name__ == "__main__":
    # 加载显示视频
    load_video()