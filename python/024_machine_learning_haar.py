'''
机器学习之Haar级联法人脸识别（注意：该算法不是很准确，建议使用dnn算法进行图像分类以及识别，具体列子在 026_machine_learning_dnn.py 文件里面）
'''

import cv2
import numpy as np

# 对人脸进行识别
def face():
    # 创建Haar级联器对象（参数是导入已经学习好了的人脸识别文件（就是机器训练后的数据））
    haar = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_default.xml")
    # 导入图片并灰度化
    img = cv2.imread("../images/p3.png")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 调用 detectMultiScale 函数进行人脸识别
    # scaleFactor（识别时自动放大或缩小的倍数，以便找到人脸位置）
    # minNeighbors（人脸识别时人脸像素最小值）
    # 返回值格式[[x,y,w,h],[x,y,w,h]]就是人脸所在位置以及宽高
    faces = haar.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    count = 0
    for (x,y,w,h) in faces:
        # 将人脸框出来
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        # 将人脸部分图像抠出来（x轴取y+h个像素，y轴取x+w个像素）
        roi_img = img[y:y+h,x:x+w]
        # 显示抠出来的人脸
        cv2.imshow(f'face_{count}',roi_img)
        count+=1

    cv2.imshow("img",img)
    cv2.waitKey(-1)


# 对眼鼻口进行识别
def facial():
    # 创建Haar级联器对象（参数是导入已经学习好了的眼睛识别文件（就是机器训练后的数据））
    haar = cv2.CascadeClassifier("../haarcascades/haarcascade_eye.xml")
    # 创建Haar级联器对象（参数是导入已经学习好了的口识别文件（就是机器训练后的数据））
    haar = cv2.CascadeClassifier("../haarcascades/haarcascade_mcs_mouth.xml")
    # 创建Haar级联器对象（参数是导入已经学习好了的鼻子识别文件（就是机器训练后的数据））
    haar = cv2.CascadeClassifier("../haarcascades/haarcascade_mcs_nose.xml")
    # 导入图片并灰度化
    img = cv2.imread("../images/p3.png")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 调用 detectMultiScale 函数进行人脸识别
    # scaleFactor（识别时自动放大或缩小的倍数，以便找到人脸位置）
    # minNeighbors（人脸识别时人脸像素最小值）
    # 返回值格式[[x,y,w,h],[x,y,w,h]]就是人脸所在位置以及宽高
    faces = haar.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

    for (x,y,w,h) in faces:
        # 将人脸框出来
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow("img",img)
    cv2.waitKey(-1)

if __name__ == "__main__":
    # 对人脸进行识别
    face()

    # 对眼鼻扣进行识别
    #facial()


