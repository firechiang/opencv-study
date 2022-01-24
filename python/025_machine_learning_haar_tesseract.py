'''
机器学习之Haar级联法车牌识别以及使用Tesseract进行文字识别和提取（注意：该算法不是很准确，建议使用dnn算法进行图像分类以及识别，具体列子在 026_machine_learning_dnn.py 文件里面）
'''

import cv2
import numpy as np
import pytesseract


if __name__ == "__main__":
    # 创建Haar级联器对象（参数是导入已经学习好了的车牌识别文件（就是机器训练后的数据））
    haar = cv2.CascadeClassifier("../haarcascades/haarcascade_russian_plate_number.xml")
    # 导入图片并灰度化
    img = cv2.imread("../images/chinacar.jpeg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 调用 detectMultiScale 函数进行车牌识别
    # scaleFactor（识别时自动放大或缩小的倍数，以便找到车牌位置）
    # minNeighbors（车牌识别时车牌像素最小值）
    # 返回值格式[[x,y,w,h],[x,y,w,h]]就是车牌所在位置以及宽高
    faces = haar.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

    for (x,y,w,h) in faces:
        # 将车牌框出来
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        # 将车牌部分图像抠出来（x轴取y+h个像素，y轴取x+w个像素）
        roi_img = gray[y:y+h,x:x+w]
        # 将车牌图像二值化
        ret,roi = cv2.threshold(roi_img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 提取文字（注意：以下识别不是很准确，最好是将识别图像进行形态学处理，以及去噪使图像更清晰，以便更好的识别）
        # lang（表示语言（chi_sim=中文，eng=英文），如果是多语言使用加号拼接）
        # config（表示配置信息 --psm 表示分页模式，--oem 表示引擎）（注意：具体使用方法，请在操作系统命令行执行命令 tesseract --help-extra 查看具体相关配置）
        plate_str = pytesseract.image_to_string(roi,lang='chi_sim+eng',config='--psm 8 --oem 3')
        print(f'识别出的车牌号是：{plate_str}')
        # 显示抠出来的车牌
        cv2.imshow('plate',roi)

    cv2.imshow("img",img)
    cv2.waitKey(-1)


