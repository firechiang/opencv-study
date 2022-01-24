'''
图像分割（将前景物体从背景物体中分离出来）
1，传统的图像分割方法（1，分水岭，2，GrabCut法，3，MeanShift法，4，背景抠出）
2，基于深度学习的图像分割方法
'''


import cv2
import numpy as np
from matplotlib import pyplot as plt

# 传统的图像分割方法之GrabCut法实现（通过交互的方式获取前景物体（就是手动指定要分割图像的区域））
# 程序使用方法：首先鼠标选择要分割的区域，然后按G键进行分割
class App:
    startX = 0
    startY = 0
    rect = (0,0,0,0)
    flag_rect = False
    # 鼠标事件
    def on_mouse(self,event,x,y,flags,params):
        # 鼠标左键按下
        if event == cv2.EVENT_LBUTTONDOWN:
            self.startX = x
            self.startY = y
            self.flag_rect = True
        # 鼠标左键抬起
        if event == cv2.EVENT_LBUTTONUP:
            self.flag_rect = False
            # 绘制矩形
            cv2.rectangle(self.img,
                          (self.startX,self.startY),
                          (x,y),
                          (0,0,255),
                          3)
            # 选择区域
            self.rect = (min(self.startX,x),
                         min(self.startY,y),
                         abs(self.startX-x),
                         abs(self.startY-y))

        # 鼠标移动
        if event == cv2.EVENT_MOUSEMOVE:
            if self.flag_rect == True:
                # 拷贝新图用来绘制矩形
                self.img = self.img2.copy()
                # 绘制矩形
                cv2.rectangle(self.img,
                              (self.startX, self.startY),
                              (x, y),
                              (255, 0, 0),
                              3)

    def run(self):
        wind_name_input = "input"
        wind_name_output = "output"
        cv2.namedWindow(wind_name_input)
        cv2.namedWindow(wind_name_output)
        cv2.setMouseCallback(wind_name_input,self.on_mouse)
        self.img = cv2.imread("../images/nnn.png")
        self.img2 = self.img.copy()
        # 值为zeros就是0，第一个参数取img高度和宽度
        self.mask = np.zeros(self.img.shape[:2],dtype = np.uint8)
        self.output = np.zeros(self.img.shape,np.uint8)
        while True:
            cv2.imshow(wind_name_input, self.img)
            cv2.imshow(wind_name_output,self.output)
            key = cv2.waitKey(100)
            if key == 27:
                break
            if key == ord('g'):
                # 提取指定区域图像
                # mask（值为0（BGD）表示背景，值为1（FGD）表示前景，值为2（PR_BGD）表示可能是背景，值为3（PR_FGD）表示可能是前景）
                # rect（提取区域）
                # bgdModel背景模式（取64位的浮点值，范围一般是1-65）
                bgdModel = np.zeros((1,65),np.float64)
                # fgdModel前景模式（取64位的浮点值，范围一般是1-65）
                fgdModel = np.zeros((1, 65), np.float64)
                # iterCount（迭代次数）
                # mode（cv2.GC_INIT_WITH_RECT，第一次调用使用该模式（在指定区域内找前景），cv2.GC_INIT_WITH_MASK（第一次以后调用使用该模式））
                cv2.grabCut(img=self.img2,mask=self.mask,rect=self.rect,bgdModel=bgdModel,fgdModel=fgdModel,iterCount=1,mode=cv2.GC_INIT_WITH_RECT)
                # mask值为1或者为3，将值设置成255，最后将值的类型修改为uint8
                mask2 = np.where((self.mask == 1) | (self.mask == 3),255,0).astype('uint8')
                # 与运算
                self.output = cv2.bitwise_and(self.img2,self.img2,mask=mask2)


# 传统的图像分割方法之GrabCut法实现（通过交互的方式获取前景物体（就是手动指定要分割图像的区域））
if __name__ == "__main__":
    app = App()
    app.run()



