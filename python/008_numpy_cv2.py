'''
numpy 矩阵数组的检索和赋值
'''
import numpy as np
import cv2

if __name__ == "__main__":
    # 创建像素图像，行的个数480，列的个数640，通道数/层数3，每一个元素的类型是uint8值是0（注意：0就是黑色）
    # 可以想象成创建3副画，然后跌在一起。也就是一张画有三层
    img = np.zeros((480,640,3),np.uint8)

    count = 0
    while count < 200:
        # 改变像素位置的值，让其等于255，也就是白色（注意：count是Y轴，100是X轴，1表示只修改BGR 3层的第2层G的值，这个参数也可以不填）
        # 最后的结果是会在图像上画一条竖线
        img[count,100,1] = 255
        # 改变像素位置的值（注意：count是Y轴，100是X轴。第一层赋值0，第二层赋值0，第三层赋值255）
        #img[count, 100] = [0,0,255]
        count = count + 1

    # 显示图像
    cv2.imshow("img",img)
    key = cv2.waitKey(0)
    if key & 0xFF == ord("q"):
        cv2.destroyAllWindows()