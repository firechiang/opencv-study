'''
图像修复
'''

import cv2
import numpy as np

if __name__ == "__main__":
    img = cv2.imread("../images/inpaint.png")
    # inpaintMask（是一张与原始图像尺寸一致的黑底白色残缺位置的图像）（参数 0 表示值是8位的）
    inpaint_mask = cv2.imread("../images/inpaint_mask.png",0)
    # inpaintRadius（修复破损位置半径）
    inpaint_radius = 5
    # flags（cv2.INPAINT_NS表示修复图像算法复杂，cv2.INPAINT_TELEA表示修复图像算法简单）
    flags = cv2.INPAINT_TELEA
    # 修复图像
    new = cv2.inpaint(img,inpaint_mask,inpaint_radius,flags)

    cv2.imshow("img",img)
    cv2.imshow("new",new)

    cv2.waitKey(-1)