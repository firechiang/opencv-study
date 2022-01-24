'''
深度学习模型（使用DNN网络对图像进行识别和分类）
1，DNN（深度神经网络）
2，RNN（循环神经网络，应用场景：语音识别，机器翻译，生成图像描述信息）
3，CNN（卷积神经网络，应用场景：图像分类，检索，目标定位，目标分割，人脸识别）

比较主流的深度学习框架
1，Tensorflow
2，Caffe -> caffle2 -> torch(pytorch)
3，MXNet
4，DarkNet（快速目标检测训练框架）

# 训练数据集（也就是用于训练的数据）
1，MNIST（Fashion-MNIST）用于手写字识别
2，VOC（举办挑战赛时的数据集，2012年以后就没有了，因为不在举办）
3，COCO（用于目标检测的大型数据集）
4，ImageNet

# 训练后模型文件
1，tensorflow 训练出来的模型数据是 .pb文件
2，pytorch 训练出来的模型数据是 .pth文件
3，caffe 训练出来的模型数据是 .caffe文件
4，ONNX开放性神经网络交换格式是 .onnx文件
'''

import cv2.dnn
import numpy as np

if __name__ == "__main__":
    # 导入训练后模型，得到深度神经网络
    # 导入Tensorflow的训练后模型
    #cv2.dnn.readNetFromTensorflow()
    # 导入Caffe的训练后模型
    net = cv2.dnn.readNetFromCaffe("../model/bvlc_googlenet.prototxt","../model/bvlc_googlenet.caffemodel")
    # 导入DarkNet的训练后模型
    #cv2.dnn.readNetFromDarknet()
    # 导入训练后模型（注意：这个函数会自动识别该模型是用那个框架训练出来的）
    #cv2.dnn.readNet()

    # 读取图像
    img = cv2.imread("../images/smallcat.jpeg")

    # 将图像转成张量
    # scalefactor（图像缩放倍数）
    # size（图像尺寸）
    # mean（平均差值，目的是消除图像中关照的变化，以方便图像分析；对于ImageNet的数据集一般取值是 103，116，123）（注意：该值参考训练模型所提供的值）
    # swapRB（R与B是否进行交换（默认不交换）深度学习一般是RGB，而OpenCV是BGR）
    # crop（对图像是否进行裁剪（默认不裁剪））
    blob = cv2.dnn.blobFromImage(image=img,scalefactor=1.0,size=(224,224),mean=(104,117,123))
    # 将张量数据送入深度神经网络
    net.setInput(blob)
    # 预测结果（注意：该结果是分类后结果）
    r = net.forward()
    # 将预测结果做倒序排列（每一行的第一项进行倒序排列）（可能性最大的排第一，最小的排最后）
    order = sorted(r[0],reverse=True)

    classes = []
    # 读入类目（数据格式：类目，类目描述）
    with open("../model/synset_words.txt",'rt') as f:
        # 将文件中的每一行数据读取到classes数组当中
        classes = [x [x.find(" ")+1:] for x in f]

    # 分析结果
    # 数组最多3个元素
    z = list(range(3))
    # 将匹配最高的3项进行匹配
    for i in list(range(0,3)):
        # 如果相等，取出值放到z数组当中
        z[i] = np.where(r[0] == order[i])[0][0]
        # 打印结果
        print(f'第{i+1}项，匹配:{classes[z[i]]}',end='')
        print(f'类所在行{z[i]+1}，可能性{order[i]}')