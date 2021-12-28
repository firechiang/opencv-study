'''
numpy 矩阵数组简单操作
'''
import numpy as np

if __name__ == "__main__":
    a = np.array([1,2,3])
    b = np.array([[1,2,3],[4,5,6]])
    #print(a)
    #print(b)

    # 行的个数2，列的个数2，通道数/层数3，每一个元素的类型是uint8值是0
    c = np.zeros((2,2,3),np.uint8)
    #print(c)

    # 行的个数2，列的个数2，通道数/层数3，每一个元素的类型是uint8值是1
    d = np.ones((2,2,3),np.uint8)
    #print(d)

    # 行的个数2，列的个数2，每一个元素的类型是uint8值是255（注意：255是我们自己顺便填的）
    f = np.full((2,2),255,np.uint8)
    #print(f)

    # 单位矩阵，正方形（就是斜对角都是1）
    g = np.identity(3)
    #print(g)

    # 单位矩阵，可以是长方形（传两个参数就是长方形，一个参数就是正方行）
    h = np.eye(3,5)
    print(h)