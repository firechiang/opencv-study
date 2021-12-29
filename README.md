#### Python环境搭建
```bash
# 全局安装 numpy（矩阵操作），matplotlib（图像显示），opencv_python（opencv核心库）
$ pip3 install numpy matplotlib opencv_python
```

#### OpenCV色彩空间
 - 计算机一般是三原色RGB，而OpenCV则是BGR，也就是排列顺序不一样（排列顺序不一样颜色显示就不一样）
 - 三原色RGB可以想象成创建3副颜色不同的纯色画，然后跌在一起，形成另一个颜色的画


#### OpenCV HSV说明
 - Hue: 色相，即色彩，入红色，蓝色
 - Saturation: 饱和度，颜色的纯度
 - Value: 明亮度

#### OpenCV HUE度数对应颜色说明
 - 0度（red）红色
 - 60度（yellow）黄色
 - 120度（green）绿色
 - 180（cyan）青色
 - 240度（blue）蓝色
 - 300度（magenta）洋红色

#### openCV Mat（矩阵）属性说明
|字段    |说明 |
|----   | ----|
dims    | 维度，一般就是2维数据
rows    | 行数
cols    | 列数
depth   | 像素的位深
channels| 通道数/层数（RGB就是3）
size    | 矩阵大小
type    | dep+dt+chs（比如 CV_8UC3，8就表示为“像素的位深” + U是数据类型也就是无符号的整型，C3表示数据，C就是char，3表示char的个数就是3）
data    | 存放数据

