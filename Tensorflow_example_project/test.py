from skimage import io, transform  # skimage包主要用于图像数据的处理，在该实验当中，
# io模块主要图像数据的读取（imread）和输出（imshow）操作，transform模块主要用于改变图像的大小（resize函数）
import glob  # glob包主要用于查找符合特定规则的文件路径名，跟使用windows下的文件搜索差不多，查找文件只用到三个匹配符：”*”, “?”, “[]”。
# ”*”匹配0个或多个字符；”?”匹配单个字符；”[]”匹配指定范围内的字符，如：[0-9]匹配数字。
# 该实验中，glob主要用于返回所有匹配的文件路径列表。它只有一个参数pathname，定义了文件路径匹配规则，这里可以是绝对路径，也可以是相对路径。
import os  # os模块主要用于处理文件和目录，比如：获取当前目录下文件，删除制定文件，改变目录，查看文件大小等。
# 该案例中os主要用于列举当前目录下所有文件（listdir）和判断某一路径是否为目录（os.path.isdir）。
import tensorflow as tf  # tensorflow是目前业界最流行的深度学习框架，在图像，语音，文本，目标检测等领域都有深入的应用。是该实验的核心，主要用于定义占位符，定义变量，创建卷积神经网络模型

import numpy as np  # numpy是一个基于python的科学计算包，在该实验中主要用来处理数值运算，包括创建爱你等差数组，生成随机数组，聚合运算等。
import time  # time模块主要用于处理时间系列的数据，在该实验主要用于返回当前时间戳，计算脚本每个epoch运行所需要的时间。
img = io.imread('F:/datas/flower_photos/tulips\8908097235_c3e746d36e_n.jpg')
for i in range(len(img)):
    print(img[i])

