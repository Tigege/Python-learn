import numpy as np

# array=np.array([[1,2,3],[4,5,6]])
# print(array)
# print(array.ndim) #维度
# print(array.shape) #行数和列数
# print(array.size)  #元素个数

#指定数据dtype
a=np.array([2,23,12],dtype=np.int32)
print(a.dtype)

#创建全零
a=np.zeros((3,4))
print(a)
"""
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])
"""

#创建连续数组
a = np.arange(10,20,2) # 10-19 的数据，2步长
"""
array([10, 12, 14, 16, 18])
"""
#改变数据的形状
a = np.arange(12).reshape((3,4))    # 3行4列，0到11
"""
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
"""

#线段型数据
a = np.linspace(1,10,20)    # 开始端1，结束端10，且分割成20个数据，生成线段
"""
array([  1.        ,   1.47368421,   1.94736842,   2.42105263,
         2.89473684,   3.36842105,   3.84210526,   4.31578947,
         4.78947368,   5.26315789,   5.73684211,   6.21052632,
         6.68421053,   7.15789474,   7.63157895,   8.10526316,
         8.57894737,   9.05263158,   9.52631579,  10.        ])
"""
