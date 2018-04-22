import matplotlib.pyplot as plt
import numpy as np

# 使用np.linspace定义x：范围是(-1,1);个数是50. 仿真一维数据组(x ,y)表示曲线1.
x=np.linspace(-1,1,50)
y1=2*x+1
y2 = x**2

#散点图
n=1024
X=np.random.normal(0,1,n) #散点的随机XY值
Y=np.random.normal(0,1,n)
T=np.arctan2(Y,X)  #计算颜色
plt.scatter(X,Y,s=75,c=T,alpha=0.5)



# 使用plt.figure定义一个图像窗口. 使用plt.plot画(x ,y)曲线. 使用plt.show显示图像./''
#plot(x,y)  第一个是X轴  第二个是Y轴
plt.figure()
plt.plot(x, y1)
#plt.show()#展示

plt.figure(num=3, figsize=(8, 5),)
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
plt.show()

'''
使用plt.figure定义一个图像窗口：
编号为3；大小为(8, 5). 使用plt.plot画(x ,y2)曲线. 
使用plt.plot画(x ,y1)曲线，曲线的颜色属性(color)为红色;
曲线的宽度(linewidth)为1.0；曲线的类型(linestyle)为虚线. 
使用plt.show显示图像.
'''