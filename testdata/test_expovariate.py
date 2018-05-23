# -*- coding: utf-8 -*-
"""
create on 2018-5-22 20:59:44
by：weifeng

"""
###测试指数分布

import numpy as np
import matplotlib.pyplot as plt  ##使用matplot的子模块
import random

x = np.linspace(0, 10, 100)
exp1 = [random.expovariate(1) for i in range(0,100)]
exp10 = [random.expovariate(10) for i in range(0,100)]
sum1 = 0
for i in exp1:
    sum1 += i
print(sum1/100)
sum10 = 0
for i in exp10:
    sum10 += i
print(sum10/100)


plt.plot(x, exp1, color = "blue",label="exp1")  # label表示的是线型标志
plt.plot(x, exp10, color="red", label="exp10")
# plt.xlim(-5,15)  #表示的是范围的调节
# plt.ylim(0,1.5)  #表示的是范围的调节
plt.axis([-1, 11, 0, 10])  # x,y轴范围的调节
plt.xlabel("x axis")  # 每个轴添加文本
plt.ylabel("y value")  # 每个轴添加文本
plt.legend()  # 表示添加上label的图示，是和参数label对应的
plt.title("machine learning")  # 添加标题
plt.show()