'''
:@Author: Guod Wu
:@Date: 2024/4/26 20:06:01
:@LastEditors: Guod Wu
:@LastEditTime: 2024/4/26 20:06:01
:Description: 
:Copyright: Copyright (©)}) 2024 Guod Wu. All rights reserved.
'''

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

## 读取Excel文件
file_path = r'C:\Users\guod\Desktop\EI-CENTRO_WE.xlsx'
data = pd.read_excel(file_path, encoding='utf-8')

## 提取时间和加速度数据
t = np.array(data.iloc[:, 1])  # 第一列为时间
a = np.array(data.iloc[:, 2]) * 9.8  # 第二列为加速度，乘以9.8得到重力加速度

## 计算时间间隔
dx = t[1] - t[0]

## 加速度积分为速度和位移
v = scipy.integrate.cumtrapz(a, t, dx, initial=0)
d = scipy.integrate.cumtrapz(v, t, dx, initial=0)


## 绘制时间历程曲线


plt.subplot(1, 3, 1)
plt.plot(t, a)
plt.ylabel('加速度 (m/s^2)')
plt.xlabel('时间 (s)')
plt.xlim(left=0, right=t[-1])
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(t, v)
plt.ylabel('速度 (m/s)')
plt.xlabel('时间 (s)')
plt.xlim(left=0, right=t[-1])
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(t, d)
plt.ylabel('位移 (m)')
plt.xlabel('时间 (s)')
plt.xlim(left=0, right=t[-1])

plt.show()


# 持续时间代码

# 读取Excel文件
file_path = r'C:\Users\guod\Desktop\EI-CENTRO_NS.xlsx'
data = pd.read_excel(file_path, encoding='utf-8')

# 提取时间和加速度数据
t = data.iloc[:, 1]   #第一列为时间
a = data.iloc[:, 2] * 9.8 #第二列为加速度，乘以9.8得到重力加速度
#计算能量
energy = scipy.integrate.trapz(np.abs((a)^2), t, dx, initial=0)

#计算能量累积曲线
cumulative_energy = scipy.integrate.cumtrapz(np.abs((a)^2), t, dx, initial=0)

#计算能量达到95%和5%的阈值
threshold_95 = 0.95 * energy
threshold_5 =  0.05* energy

#找到能量达到95%和5%的时间点
index_95 = np.where(cumulative_energy >= threshold_95)[0]
index_5 = np.where(cumulative_energy >= threshold_5)[0]

#计算持续时间
duration = t[index_95] - t[index_5]

print('重要特征持续时间：', str(duration), '秒')
