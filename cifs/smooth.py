import numpy
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import csv
import statsmodels.api as sm
lowess = sm.nonparametric.lowess



magnitude_file_path = './TEST.csv'
magnitude_file_path1 = './TEST1.csv'
magnitude_pgd = []
data = np.genfromtxt(magnitude_file_path) #delimiter: 用于分隔的str
data1 = np.genfromtxt(magnitude_file_path1) #delimiter: 用于分隔的str
x = range(len(data))
result = lowess(data, x, frac=0.02, it=2, delta=0)
result1 = lowess(data1, x, frac=0.02, it=2, delta=0)
y = []
y1 = []
for i in range(512):
    y.append(result[i][1])
    y1.append(result1[i][1])

plt.figure()

plt.plot(y)
plt.plot(y1)
plt.grid()
plt.show()



