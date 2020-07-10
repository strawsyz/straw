from matplotlib import font_manager
from matplotlib import pyplot as plt

'''散点图'''
font = font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
y_3 = [11, 17, 16, 11, 12, 11, 12, 6, 6, 7, 8, 9, 12, 15, 14, 17, 18, 21, 16, 17, 20, 14, 15, 15, 15, 19, 21, 22, 22,
       22, 23]
y_10 = [26, 26, 28, 19, 21, 17, 16, 19, 18, 20, 20, 19, 22, 23, 17, 20, 21, 20, 22, 15, 11, 15, 5, 13, 17, 10, 11, 13,
        12, 13, 6]

x_3 = range(1, 32)
x_10 = range(51, 82)

plt.figure(figsize=(20, 8), dpi=80)

plt.scatter(x_3, y_3)
plt.scatter(x_10, y_10)


_x = list(x_3) + list(x_10)
_xtick_labels = ['3月{}日'.format(i) for i in x_3]
_xtick_labels += ['10月{}日'.format(i-50) for i in x_10]

plt.xticks(_x[::3], _xtick_labels[::3], fontproperties=font, rotation=30)

plt.legend(prop=font)

plt.xlabel('时间', fontproperties=font)
plt.ylabel('温度', fontproperties=font)
plt.title('图表名', fontproperties=font)

plt.show()

