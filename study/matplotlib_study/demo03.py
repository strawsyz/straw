from matplotlib import font_manager
from matplotlib import pyplot as plt

font = font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')

y_1 = [1, 0, 1, 1, 2, 4, 3, 2, 3, 4, 4, 5, 6, 5, 4, 3, 3, 1, 1, 1]
y_2 = [1, 0, 3, 1, 2, 2, 3, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]

x = range(11, 31)

plt.figure(figsize=(20, 8), dpi=80)

plt.plot(x, y_1, label='小明', color='#F08080')
plt.plot(x, y_2, label="小红", color="#DB7093", linestyle="--")

_xtick_labels = ["{}岁".format(i) for i in x]
plt.xticks(x, _xtick_labels, fontproperties=font)

plt.grid(alpha=0.4, linestyle=':')

plt.legend(prop=font, loc='upper left')

plt.show()
