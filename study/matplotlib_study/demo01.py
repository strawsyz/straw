from matplotlib import pyplot as plt

x = range(2, 13)

y = [12, 43, 12, 45, 65, 86, 47, 16, 95, 35, 23]

# 设置图片大小和像素
plt.figure(figsize=(20, 8),  dpi=80)
# 绘图
plt.plot(x, y)
# 设置x的刻度
_xtick_labels = [i/2 for i in range(3, 27)]
plt.xticks(_xtick_labels)
plt.yticks(range(min(y), max(y)+1))
# 保存图片到本地   也可以保存svg这类矢量图，不会失真
# plt.savefig('demo01-temp.png')
# 展示图片
plt.show()
