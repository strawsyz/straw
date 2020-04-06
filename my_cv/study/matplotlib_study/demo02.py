from matplotlib import pyplot as plt
import random
from matplotlib import font_manager

# 用于win7显示中文的问题
font = font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')

x = range(0, 120)
y = [random.randint(10, 25) for i in range(120)]

# 设置图片大小和像素
plt.figure(figsize=(20, 8),  dpi=80)
# 绘图
plt.plot(x, y)

# _x = x
_xtick_labels = ["10点{}分".format(i) for i in range(60)]
_xtick_labels += ["11点{}分".format(i) for i in range(60)]
# rotation修改旋转的度数
plt.xticks(x[::3], _xtick_labels[::3], rotation=45, fontproperties=font)

# 添加描述信息
plt.xlabel("时间", fontproperties=font)
plt.ylabel("温度 单位（℃）", fontproperties=font)
plt.title('10点到12点每分钟的气温变化情况', fontproperties=font)
# 展示图片
plt.show()
