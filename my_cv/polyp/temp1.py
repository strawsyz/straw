import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()
# x = np.linspace(0, 255, 255)
x = range(0, 256, 1)
y = np.linspace(0, 256, 256)
plt.plot(x, y, label="123")
plt.xlabel("threshold")
plt.ylabel("calcu_iou")
plt.title("threshold-calcu_iou")
plt.legend()
plt.show()
