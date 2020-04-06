import cv2
import numpy as np

""""随机生成两个图像"""

flat_numpy_array = np.random.randint(0, 256, 120000)

gray_image = flat_numpy_array.reshape(300, 400)
cv2.imwrite('RandomGray.png', gray_image)

bgr_image = flat_numpy_array.reshape((100, 400, 3))
cv2.imwrite('RandomColor.png', bgr_image)

