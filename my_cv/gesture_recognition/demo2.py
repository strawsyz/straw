# Python 3.6.5
# Skimage Library
from skimage import io
import matplotlib.pyplot as plt

num = 0
list = ['res.png', 'test.jpg']



def show():
    global num
    for i in range(2):
        img = io.imread(list[i])
        io.imshow(img)
        plt.ion()
        # plt.pause(0.01)
        # input("Press Enter to Continue")  # 之后改成识别输出
        plt.ginput()

    plt.ioff()
if __name__ == '__main__':
    show()
