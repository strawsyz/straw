import datetime
import sys, random, argparse
import numpy as np
import math
import turtle
import random
import time
from PIL import Image

import Spiro
import SpiroAnimator

# 保存图片的功能还不能用，t,空格快捷键也不能用，无法进入函数
'''用turtle画出好看的曲线，
用来熟悉一些库'''


# 将图片保存为PNG图像文件
def saveDrawing():
    turtle.hideturtle()
    dateStr = (datetime.datetime.now()).strftime("%d%b%Y-%H%M%S")
    fileName = 'spiro-' + dateStr
    print('saving drawing to ' + fileName)
    canvas = turtle.getcanvas()
    canvas.postscript(file=fileName + '.eps')  # 将窗口保存为EPS文件格式，EPS是矢量格式
    time.sleep(1)
    print(fileName)
    img = Image.open(fileName + '.eps')  # 用Pillow打开EPS文件
    img.save(fileName + '.png', 'png')  # 保存为PNG文件


if __name__ == '__main__':
    # use sys.argv if needed
    print('generating spirograph...')
    # create parser
    descStr = """This program draws Spirographs using the Turtle module.
    When run with no arguments, this program draws random Spirographs.
    Terminology:
    R: radius of outer circle
    r: radius of inner circle
    l: ratio of hole distance to r
    """
    parser = argparse.ArgumentParser(description=descStr)  # 创建解析器对象
    parser.add_argument('--sparams', nargs=3, dest='sparams', required=False,
                        help='The three agruments in sparams:R, r, l.')
    args = parser.parse_args()  # 调用函数惊醒实际的解析

    turtle.setup(width=0.8)  # 设置宽度为整个屏幕宽度的80%

    turtle.shape('turtle')  # 设置光标形状为海龟

    turtle.title("straw'Spirographs")  # 设置窗口标题
    turtle.onkey(saveDrawing, "s")  # 设置事件，按下s时保存图片
    turtle.listen()  # 让窗口监听用户事件

    turtle.hideturtle()

    if args.sparams:
        params = [float(x) for x in args.sparams]
        col = [0.0, 0.0, 0.0]
        spiro = Spiro.Spiro(0, 0, col, *params)
        spiro.draw()
    else:
        spiroAnim = SpiroAnimator.SpiroAnimator(10)
        turtle.onkey(spiroAnim.toggleTurtles, "t")
        turtle.onkey(spiroAnim.restart, "space")

    turtle.mainloop()  # 让tkinter窗口保持打开，监听事件
