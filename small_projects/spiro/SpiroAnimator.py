import datetime
import turtle
import Spiro
import random
from PIL import Image


class SpiroAnimator:
    def __init__(self, num):
        self.deltaT = 10  # 时间间隔
        self.width = turtle.window_width()
        self.height = turtle.window_height()
        self.spiros = []
        for i in range(num):
            rparams = self.genRandomParams()
            spiro = Spiro.Spiro(*rparams)  # 用*运算符将元祖转化成参数列表
            self.spiros.append(spiro)
            turtle.ontimer(self.update, self.deltaT)  # 每隔DeltaT毫秒就调用update函数

    def genRandomParams(self):
        width, height = self.width, self.height
        R = random.randint(50, min(width, height)//2)  # R设置为50至窗口的一般长度的随机整数
        r = random.randint(1*R//10, 9*R//10)  # r设置为10至R的90%之间
        l = random.uniform(0.1, 0.9)
        # 随机点的位置
        xc = random.randint(-width//2, width//2)
        yc = random.randint(-height//2, height//2)

        #随机颜色
        col = (random.random(),
               random.random(),
               random.random())
        return (xc, yc, col, R, r, l)

    def restart(self):
        for spiro in self.spiros:
            # 清除之前绘制的线
            spiro.clear()
            rparams = self.genRandomParams()
            spiro.setparams(*rparams)
            spiro.restart()

    def update(self):
        nComplete = 0  # 记录已完成的对象的数目
        for spiro in self.spiros:
            spiro.update()
            if spiro.drawingComplete:
                nComplete += 1
        if nComplete == len(self.spiros):  # 如果所有的线都画完了，就重新来
            self.restart()
        turtle.ontimer(self.update, self.deltaT)

    # 控制海龟光标的开关
    def toggleTurtles(self):
        print(12)
        for spiro in self.spiros:
            if spiro.t.isVisible():
                spiro.t.hideturtle()
            else:
                spiro.t.showturtle()


