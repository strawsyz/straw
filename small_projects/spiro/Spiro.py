import turtle
import PIL
import math


# from fractions import gcd
class Spiro:
    def __init__(self, xc, yc, col, R, r, l):
        # 创建新的实例
        self.t = turtle.Turtle()
        # 设置鼠标样式为 海龟
        self.t.shape('turtle')
        # 将参数绘图角度的增量设置为5度
        self.step = 5
        self.drawingComplete = False

        # 设置参数
        self.setparams(xc, yc, col, R, r, l)

        # 初始化图像
        self.restart()

    def setparams(self, xc, yc, col, R, r, l):
        self.xc = xc
        self.yc = yc
        self.R = int(R)
        self.r = int(r)
        self.l = l
        self.col = col

        #  gcd函数返回两个参数的最大公约数
        gcdVal = math.gcd(self.r, self.R)
        self.nRot = self.r // gcdVal
        self.k = r / float(R)
        #  设置颜色
        self.t.color(*col)
        #  保存当前的角度
        self.a = 0

    def restart(self):
        self.drawingComplete = False
        self.t.showturtle()  # 显示海龟光标
        self.t.up()  # 提起海龟，让海龟在down之前无法画线
        R, k, l = self.R, self.k, self.l
        a = 0.0
        x = R * ((1 - k) * math.cos(a) + math.cos((1 - k) * a / k))
        y = R * ((1 - k) * math.cos(a) - math.cos((1 - k) * a / k))
        self.t.setpos(self.xc + x, self.yc + y)
        self.t.down()

    #  绘制曲线
    def draw(self):
        R, k, l = self.R, self.k, self.l
        for i in range(0, 360 * self.nRot + 1, self.nRot):
            a = math.radians(i)  # 度转弧度
            x = R * ((1 - k) * math.cos(a) + l * k * math.cos((1 - k) * a / k))
            y = R * ((1 - k) * math.sin(a) - l * k * math.sin((1 - k) * a / k))
            self.t.setpos(self.xc + x, self.yc + y)
        self.t.hideturtle()  # 藏起海龟图标

    # 创建动画
    def update(self):
        if self.drawingComplete:
            return
        self.a += self.step

        R, k, l = self.R, self.k, self.l
        a = math.radians(self.a)
        x = self.R * ((1 - k) * math.cos(a) + l * k * math.cos((1 - k) * a / k))
        y = self.R * ((1 - k) * math.sin(a) - l * k * math.sin((1 - k) * a / k))
        self.t.setpos(self.xc + x, self.yc + y)
        if self.a >= 360 * self.nRot:
            self.drawingComplete = True
            self.t.hideturtle()

    def clear(self):
        self.t.clear()
