import math
import turtle

#  画圆的函数 x,y为坐标，r为圆的半径
def drawCirleTurtle(x,y,r):
    # move to the start of circle
    turtle.up()
    turtle.setpos(x + r, y)
    turtle.down()

    # draw the circle
    for i in range(0, 361):
        print(i)
        a = math.radians(i)
        turtle.setpos(x + r*math.cos(a), y + r*math.sin(a))

drawCirleTurtle(100, 100, 50)
# 保持图形界面打开
turtle.mainloop()

