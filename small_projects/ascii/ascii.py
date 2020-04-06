import argparse

from PIL import Image
import numpy as np

'''
使用文字画出图片
需要设置参数 图片路径 
'''

# 70级灰度
gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^ `. "

# 10级灰度
gscale2 = "@%#*+=-:. "


# 计算每个小块的平均亮度
def getAverageL(image):
    im = np.array(image)
    w, h = im.shape
    # 用 numpy.reshape()先将维度为宽和高(w，h)的二维数组转换成扁平的一维，其长度
    # 是宽度乘以高度(w*h)。然后 numpy.average()调用对这些数组值求和并计算平均值。
    return np.average(im.reshape(w * h))


def covertImageToAscii(fileName, cols, scale, moreLevels):
    global gscale1, gscale2
    # Image.open()打开文件，convert()转成灰度图像
    # L表示 luminance，是图像亮度的单位
    image = Image.open(fileName).convert("L")

    W, H = image.size[0], image.size[1]
    print("input image dims: %d x %d" % (W, H))
    # 计算小块的宽度
    # 使用浮点而不是整数除法，避免在计算小块尺寸时的截断误差。
    w = W / cols
    # 计算小块的高度,scale是垂直比例
    h = w / scale
    # 计算行数
    rows = int(H / h)
    print("cols: %d, rows: %d" % (cols, rows))
    print("tile dims: %d x %d" % (w, h))
    if cols > W or rows > H:
        print("Image too small for specified cols!")
        exit(0)  # an ASCII image is a list of character strings

    # 从图像生成ascii内容
    aimg = []
    for j in range(rows):
        y1 = int(j * h)
        y2 = int((j + 1) * h)
        if j == rows - 1:
            y2 = H
        aimg.append("")
        for i in range(cols):
            x1 = int(i * w)
            x2 = int((i + 1) * w)
            if i == cols - 1:
                x2 = W
            # 取一个小块
            img = image.crop((x1, y1, x2, y2))
            # 获得小块的平均亮度
            avg = int(getAverageL(img))
            if moreLevels:
                gsval = gscale1[int((avg * 69) / 255)]
            else:
                gsval = gscale2[int((avg * 9) / 255)]
            aimg[j] += gsval
    return aimg


def main():
    # create parser
    descStr = "This program converts an image into ASCII art."
    parser = argparse.ArgumentParser(description=descStr)
    parser.add_argument('--file', dest='imgFile', required=False)
    parser.add_argument('--scale', dest='scale', required=False)
    parser.add_argument('--out', dest='outFile', required=False)
    parser.add_argument('--cols', dest='cols', required=False)
    parser.add_argument('--moreLevels', dest='moreLevels', action='store_true')

    # parse arguments
    args = parser.parse_args()

    imgFile = 'cover.png'
    if args.imgFile:
        imgFile = args.imgFile
        # set output file
    outFile = 'out.txt'
    if args.outFile:
        outFile = args.outFile
    # set scale default as 0.43, which suits a Courier font
    scale = 0.43
    if args.scale:
        scale = float(args.scale)
    # set cols
    cols = 80
    if args.cols:
        cols = int(args.cols)
    print('generating ASCII art...')
    # convert image to ASCII text
    aimg = covertImageToAscii(imgFile, cols, scale, args.moreLevels)
    f = open(outFile, 'w')
    for row in aimg:
        f.write(row + '\n')
    f.close()
    print("ASCII art written to %s" % outFile)


if __name__ == '__main__':
    main()
