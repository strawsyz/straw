from  my_network.preprocess import *

def usage():
    img = cv2.imread("/home/straw/PycharmProjects/straw/my_cv/temp.jpg")
    w = img.shape[0]
    h = img.shape[1]
    # 交换通道的顺序
    img = img[:, :, (2, 1, 0)]
    img_new = np.zeros((w, h, 3))
    lab = np.zeros((w, h, 3))
    for i in range(w):
        for j in range(h):
            # 对每个像素都要进行运算
            Lab = rgb2lab(img[i, j])
            lab[i, j] = (Lab[0], Lab[1], Lab[2])
    from PIL import Image
    # lab空间的图片是无法直接使用Image显示的
    img = Image.fromarray(lab)
    img.show()
    cv2.imwrite("test1.jpg",img)
    # 从Lab通道转换会RGB通道
    for i in range(w):
        for j in range(h):
            rgb = Lab2RGB(lab[i, j])
            img_new[i, j] = (rgb[2], rgb[1], rgb[0])

    cv2.imwrite("test.jpg", img_new)
if __name__ == '__main__':
    usage()