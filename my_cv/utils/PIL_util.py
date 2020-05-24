from PIL import Image


def show(path):
    Image.open(path).show()


def show_numpy_image(np_array):
    Image.fromarray(np_array).show()


def thumbnail(img, size):
    return img.thumbnail(size)


def copy(img, box):
    return img.crop(box)


def paste(img, region, point):
    img.paste(region, point)


def transpose(img, method=Image.ROTATE_90):
    # Image.ROTATE_90
    # Image.FLIP_TOP_BOTTOM
    # Image.FLIP_LEFT_RIGHT
    return img.transpose(method)


def save(img, path):
    img.save(path)


def resize(img, size):
    return img.resize(size)


def rotate(img, angle, resample=Image.NEAREST, expand=0, center=None):
    return img.rotate(angle, resample, expand, center)


def get_info(img):
    print(' format is {0}\n palette is {1}\n mode is {3}\n info is {3}'.format(
        img.format, img.palette, img.mode, img.info
    ))


# not testing
def seek(gif, num):
    return gif.seek(num)


def tell(gif):
    return gif.tell()


def show_gif(gif):
    from PIL import ImageSequence
    for i in ImageSequence.Iterator(gif):
        gif.seek(i)
        gif.show()

# if __name__ == '__main__':
#     img
