from PIL import Image


def getImage(path="sample.jpg", mode="RGB"):
    """ get a image sample"""
    if mode is None:
        return Image.open(path)
    else:
        return Image.open(path).convert(mode)
