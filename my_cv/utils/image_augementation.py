from imgaug import augmenters as iaa
import imgaug as ia
import cv2
import os


def handle_function():
    return iaa.Sequential([
        # iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(1),  # horizontally flip 50% of the images
        iaa.Flipud(1),  # vertically flip 50% of the images
        iaa.CropToFixedSize(width=640, height=640),
        iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
    ])


def handle_images(source_path, target_path, handle_function):
    # print(np.__version__)
    def make_directory(dir_name):
        """判断文件夹是否存在，如果不存在就新建文件夹"""
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, mode=0o777)

    make_directory(target_path)
    seq = handle_function()
    imglist = []
    img_name_list = []
    for file in os.listdir(source_path):
        path = os.path.join(source_path, file)
        if not os.path.isdir(path):
            img_name_list.append(file)
            imglist.append(cv2.imread(path))

    images_aug = seq.augment_images(imglist)
    index = 0
    for i in img_name_list:
        image_path = os.path.join(target_path, i)
        cv2.imwrite(image_path, images_aug[index])
        index = index + 1



if __name__ == '__main__':
    seed = 1
    ia.seed(seed)
    source_path = '/home/straw/下载/dataset/LSTDataset/mask'
    target_path = '/home/straw/下载/dataset/AGU/LSTDataset/mask'
    handle_images(source_path, target_path, handle_function)
    ia.seed(seed)
    source_path = '/home/straw/下载/dataset/LSTDataset/train'
    target_path = '/home/straw/下载/dataset/AGU/LSTDataset/train'
    handle_images(source_path, target_path, handle_function)

    ia.seed(seed)
    source_path = '/home/straw/下载/dataset/LSTDataset/trainH2c'
    target_path = '/home/straw/下载/dataset/AGU/LSTDataset/trainH2c'
    handle_images(source_path, target_path, handle_function)



