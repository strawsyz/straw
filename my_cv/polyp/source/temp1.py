import os

import torchvision.transforms.functional as functional
from PIL import Image
from torchvision import transforms

image_transforms = transforms.Compose([
    # 随机调整亮度，对比度，饱和度，色相
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(100),

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.482, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])


def rotate4_imglist(images):
    # 拿到角度的随机数。angle是一个-180到180之间的一个数
    angle = transforms.RandomRotation.get_params([-180, 180])
    # 对image和mask做相同的旋转操作，保证他们都旋转angle角度

    for ind, image in enumerate(images):
        images[ind] = image.rotate(angle)
    # mask = mask.rotate(angle)
    return images


def vflip_imagelist(images):
    # 50%的概率应用垂直，水平翻转。
    if random.random() > 0.5:
        for i, image in enumerate(images):
            images[i] = functional.vflip(image)
    return images


def hflip_imagelist(images):
    # 50%的概率应用垂直，水平翻转。
    if random.random() > 0.5:
        for i, image in enumerate(images):
            images[i] = functional.hflip(image)
    return images


mask_transforms = transforms.Compose([
    # 随机调整亮度，对比度，饱和度，色相
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(100),

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.482, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])
to_image = transforms.ToPILImage()

if __name__ == '__main__':
    import numpy as np
    import random

    np.random.seed(7)  # numpy
    random.seed(7)  # random and transforms
    dir1 = "D:\Download\datasets\polyp\\05\\target"
    dir2 = "D:\Download\datasets\polyp\\05\\target1"
    dir3 = "D:\Download\datasets\polyp\\05\\target2"
    dir4 = "D:\Download\datasets\polyp\\05\\target3"
    for file_name in os.listdir(dir1):
        source1_path = os.path.join(dir1, file_name)
        source2_path = os.path.join(dir2, file_name)
        target1_path = os.path.join(dir3, file_name)
        target2_path = os.path.join(dir4, file_name)
        source_image = Image.open(source1_path)
        source2_image = Image.open(source2_path)
        images = [source_image, source2_image]
        images = vflip_imagelist(images)
        images = hflip_imagelist(images)
        target1_image, target2_image = images[0], images[1]
        target1_image.save(target1_path)
        target2_image.save(target2_path)
    # dir1 = "D:\Download\datasets\polyp\\05\mask"
    # dir2 = "D:\Download\datasets\polyp\\05\data"
    # target_dir = "D:\Download\datasets\polyp\\05\\target"
    # target2_dir = "D:\Download\datasets\polyp\\05\\target1"
    # for file in os.listdir(dir1):
    #     path = os.path.join(dir1, file)
    #     target_path = os.path.join(target_dir, file)
    #     img = Image.open(path)
    #     img = image_transforms(img)
    #     img = to_image(img)
    #     img.save(target_path)
    # for file in os.listdir(dir2):
    #     path = os.path.join(dir2, file)
    #     target_path = os.path.join(target2_dir, file)
    #     img = Image.open(path)
    #     img = mask_transforms(img)
    #     img = to_image(img)
    #     img.save(target_path)
