import os

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
if __name__ == '__main__':
    dir1 = ""
    for file in os.listdir(dir1):
        path = os.path.join(dir1, file)
        img = Image.open(path)
        image_transforms.transforms()
