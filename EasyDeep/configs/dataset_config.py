from torchvision import transforms


# 这边有的参数可以设置为None，但是不能删除
# 可以再配置一个配置文件
class ImageDataSetConfig:
    def __init__(self):
        self.image_path = "D:\Download\datasets\polyp\\05\data"
        self.image_path = "D:\Download\\back\艺术\莲华"
        self.mask_path = "D:\Download\datasets\polyp\\05\mask"
        self.mask_path = "D:\Download\\back\艺术\莲华"
        self.random_state = 0
        self.shuffle = True
        self.image_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(100),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            # 将输出设置为一个通道
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
