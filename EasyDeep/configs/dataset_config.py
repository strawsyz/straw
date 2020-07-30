import platform

from torchvision import transforms


class ImageDataSetConfig:
    def __init__(self):
        __system = platform.system()
        if __system == "Windows":
            # self.image_path = "D:\Download\datasets\polyp\\06\data"
            # self.mask_path = "D:\Download\datasets\polyp\\06\mask"
            # tmp
            self.image_path = "C:\data_analysis\datasets\samples"
            self.mask_path = "C:\data_analysis\datasets\samples"
        elif __system == "Linux":
            self.image_path = "/home/straw/Downloads/dataset/polyp/TMP/07/data"
            self.mask_path = "/home/straw/Downloads/dataset/polyp/TMP/07/mask"

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
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        # dataloader
        self.random_state = 0
        self.batch_size = 2
        self.test_rate = 0.2
        import os
        num_samples = len(os.listdir(self.mask_path))
        self.num_test = int(num_samples * self.test_rate)
        self.num_train = num_samples - self.num_test
        self.valid_rate = 0.2
        self.batch_size4test = 8


class BaseDataSetConfig:
    def __init__(self):
        self.batch_size = 1
        self.batch_size4test = 1
        self.random_state = 0

        self.shuffle = True

        self.test_rate = 0.2
        self.valid_rate = 0.2


class CSVDataSetConfig:
    def __init__(self):
        self.xls_path = "C:\data_analysis\datasets\ecgs\ecg_list_dummy.xlsx"
        self.csv_dir = "C:\data_analysis\datasets\ecgs\csv"
        # self.xls_path = "D:\dataset\ecgs\ecg_list.xlsx"
        # self.csv_dir = "D:\dataset\ecgs\csv"
        self.test_rate = 0.3
        self.valid_rate = 0.2

        self.shuffle = True
        # dataloader
        self.random_state = 0
        self.batch_size = 1
        self.batch_size4test = 1


if __name__ == '__main__':
    a = ImageDataSetConfig()
    print(a)
