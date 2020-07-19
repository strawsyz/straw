import platform

from torchvision import transforms


class ImageDataSetConfig:
    def __init__(self):
        __system = platform.system()
        if __system == "Windows":
            self.image_path = "D:\Download\datasets\polyp\\06\data"
            self.mask_path = "D:\Download\datasets\polyp\\06\mask"

        elif __system == "Linux":
            self.image_path = "/home/straw/Downloads/dataset/polyp/TMP/06/data"
            self.mask_path = "/home/straw/Downloads/dataset/polyp/TMP/06/mask"

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

    # def __str__(self):
    #     return "124"

    def __repr__(self):
        return "123"


if __name__ == '__main__':
    a = ImageDataSetConfig()
    print(a)


class CSVDataSetConfig:
    def __init__(self):
        self.xls_path = "ecg_list.xlsx"
        self.csv_dir = "csv"
        self.test_size = 0.3

        self.shuffle = True
        # dataloader
        self.random_state = 0
        self.batch_size = 2
        self.num_train = 600
        self.num_test = 140
        self.valid_rate = 0.5
        self.batch_size4test = 8
