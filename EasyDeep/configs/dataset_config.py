import os
import platform

from torchvision import transforms


class ImageDataSetConfig:
    def __init__(self):
        __system = platform.system()
        if __system == "Windows":
            self.image_path = r"C:\(lab\datasets\polyp\TMP\07\data"
            self.mask_path = r"C:\(lab\datasets\polyp\TMP\07\mask"
        elif __system == "Linux":
            self.image_path = "/home/straw/Downloads/dataset/polyp/TMP/07/data"
            self.mask_path = "/home/straw/Downloads/dataset/polyp/TMP/07/mask"

        self.shuffle = True
        self.image_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        self.image_transforms_4_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # dataloader
        self.random_state = 0
        self.batch_size = 2
        self.test_rate = 0.2
        num_samples = len(os.listdir(self.mask_path))
        self.num_test = int(num_samples * self.test_rate)
        self.num_train = num_samples - self.num_test
        self.valid_rate = 0.2
        self.batch_size4test = 8


class ImageDataSet4EdgeConfig:
    def __init__(self):
        __system = platform.system()
        if __system == "Windows":
            self.image_path = r"C:\(lab\datasets\polyp\TMP\07\data"
            self.mask_path = r"C:\(lab\datasets\polyp\TMP\07\mask"
            self.edge_path = r"C:\(lab\datasets\polyp\TMP\07\edge_bi"
            self.predict_path = r"C:\(lab\datasets\polyp\TMP\07\predict_step_one"
        elif __system == "Linux":
            # train step 1
            # self.image_path = "/home/straw/Downloads/dataset/polyp/TMP/07/data"
            # self.mask_path = "/home/straw/Downloads/dataset/polyp/TMP/07/mask"
            self.edge_path = "/home/straw/Downloads/dataset/polyp/TMP/07/edge_bi"
            self.image_path = "/home/straw/Downloads/dataset/polyp/TMP/07/data"
            self.predict_path = "/home/straw/Downloads/dataset/polyp/TMP/07/predict_step_one/"
            self.mask_path = "/home/straw/Downloads/dataset/polyp/TMP/07/mask"

        self.batch_size = 8
        self.shuffle = True
        self.image_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        self.edge_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        self.predict_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        self.image_transforms_4_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.random_state = 7
        self.test_rate = 0.2
        num_samples = len(os.listdir(self.mask_path))
        self.num_test = int(num_samples * self.test_rate)
        self.num_train = num_samples - self.num_test
        self.valid_rate = 0.2
        self.batch_size4test = 8


class CSVDataSetConfig:
    def __init__(self):
        self.xls_path = ""
        self.csv_dir = ""
        self.test_rate = 0.3
        self.valid_rate = 0.2

        self.shuffle = True
        self.random_state = 0
        self.batch_size = 1
        self.batch_size4test = 1


class MNISTDatasetConfig:
    def __init__(self):
        self.dataset_path = r"C:\(lab\datasets\mnist"
        self.shuffle = True
        self.batch_size = 1
        self.batch_size_4_test = 32
        self.valid_rate = 0.2
