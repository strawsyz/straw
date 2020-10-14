import os
import platform

from torchvision import transforms
from base.base_config import BaseDataSetConfig


class ImageDataSetConfig(BaseDataSetConfig):
    def __init__(self):
        super(ImageDataSetConfig, self).__init__()
        if self.system == "Windows":
            self.image_path = r"{image_path}}"
            self.mask_path = r"{mask_path}"
        elif self.system == "Linux":
            self.image_path = ""
            self.mask_path = ""

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


class ImageDataSet4EdgeConfig(BaseDataSetConfig):
    def __init__(self):
        super(ImageDataSet4EdgeConfig, self).__init__()
        # __system = platform.system()
        if self.system == "Windows":
            self.image_path = r""
            self.mask_path = r""
            self.edge_path = r""
            self.predict_path = r""
        elif self.system == "Linux":
            # train step 1
            self.edge_path = ""
            self.image_path = ""
            self.predict_path = ""
            self.mask_path = ""

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


class CSVDataSetConfig(BaseDataSetConfig):
    def __init__(self):
        super(CSVDataSetConfig, self).__init__()
        self.xls_path = ""
        self.csv_dir = ""
        self.test_rate = 0.3
        self.valid_rate = 0.2

        self.shuffle = True
        self.random_state = 0
        self.batch_size = 1
        self.batch_size4test = 1


class MNISTDatasetConfig(BaseDataSetConfig):
    def __init__(self):
        super(MNISTDatasetConfig, self).__init__()
        self.dataset_path = r""
        self.shuffle = True
        self.batch_size = 8
        self.batch_size_4_test = 32
        self.valid_rate = 0.2
        self.train_num = 100
        self.valid_num = 20
        self.test_num = 20
        self.set_dataset_num = True


class LoanDatasetConfig(BaseDataSetConfig):
    def __init__(self):
        super(LoanDatasetConfig, self).__init__()
        self.train_path = r""
        self.test_path = r""
        self.shuffle = True
        self.batch_size = 1
        self.batch_size_4_test = 64
        self.valid_rate = 0.9
        self.set_dataset_num = False
        self.feature_names = ['term', 'interest_rate', 'grade', 'credit_score']
        self.target_names = ["loan_status"]
        self.feature_target_names = self.feature_names + self.target_names
        self.preprocess = ["normalization"]
        # self.preprocess = ["normalization", "z_score"]
        self.train_num = None
        self.test_num = None
        self.test_data = None
        self.test_gt = None