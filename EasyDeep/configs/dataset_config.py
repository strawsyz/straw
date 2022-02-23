import os

from torchvision import transforms

from Args.video_args import get_args
from base.base_config import BaseDataSetConfig


class ImageDataSetConfig(BaseDataSetConfig):
    def __init__(self):
        super(ImageDataSetConfig, self).__init__()
        if self.system == "Windows":
            self.image_path = r"C:\(lab\datasets\polyp\TMP\07\data"
            self.mask_path = r"C:\(lab\datasets\polyp\TMP\07\mask"
        elif self.system == "Linux":
            self.image_path = r"/home/shi/Downloads/dataset/polyp/TMP/07/data"
            self.mask_path = r"/home/shi/Downloads/dataset/polyp/TMP/07/mask"

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


class ImageCsvDatasetConfig(BaseDataSetConfig):
    def __init__(self):
        super(ImageCsvDatasetConfig, self).__init__()
        self.train_path = None
        self.test_path = None
        self.train_csv_filepath = None
        self.test_csv_filepath = None
        # k_folder
        self.train_part = None
        self.valid_part = None
        self.shuffle = True
        self.train_transforms = None
        self.valid_transforms = None
        self.test_transforms = None

        # dataloader
        self.random_state = None
        self.train_batch_size = None
        self.valid_batch_size = None
        self.test_batch_size = None

        num_samples = len(os.listdir(self.mask_path))
        self.num_test = int(num_samples * self.test_rate)
        self.num_train = num_samples - self.num_test
        self.batch_size4test = 8


class MultiLabelDatasetConfig(BaseDataSetConfig):
    def __init__(self):
        super(MultiLabelDatasetConfig, self).__init__()
        self.train_path = r""
        self.test_path = r""
        self.shuffle = True
        self.batch_size = 4
        self.batch_size_4_test = 64
        self.valid_rate = 0.2
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
        self.random_state = 0


class VideoFeatureDatasetConfig(BaseDataSetConfig):
    def __init__(self):
        self.random_state = 0
        super(VideoFeatureDatasetConfig, self).__init__()
        # self.dataset_path = r""
        self.batch_size = 8
        self.batch_size_4_test = 32
        self.split_num = 3
        self.FPS = 4
        args = get_args()
        self.clip_length = args.clip_length
        self.FPS = args.FPS

        # self.valid_rate = 0.2
        # self.train_num = 100
        # self.valid_num = 20
        # self.test_num = 20
        # self.set_dataset_num = True
        self.use_rate = 1.0

        import socket

        host_name = socket.gethostname()
        if host_name == "26d814d5923d":
            self.use_rate = 1.0
            self.root_path = r"/workspace/datasets/UCF101"
        else:
            self.root_path = r"C:\(lab\datasets\UCF101"

        self.label_filepath = os.path.join(self.root_path,
                                           r"UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt")
        # self.train_dataset_root_path = os.path.join(self.root_path, r"features/train")
        if self.FPS == 4:
            self.train_dataset_root_path = os.path.join(self.root_path, r"features/train-4FPS-ResNet152")
        elif self.FPS == 2:
            self.train_dataset_root_path = os.path.join(self.root_path, r"features/train")
        else:
            raise NotImplementedError("No such FPS")
        self.test_dataset_root_path = self.train_dataset_root_path

        if self.split_num == 1:
            self.test_annotation_filepath = os.path.join(self.root_path,
                                                         r"UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist01.txt")
            self.train_annotation_filepath = os.path.join(self.root_path,
                                                          r"UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist01.txt")
        elif self.split_num == 2:
            self.test_annotation_filepath = os.path.join(self.root_path,
                                                         r"UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist02.txt")
            self.train_annotation_filepath = os.path.join(self.root_path,
                                                          r"UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist02.txt")
        elif self.split_num == 3:
            self.test_annotation_filepath = os.path.join(self.root_path,
                                                         r"UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist03.txt")
            self.train_annotation_filepath = os.path.join(self.root_path,
                                                          r"UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist03.txt")
