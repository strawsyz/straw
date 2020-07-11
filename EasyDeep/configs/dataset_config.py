from torchvision import transforms


class ImageDataSetConfig:
    def __init__(self):
        self.image_path = "D:\Download\datasets\sample"
        self.mask_path = "D:\Download\datasets\sample"
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
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        # dataloader
        self.random_state = 0
        self.batch_size = 2
        self.num_train = 600
        self.num_test = 140
        self.valid_rate = 0.5
        self.batch_size4test = 8
