from datasets.image_dataset import ImageDataSet
def get_dataset(dataset_name):
    from configs.dataset_config import ImageDataSetConfig
    dataset_config = ImageDataSetConfig
    ImageDataSet(dataset_config)
