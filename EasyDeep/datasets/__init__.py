from datasets.image_dataset import ImageDataSet
def get_dataset(dataset_name):
    if dataset_name == "csv":
        from configs.dataset_config import CSVDataSetConfig
        dataset_config = CSVDataSetConfig()
        CsvDataSet(dataset_config)
    elif dataset_name=='image':
        from configs.dataset_config import ImageDataSetConfig
        dataset_config = ImageDataSetConfig
        ImageDataSet(dataset_config)
