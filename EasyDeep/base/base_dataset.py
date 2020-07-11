from torch.utils.data import Dataset


class BaseDataSet(Dataset):
    def __init__(self):
        super(BaseDataSet, self).__init__()
        self.load_config()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def config_load(self):
        raise NotImplementedError
