from torch.utils.data import Dataset


class BaseDataSet(Dataset):
    def __init__(self):
        super(BaseDataSet, self).__init__()

    def __len__(self):
        return 123

    def __getitem__(self, index):
        pass

    def config_load(self):
        pass
