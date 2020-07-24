class BaseHistory:
    __slots__ = ["train_loss", "valid_loss"]

    def __init__(self, train_loss, valid_loss=None):
        self.train_loss = train_loss
        self.valid_loss = valid_loss

    def get_record_dict(self):
        record_dict = {}
        for attr in self.__slots__:
            record_dict[attr] = getattr(self, attr, None)
        return record_dict


class CustomHistory:
    def __init__(self, attr_names: list = []):
        for attr_name in attr_names:
            setattr(self, attr_name, None)


def main():
    recorder = BaseHistory


if __name__ == '__main__':
    main()
