class BaseHistory:
    __slots__ = ["train_loss", "valid_loss"]

    def __init__(self, train_loss, valid_loss=None):
        self.train_loss = train_loss
        self.valid_loss = valid_loss

    def get_recoder_dict(self):
        recoder_dict = {}
        for attr in self.__slots__:
            recoder_dict[attr] = getattr(self, attr, None)
        return recoder_dict


def main():
    recorder = BaseHistory


if __name__ == '__main__':
    main()
