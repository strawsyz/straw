class BaseHistory:
    def __init__(self, train_loss, valid_loss=None):
        self.train_loss = train_loss
        self.valid_loss = valid_loss


def main():
    recorder = BaseHistory


if __name__ == '__main__':
    main()
