class HistoryRecorder:
    def __init__(self, train_loss, valid_loss):
        self.train_loss = train_loss
        self.valid_loss = valid_loss


def main():
    recorder = HistoryRecorder


if __name__ == '__main__':
    main()
