import matplotlib.pyplot as plt


def lineplot(y_list: list, x_list: list = None, labels=None, title="broken line graph", style_list=None, axis='on'):
    if x_list is None:
        x_list = [[x for x in range(len(y_list[0]))] for _ in range(len(y_list))]
    assert len(x_list) == len(y_list)

    plt.figure()
    if labels is None:
        labels = [x for x in range(len(x_list))]
    for i in range(len(x_list)):
        if style_list is None:
            plt.plot(x_list[i], y_list[i], label=labels[i])
        else:
            plt.plot(x_list[i], y_list[i], style_list[i], label=labels[i])
    plt.legend()
    plt.title(title)
    plt.axis(axis)


def boxplot(x_labels, data_list, base_color="#539caf", median_color="#297083", x_label="", y_label="", title=""):
    _, ax = plt.subplots()

    ax.boxplot(data_list
               , patch_artist=True
               , medianprops={'color': median_color}
               , boxprops={'color': base_color, 'facecolor': base_color}
               , whiskerprops={'color': base_color}
               , capprops={'color': base_color})

    ax.set_xticklabels(x_labels)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)


def show():
    plt.show()


if __name__ == '__main__':
    from datasets.csv_dataset import CsvDataSet
    from configs.dataset_config import CSVDataSetConfig

    dataset = CsvDataSet(config_instance=CSVDataSetConfig())
    dataset.get_dataloader(dataset)
    lineplot([dataset.hm_delay, dataset.hm_early, dataset.wr / 10],
             labels=["hm_delay", "hm_early", "wr"], title="analysis")

    # boxplot(["hm_delay", "hm_early", "wr/10"],  [dataset.hm_delay.squeeze(), dataset.hm_early.squeeze(), dataset.wr.squeeze() / 10])

    # x_data = [range(len(dataset.X[0]))]
    # y_data = dataset.X[0].reshape(8, 5000)
    # print(y_data[0], y_data[1])
    # lineplot(y_data)
    show()