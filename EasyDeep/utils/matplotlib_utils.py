import matplotlib.pyplot as plt


def lineplot(y_list: list, x_list: list = None, labels=None, title="broken line graph", style_list=None, axis='on',
             figsize=None):
    if x_list is None:
        x_list = [[x for x in range(len(y_list[0]))] for _ in range(len(y_list))]
    assert len(x_list) == len(y_list)
    if figsize is None:
        plt.figure()
    else:
        plt.figure(figsize=figsize)
    if labels is None:
        labels = [x for x in range(len(x_list))]
    for i in range(len(x_list)):
        if style_list is None:
            plt.plot(x_list[i], y_list[i], label=labels[i])
        else:
            plt.plot(x_list[i], y_list[i], style_list[i], label=labels[i])
    plt.legend()
    if title is not None:
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


def save(save_path):
    plt.savefig(save_path)


def save_ax(ax, figure, save_path, x_expand=1.1, y_expand=1.2):
    from utils.file_utils import create_unique_name
    save_path = create_unique_name(save_path)
    # Save just the portion _inside_ the second axis's boundaries
    extent = ax.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
    # Pad the saved area by 10% in the x-direction and 20% in the y-direction
    figure.savefig(save_path, bbox_inches=extent.expanded(x_expand, y_expand))
