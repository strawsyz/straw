import matplotlib.pyplot as plt


def plot(x_list, y_list, labels, title, style_list=None, axis='on'):
    assert len(x_list) == len(y_list)
    plt.figure()
    for i in range(len(x_list)):
        if style_list is None:
            plt.plot(x_list[i], y_list[i], label=labels[i])
        else:
            plt.plot(x_list[i], y_list[i], style_list[i], label=labels[i])
    plt.legend()
    plt.title(title)
    plt.axis(axis)
    plt.show()
