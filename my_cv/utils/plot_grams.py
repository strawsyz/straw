from matplotlib import pyplot as plt
import numpy as np

# 绘制散点图
def scatterplot(x_data, y_data, size=10, title="", x_label="", y_label="", color="r", yscale_log=True):

    # Create new plot object
    _, ax = plt.subplot()

    # Plot the data, set the size(s)（点的大小）, color（点的颜色）
    #  and transparency (alpha) of points
    ax.scatter(x_data, y_data, s=size, color=color, alpha=0.75)

    if yscale_log:
        ax.set_yscale('log')

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

# 绘制折线图
def lineplot(x_data, y_data, x_label="", y_label="", title=""):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    ax.plot(x_data, y_data, lw=2, color='#231312', alpha=1)

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


# 绘制直方图
# ，' n_bins '参数控制我们需要多少个离散的箱子来制作我们的直方图。更多的箱子会给我们更好的信息，
# 但也可能引入噪音，让我们远离大局，另一方面，更少的箱子给我们一个更“鸟瞰”和一个更大的画面
# 发生了什么，但是没有更详细的细节。其次，“累积”参数是一个布尔值，它允许我们选择直方图是否是累积的。
# 这基本上是选择概率密度函数(PDF)或累积密度函数(CDF)。
def histogram(data, n_bins, cumulative=False, x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    ax.hist(data, n_bins=n_bins, cumulative=cumulative, color='#539caf')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

# 位置重叠的两个直方图
def overlaid_histogram(data1, data2, n_bins=0, data1_name="", data1_color="#539caf",
                       data2_name="", data2_color="#7663b0", x_label="", y_label="", title=""):
    max_nbins= 20
    data_range = [min(min(data1), min(data2)), max(max(data1), max(data2))]
    binwidth = (data_range[1] - data_range[0])/max_nbins

    if n_bins == 0:
        bins = np.arange(data_range[0], data_range[1]+binwidth, binwidth)
    else:
        bins = n_bins
    _, ax = plt.subplots()
    ax.hist(data1, bins=bins, color=data1_color, alpha=1, label=data1_name)
    ax.hist(data2, bins=bins, color=data2_color, alpha=0.6, label=data2_name)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc='best')

# 一般的条形图画法
def barplot(x_data, y_data, error_data, x_label="", y_label="", title=""):
    _, ax = plt.subplots()

    # Draw bars, position them in the center of
    # the tick mark(刻度线) on the x-axis
    ax.bar(x_data, y_data, color= '#539caf', align ='center')
    # Draw error bars to show standard deviation, set ls to 'none'
    # to remove line between points
    ax.errorbar(x_data, y_data, yerr=error_data, color='#297083', ls='none', lw=2, capthick=2)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)

# 分组条形图画法
def stackedbarplot(x_data, y_data_list, colors, y_data_names="", x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    # Draw bars, one category at a time
    for i in range(0, len(y_data_list)):
        if i == 0:
            ax.bar(x_data, y_data_list[i], color = colors[i], align = 'center', label = y_data_names[i])
        else:
            # For each category after the first, the bottom of the
            # bar will be the top of the last category
            ax.bar(x_data, y_data_list[i], colors=colors[i], bottom=y_data_list[i-1], align='center', label=y_data_names[i])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc='uppper right')

# 堆叠条形图
def groupedbarplot(x_data, y_data_list, colors, y_data_names="", x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    # Total width for all bars at one x location
    total_width = 0.8
    # Width of each individual(个别) bar
    ind_width = total_width/len(y_data_list)
    # This centers each cluster(簇) of bars about the x tick mark
    alteration = np.arange(-(total_width/2), total_width/2, ind_width)

    # Draw bars, one category at a time
    for i in range(0, len(y_data_list)):
        # Move the bar to the right on the x-axis so it doesn't
        # overlap(重叠，相交) with previously drawn ones
        ax.bar(x_data + alteration[i], y_data_list[i], color=colors[i], label=y_data_list[i], width=ind_width)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc='uppper right')

# 绘制线箱图
def boxplot(x_data, y_data, base_color="#539caf", median_color="#297083", x_label="", y_label="", title=""):
    _, ax = plt.subplots()

    # Draw boxplots, specifying desired style
    ax.boxplot(y_data
               # patch_artist must be True to control box fill
               , patch_artist = True
               # Properties of median line
               , medianprops = {'color': median_color}
               # Properties of box
               , boxprops = {'color': base_color, 'facecolor': base_color}
               # Properties of whiskers
               , whiskerprops = {'color': base_color}
               # Properties of whisker caps
               , capprops = {'color': base_color})

    # By default, the tick label starts at 1 and increments by 1 for
    # each box drawn. This sets the labels to the ones we want
    ax.set_xticklabels(x_data)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)


if __name__ == '__main__':
    x_data = np.arange(12)
    y_data = np.arange(12)
    print(x_data)
    scatterplot(x_data, y_data)
    plt.draw()