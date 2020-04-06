"""滑动窗口的函数"""


def sliding_window(image, step_size, window_size):
    """给定一副图像，返回一个从左向右滑动的窗口，直至覆盖整个图像"""
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
