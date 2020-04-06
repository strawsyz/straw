import numpy as np
from tensorflow.contrib.graph_editor import Transformer

def crop(image, bbox, x, y, length):
    x, y, bbox = x.astype(np.int), y.astype(np.int), bbox.astype(np.int)

    x_min, y_min, x_max, y_max = bbox
    w, h = x_max - x_min, y_max - y_min

    # Crop image to bbox
    image = image[y_min:y_min + h, x_min:x_min + w, :]

    # Crop joints and bbox
    x -= x_min
    y -= y_min
    bbox = np.array([0, 0, x_max - x_min, y_max - y_min])

    # Scale to desired size
    side_length = max(w, h)
    f_xy = float(length) / float(side_length)
    image, bbox, x, y = Transformer.scale(image, bbox, x, y, f_xy)

    # Pad
    new_w, new_h = image.shape[1], image.shape[0]
    cropped = np.zeros((length, length, image.shape[2]))

    dx = length - new_w
    dy = length - new_h
    x_min, y_min = int(dx / 2.), int(dy / 2.)
    x_max, y_max = x_min + new_w, y_min + new_h

    cropped[y_min:y_max, x_min:x_max, :] = image
    x += x_min
    y += y_min

    x = np.clip(x, x_min, x_max)
    y = np.clip(y, y_min, y_max)

    bbox += np.array([x_min, y_min, x_min, y_min])
    return cropped, bbox, x.astype(np.int), y.astype(np.int)


def scale(image, bbox, x, y, f_xy):
    (h, w, _) = image.shape
    h, w = int(h * f_xy), int(w * f_xy)
    from numpy import resize
    image = resize(image, (h, w), preserve_range=True, anti_aliasing=True, mode='constant').astype(np.uint8)

    x = x * f_xy
    y = y * f_xy
    bbox = bbox * f_xy

    x = np.clip(x, 0, w)
    y = np.clip(y, 0, h)

    return image, bbox, x, y


def flip(image, bbox, x, y):
    image = np.fliplr(image).copy()
    w = image.shape[1]
    x_min, y_min, x_max, y_max = bbox
    bbox = np.array([w - x_max, y_min, w - x_min, y_max])
    x = w - x
    x, y = Transformer.swap_joints(x, y)
    return image, bbox, x, y


def rotate(image, bbox, x, y, angle):
    # image - -(256, 256, 3)
    # bbox - -(4,)
    # x - -[126 129 124 117 107  99 128 107 108 105 137 155 122  99]
    # y - -[209 176 136 123 178 225  65  47  46  24  44  64  49  54]
    # angle - --8.165648811999333
    # center of image [128,128]
    o_x, o_y = (np.array(image.shape[:2][::-1]) - 1) / 2.
    width, height = image.shape[0], image.shape[1]
    x1 = x
    y1 = height - y
    o_x = o_x
    o_y = height - o_y
    image = rotate(image, angle, preserve_range=True).astype(np.uint8)
    r_x, r_y = o_x, o_y
    angle_rad = (np.pi * angle) / 180.0
    x = r_x + np.cos(angle_rad) * (x1 - o_x) - np.sin(angle_rad) * (y1 - o_y)
    y = r_y + np.sin(angle_rad) * (x1 - o_x) + np.cos(angle_rad) * (y1 - o_y)
    x = x
    y = height - y
    bbox[0] = r_x + np.cos(angle_rad) * (bbox[0] - o_x) + np.sin(angle_rad) * (bbox[1] - o_y)
    bbox[1] = r_y + -np.sin(angle_rad) * (bbox[0] - o_x) + np.cos(angle_rad) * (bbox[1] - o_y)
    bbox[2] = r_x + np.cos(angle_rad) * (bbox[2] - o_x) + np.sin(angle_rad) * (bbox[3] - o_y)
    bbox[3] = r_y + -np.sin(angle_rad) * (bbox[2] - o_x) + np.cos(angle_rad) * (bbox[3] - o_y)
    return image, bbox, x.astype(np.int), y.astype(np.int)
