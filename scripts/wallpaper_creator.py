import numpy as np
from PIL import Image
from pylab import *


def create_wallpaper(wallpaper, position):
    position = position_map[position]
    if position == position_map['right']:
        wallpaper[height_start:height_end, width - component_width:width, :] = array(component)
    if position == position_map['left']:
        wallpaper[height_start:height_end:, :component_width, :] = array(component)
    if position == position_map['top']:
        wallpaper[:component_height, width_start: width_end, :] = array(component)
    if position == position_map['bottom']:
        wallpaper[height - component_height:, width_start: width_end, :] = array(component)
    if position == position_map['rt']:
        wallpaper[:component_height, width - component_width:width, :] = array(component)
    if position == position_map['lt']:
        wallpaper[:component_height, :component_width, :] = array(component)
    if position == position_map['rb']:
        wallpaper[height - component_height:, width - component_width:width, :] = array(component)
    if position == position_map['lb']:
        wallpaper[height - component_height:, :component_width, :] = array(component)
    if position == position_map['center']:
        wallpaper[height_start:height_end, width_start:width_end, :] = array(component)
    Image.fromarray(wallpaper).save('wallpaper_{}.jpg'.format(position))


if __name__ == '__main__':
    # height and width of wallpaper
    height = 1080
    width = 1920
    position = 'default'
    image_path = ''

    component = Image.open(image_path)
    component_height, component_width = np.array(component).shape[:2]
    if component_height > height or component_width > width:
        if component_height / component_width < 1080 / 1920:
            width = component_width
            height = int(width / 1080 * 1920)
        else:
            height = component_height
            width = int(height * 1920 / 1080)
    # else:
    #     # 图片小于等于桌面
    #     pass

    wallpaper = np.zeros((height, width, 3), np.uint8)

    height_start = int((height - component_height) / 2)
    height_end = int((height + component_height) / 2)

    width_start = int((width - component_width) / 2)
    width_end = int((width + component_width) / 2)
    position_map = {
        'right': 'right',
        'left': 'left',
        'top': 'top',
        'bottom': 'bottom',
        'rt': 'rt',
        'rb': 'rb',
        'lt': 'lt',
        'lb': 'lb',
        'center': 'center',
    }
    right = 'right'
    left = 'left'
    top = 'top'
    bottom = 'bottom'
    rt = 'rt'
    rb = 'rb'
    lt = 'lt'
    lb = 'lb'
    center = 'center'

    if position == 'default':
        for key in position_map.keys():
            create_wallpaper(np.zeros((height, width, 3), np.uint8), key)
    else:
        create_wallpaper(np.zeros((height, width, 3), np.uint8), position)
