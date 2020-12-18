#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/12/19 1:48
# @Author  : strawsyz
# @File    : win_utils.py
# @desc:
import numpy
import win32api
import win32con
import win32gui
import win32ui
from PIL import Image


def get_window(class_name, title_name):
    window = win32gui.FindWindow(class_name, title_name)
    return window


def get_window_position(window):
    left, top, right, bottom = win32gui.GetWindowRect(window)
    return left, top, right, bottom


def get_sub_windows(parent_window):
    sub_windows = []
    win32gui.EnumChildWindows(parent_window, lambda window, param: param.append(window), sub_windows)
    return sub_windows


def get_window_title(window):
    title = win32gui.GetWindowText(window)
    return title


def get_class_name(window):
    class_name = win32gui.GetClassName(window)
    return class_name


def left_click(position: list):
    win32api.SetCursorPos(position)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP | win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)


def right_click(position: list):
    win32api.SetCursorPos(position)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTTUP | win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)


def close_window(window):
    win32gui.PostMessage(window, win32con.WM_CLOSE, 0, 0)


def show_all_windows():
    windows = []
    win32gui.EnumWindows(lambda window, param: param.append(window), windows)
    print("num of windows : {}".format(len(windows)))

    for window in windows:
        title = win32gui.GetWindowText(window)
        class_name = win32gui.GetClassName(window)
        print("window's title : {}\t class name : {}".format(title, class_name))


def capture_window(class_name: str, save_path: str):
    # get window
    window = win32gui.FindWindow(class_name, None)
    left, top, right, bottom = win32gui.GetWindowRect(window)
    width = right - left
    height = bottom - top
    # get window DC
    window_DC = win32gui.GetWindowDC(window)
    # create DC
    new_window_DC = win32ui.CreateDCFromHandle(window_DC)
    # create Compatible DC
    compatible_DC = new_window_DC.CreateCompatibleDC()
    # create bitmap
    bitmap = win32ui.CreateBitmap()
    # create space for bitmap
    bitmap.CreateCompatibleBitmap(new_window_DC, width, height)
    # save window shot  at compatible DC
    compatible_DC.SelectObject(bitmap)
    # save bitmap at window DC
    compatible_DC.BitBlt((0, 0), (width, height), new_window_DC, (0, 0), win32con.SRCCOPY)
    # save bitmap at local file
    bitmap.SaveBitmapFile(compatible_DC, save_path)

    # return PIL.Image
    bitmap_info = bitmap.GetInfo()
    bitmap_bits = bitmap.GetBitmapBits(True)
    image = Image.frombuffer("RGB", (bitmap_info['bmWidth'], bitmap_info['bmHeight']), bitmap_bits, "raw", "BGRX", 0, 1)

    # return numpy array
    bitmap_bits = bitmap.GetBitmapBits(True)
    np_array = numpy.frombuffer(bitmap_bits, dtype='uint8')
    np_array.shape = (height, width, 4)

    # release memory
    win32gui.DeleteObject(bitmap.GetHandle())
    compatible_DC.DeleteDC()
    new_window_DC.DeleteDC()
    win32gui.ReleaseDC(window, window_DC)

    return image, np_array
