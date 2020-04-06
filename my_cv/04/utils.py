import cv2
import numpy as np
import scipy.interpolate


def create_curve_func(points):
    """Return a function derived from control points."""
    if points is None:
        return None
    num_points = len(points)
    if num_points < 2:
        return None
    xs, ys = zip(*points)
    if num_points < 4:
        kind = 'linear'
    else:
        kind = 'cubic'
    return scipy.interpolate.interp1d(xs, ys, kind, bounds_error=False)


def create_lookup_array(func, length=256):
    """Return a lookup for whole-number inputs to a function.
     The lookup values are clamped to [0, length - 1]."""
    if func is None:
        return None
    lookup_array = np.empty(length)
    i = 0
    while i < length:
        func_i = func(i)
        lookup_array[i] = min((max(0, func_i), length - 1))
        i += 1
    return lookup_array


def apply_lookup_array(lookup_array, src, dst):
    """使用lookup数组，映射一组src到dst中"""
    if lookup_array is None:
        return
    dst[:] = lookup_array[src]


def create_composite_func(func0, func1):
    """返回两个函数的阻组合函数"""
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    return lambda x: func0(func1(x))

def create_flat_view(array):
    """返回数组的一维视图"""
    flat_view = array.view()
    flat_view.shape = array.size
    return flat_view


