# import the necessary packages
import numpy as np


def area(box):
    """计算box的面积"""
    x1, y1, x2, y2 = box
    w = np.maximum(0, x2-x1+1)
    h = np.maximum(0, y2-y1+1)
    return w*h
    # 源代码有问题
    # return (abs(box[2] - box[0])) * (abs(box[3] - box[1]))


def overlaps(a, b, thresh=0.5):
    """判断重叠是否超过一定的阈值"""
    print("checking overlap ")
    print(a, b)
    x1 = np.maximum(a[0], b[0])
    y1 = np.maximum(a[1], b[1])
    x2 = np.minimum(a[2], b[2])
    y2 = np.minimum(a[3], b[3])
    # 计算重叠区的面积
    intersect = float(area([x1, y1, x2, y2]))
    return intersect / np.minimum(area(a), area(b)) >= thresh


def is_inside(rec1, rec2):
    def inside(a, b):
        if (a[0] >= b[0]) and (a[2] <= b[0]):
            return (a[1] >= b[1]) and (a[3] <= b[3])
        else:
            return False

    return (inside(rec1, rec2) or inside(rec2, rec1))


# Malisiewicz / Rosebrock的代码
def non_max_suppression_fast(boxes, overlap_thresh=0.5):
    """根据boxes的评分进行排序，从评分高的开始。消除重叠区域的面积超过一定阈值的box"""
    # 如果没有区域，则返回空列表
    if len(boxes) == 0:
        return []

    # 如果是int类型就转为float类型
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    scores = boxes[:, 4]
    # 从小到大排序
    score_idx = np.argsort(scores)

    while len(score_idx) > 0:
        box = scores[score_idx[0]]
        print("checking box")
        for s in score_idx:
            to_delete = []
            if s == 0:
                continue
            # 判断重叠部分是否超过阈值
            if (overlaps(boxes[s], boxes[box], overlap_thresh)):
                to_delete.append(box)
                score_idx = np.delete(score_idx, [s], 0)
        boxes = np.delete(boxes, to_delete, 0)
        score_idx = np.delete(score_idx, 0, 0)

    return boxes.astype('int')


"""


  # initialize the list of picked indexes 
  pick = []

  # grab the coordinates of the bounding boxes
  x1 = boxes[:,0]
  y1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]

  # compute the area of the bounding boxes and sort the bounding
  # boxes by the bottom-right y-coordinate of the bounding box
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(y2)

  # keep looping while some indexes still remain in the indexes
  # list
  while len(idxs) > 0:
    # grab the last index in the indexes list and add the
    # index value to the list of picked indexes
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)

    # find the largest (x, y) coordinates for the start of
    # the bounding box and the smallest (x, y) coordinates
    # for the end of the bounding box
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    # compute the ratio of overlap
    overlap = (w * h) / area[idxs[:last]]

    # delete all indexes from the index list that have
    idxs = np.delete(idxs, np.concatenate(([last],
      np.where(overlap > overlapThresh)[0])))

  # return only the bounding boxes that were picked using the
  # integer data type
  return boxes[pick].astype("int")
"""
