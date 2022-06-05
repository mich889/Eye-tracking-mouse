import math


def avg(*args):
    return sum(args)/len(args)

def distance(pt1,pt2):
    x1,y1 = pt1
    x2,y2 = pt2
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def blinkRatio(landmarks, right, left):
    rh_r = landmarks[right[0]]
    rh_l = landmarks[right[8]]
    rv_t = landmarks[right[12]]
    rv_b = landmarks[right[4]]

    lh_r = landmarks[left[0]]
    lh_l = landmarks[left[8]]
    lv_t = landmarks[left[12]]
    lv_b = landmarks[left[4]]

    rh = distance(rh_r,rh_l)
    rv = distance(rv_b,rv_t)

    lh = distance(lh_r,lh_l)
    lv = distance(lv_b, lv_t)

    return [rh/rv, lh/lv]
