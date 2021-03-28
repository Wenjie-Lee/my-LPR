# -*- coding: utf-8 -*-

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def compose(*funcs):

    """
    组合任意多个函数，从左到右执行

    Reference: https://mathieularose.com/function-composition-in-python/
    """

    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('要组合的函数序列不能为空.')



def letterbox_image(image, size):
    """
    不改变图片长宽比的情况下，使图片的某一边长度与size对应边相等

    Parameters
    --------------------------------------------

    image:                  输入图片
    size:                   图片需要适应的大小

    Returns
    --------------------------------------------

    new_image:              修改后的图片

    """
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    # 创建一个灰色底片
    new_image = Image.new('RGB', size, (128,128,128))
    # PIL Image.paste(img, box) box=2-tuple, treated as the upper left corner
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image



def rand(a=0, b=1):
    return np.random.rand()*(b-a)+a



def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''
    random preprocessing for real-time data augmentation
    对输入图像进行实时的数值归一化、随机数据增强操作

    Parameters
    --------------------------------------------

    annotation_line:        图片标注信息
    input_shape:            模型输入大小
    random:                 是否要做随机数据增强，增强方式和下面几个参数有关
    max_boxes:              一张图片上真实框数量上限
    jitter:                 图像随机长宽拉伸，jitter值越大，拉伸的效果可能越明显
    hue:                    HSV随机改变
    sat:                    HSV随机改变
    val:                    HSV随机改变
    proc_img:               是否对图像做大小修改，默认为是，即二者都修改

    Returns
    --------------------------------------------

    image_data:             大小适应、数值归一化后的图片数据, np.array
    box_data:               对当前图片的标定框做限定数量、大小适应等操作

    '''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    # 不做随机数据增强
    if not random:
        # resize image
        # 使图像大小适应输入大小的要求
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            # 归一化
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            # box 数量不能超过 max_boxes
            if len(box)>max_boxes: box = box[:max_boxes]
            # 对标定框也做上述大小适应操作
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    # 对图像做随机长宽之间的拉伸
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # place image
    # 将拉伸后的图像随机放置到模型输入大小的画布上
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    # 随机反转图像，车牌检测不应有这一步
    # flip = rand()<.5
    # if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    # 对图像做随机色调（H），饱和度（S），明度（V）改变
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    # RGB -> HSV
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # correct boxes
    # 对box做大小适应修正
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        # if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        # 标定框大小要合法
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data