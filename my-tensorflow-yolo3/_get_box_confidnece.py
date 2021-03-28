# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2



def get_box_confidence(img, gt_boxes, gt_labels, iou_threshold=0.5, 
    anchors, num_classes, downsample=32):

    """
    标注预测框有物体的置信度confidence

    -------------------------------------
    img:            输入图像数据，形状是[N, C, H, W]
    gt_boxes:       标注框，形状是[N, box_limit, 4]，其中box_limit是标注框的上限，缺省值为0
                    4是标注框坐标，坐标格式是[x, y, h, w]，都为关于输入图像数据大小的相对值
    gt_labels:      标注框的类别，形状是[N, box_limit]
    iou_threshold:  当候选框与标注框iou值不是最佳但大于一个阈值时，此候选框不会被标记为负样本，不计入损失值计算步骤
    anchors:        锚框大小
    num_classes:    类别数目
    downsample:     特征对于输入数据尺寸变化的比例

    Returns
    -------------------------------------
    box_confidence  np.array float, [batch_size, num_anchors, num_rows, num_cols]
    box_class       np.array float, [batch_size, num_anchors, num_classes, num_rows, num_cols]
    box_location    np.array float, [batch_size, num_anchors, 4, num_rows, num_cols]
    scale_location  np.array float, [batch_size, num_anchors, num_rows, num_cols]

    """

    img_shape = img.shape
    batch_size = img_shape[0]
    num_anchors = len(anchors)//2
    input_h = img_shape[2]
    input_w = img_shape[3]

    # 将图像按照downsample倍数分割为格子
    num_cols = input_w // downsample
    num_rows = input_h // downsample

    box_confidence = np.zeros([batch_size, num_anchors, num_rows, num_cols])
    box_class = np.zeros([batch_size, num_anchors, num_classes, num_rows, num_cols])
    box_location = np.zeros([batch_size, num_anchors, 4, num_rows, num_cols])

    scale_location = np.zeros([batch_size, num_anchors, num_rows, num_cols])

    # 找出最佳候选框
    for n in range(batch_size):
        # 遍历所有真实框，真实框数量有上限
        for n_gt in range(len(gt_boxes[n])):
            gt = gt_boxes[n][n_gt]
            gt_cls = gt_labels[n][n_gt]
            gt_cx = gt[0]
            gt_cy = gt[1]
            gt_h = gt[2]
            gt_w = gt[3]
            if (gt_w < 1e-3) or (gt_h < 1e-3):
                continue

            i = int(gt_h * num_rows)
            j = int(gt_w * num_cols)
            
            ious = []
            gt_box = [0., 0., float(gt_w), float(gt_h)]
            for ka in range(num_anchors):
                anchor_h = anchors[ka * 2]
                anchor_w = anchors[ka * 2 + 1]
                box = [0., 0., anchor_h/float(gt_h), anchor_w/float(gt_w)]
                # 计算 iou
                iou = box_iou_xyhw(gt_box, box)
                ious.append(iou)
                pass
            # iou不是最佳但又大于阈值的候选框，标记为-1
            # 先于最佳标注框的置信度修改，之后再将最佳标注框的置信度覆盖即可
            ious = np.array(ious)
            iou_mask = ious > iou_threshold
            box_confidence[n, iou_mask, i, j] = -1
            # 最佳标注框的置信度设置为1
            k = np.argsort(ious)[-1]
            box_confidence[n, k, i, j] = 1
            box_class[n, k, gt_cls, i, j] = 1
            # 设置最佳标注框的位置信息
            dx_box = gt_cx * num_rows - j
            dy_box = gt_cy * num_cols - i
            dh_box = np.log(gt_h * input_h / anchor_h[k*2])
            dw_box = np.log(gt_w * input_w / anchor_w[k*2+1])
            box_location[n, k, 0, i, j] = dx_box
            box_location[n, k, 1, i, j] = dy_box
            box_location[n, k, 2, i, j] = dh_box
            box_location[n, k, 3, i, j] = dw_box
            # scale 作加权参数，调节不同尺寸的锚框对loss的贡献
            scale_location[n, k, i, j] = 2.0 - gt_w * gt_h
            pass
        pass

    # 目前将每张图像上每个格子的最佳候选框的置信度设置为1，并标出了物体种类、位置回归的目标
    return box_confidence.astype('float32'), box_location.astype('float32'), \
            box_class.astype('float32'), scale_location.astype('float32')





