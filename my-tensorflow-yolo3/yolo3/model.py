# -*- coding: utf-8 -*-

from functools import wraps

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from yolo3.utils import compose



@wraps(Conv2D)
def DarknetConv2D(*arg, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""

    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*arg, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*arg, **kwargs):
    """A CBL structure, DarknetConv2D followed by BatchNormalization and LeakyReLU
        layers: 3
    """

    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)

    return compose(
        DarknetConv2D(*arg, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=.1))


def resunit_body(x, num_filters, num_units):
    """A series of resunit blocks behind of a Convolution2D for downsample
        layers: 1 + 3 + 6 + 1 = 11
    """
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0), (1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_units):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    """Darknet body has Convolution2D layers
        layers: 3 + 
    """

    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resunit_body(x, 64, 1)
    x = resunit_body(x, 128, 2)
    x = resunit_body(x, 256, 8)
    x = resunit_body(x, 512, 8)
    x = resunit_body(x, 1024, 4)
    return x


def last_block(x, num_filters, out_filters):
    """(5 + 1) CBL structures followed by a Convolution2D layer
        layers: 
    """

    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1))
        )(x)

    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1))
        )(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """YOLO_v3 model body"""

    darknet = Model(inputs, darknet_body(inputs))

    x, y0 = last_block(darknet.output, 512, num_anchors*(5+num_classes))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2)
        )(x)
    # Concatenate 1
    # print the darknet body layers to find this 152 layer
    x = Concatenate()([x, darknet.layers[152].output])

    x, y1 = last_block(x, 256, num_anchors*(5+num_classes))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2)
        )(x)
    # Concatenate 2
    # print the darknet body layers to find this 92 layer
    x = Concatenate()([x, darknet.layers[92].output])

    x, y2 = last_block(x, 128, num_anchors*(5+num_classes))

    return Model(inputs, [y0, y1, y2])



def yolo_head(features, anchors, num_classes, input_shape, calc_loss=False):
    """
    Convert final layer features to bounding box parameters.

    Parameters
    --------------------------------------------

    features:           特征图
    anchors:            锚框
    num_classes:        类别数量
    input_shape:        输入大小 yolov3为(416, 416)
    calc_loss:          默认为False


    Returns
    --------------------------------------------

    boxes:              基于原图坐标系的左下、右上角坐标，为实际像素值，非比值

    """
    
    # features` shape [x, y, h, w, confi, num of classes]
    num_anchors = len(anchors)
    # Reshape to [N, H, W, num_anchors, box_param]
    anchors_tensor = K.reshape(K.constant(anchors), shape=[1, 1, 1, num_anchors, 2])

    # build the grid panel
    grid_shape = K.shape(features)[1:3] # height and weight for [N, H, W, C] 
    '''
    a = K.arange(0, stop=grid_shape[0]) ->  [0, 1, 2, 3, 4]
    b = K.reshape(a, [-1, 1])           ->  [[0], [1], [2], [3], [4]]
    c = K.tile(b, [1, 5])               ->  [[0, 0, 0, 0, 0],
                                             [1, 1, 1, 1, 1],
                                             [2, 2, 2, 2, 2],
                                             [3, 3, 3, 3, 3],
                                             [4, 4, 4, 4, 4]]
    '''
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1,1,1,1]),
        [1, grid_shape[1], 1, 1])
    '''
    a = K.arange(0, stop=grid_shape[0]) ->  [0, 1, 2, 3, 4]
    b = K.reshape(a, [1, -1])           ->  [[0, 1, 2, 3, 4]]
    c = K.tile(b, [5, 1])               ->  [[0, 1, 2, 3, 4],
                                             [0, 1, 2, 3, 4],
                                             [0, 1, 2, 3, 4],
                                             [0, 1, 2, 3, 4],
                                             [0, 1, 2, 3, 4]]
    '''
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1,-1,1,1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(features))

    # Reshape to [N, H, W, C, CLS]
    features = K.reshape(features, 
        [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust predictions to each spatial grid point and anchor size
    # [():():(-1)]=[-1:-len(x)-1:1]
    box_xy = (K.sigmoid(features[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(features))
    box_wh = K.exp(features[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(features))
    box_confidence = K.sigmoid(features[..., 4:5])
    box_cls = K.sigmoid(features[..., 5:])

    if calc_loss == True:
        return grid, features, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_cls



def yolo_correct_box(box_xy, box_wh, input_shape, image_shape):
    """
    Get endpoints of each corrected box 

    Parameters
    --------------------------------------------

    box_xy:             候选框中心坐标值，为相对当前格子左上角坐标的偏移比值
    box_wh:             候选框宽高值,为相对原图像大小的比值
    input_shape:        模型输入图像尺寸， 长宽均是32的倍数
    image_shape:        原图尺寸，经裁剪为输入尺寸


    Returns
    --------------------------------------------

    boxes:              基于原图坐标系的lefttop、rightbottom坐标，为实际像素值，非比值

    """

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    # calculate the new_shape according the ratio between input_shape and image_shape
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    # calc endpoints 
    box_mins = box_yx - box_hw / 2.
    box_maxes = box_yx + box_hw / 2.
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_mins
        box_mins[..., 1:2],  # x_mins
        box_maxes[..., 0:1], # y_maxes
        box_maxes[..., 1:2], # x_maxes
    ])

    # convert to real pixel values
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes



def yolo_boxes_and_scores(features, anchors, num_classes, input_shape, image_shape):
    """
    Process Conv layer output.

    Parameters
    --------------------------------------------

    features:           特征图
    anchors:            锚框
    num_classes:        类别数量
    input_shape:        输入大小 yolov3为(416, 416)
    image_shape:        原图大小


    Returns
    --------------------------------------------

    boxes:              基于原图坐标系的lefttop、rightbottom坐标，为实际像素值，非比值
    box_scores:         候选框的scores

    """

    box_xy, box_wh, box_confidence, box_cls_probs = yolo_head(features, 
        anchors, num_classes, input_shape)
    
    # yolo.py draw rectangle
    # order: ymin, xmin, ymax, xmax
    boxes = yolo_correct_box(box_xy, box_wh, input_shape, image_shape)
    # 将后四位坐标合并到一个维度内
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_cls_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])

    return boxes, box_scores


#   max_boxes = 10, 环境内车牌的数量不会过多
def yolo_eval(yolo_outputs, anchors, num_classes, image_shape, 
    max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Evaluate the model on given input, return filtered boxes.

    Parameters
    --------------------------------------------

    yolo_outputs:       yolo_body的输出
    anchors:            锚框
    num_classes:        类别数量
    image_shape:        原图大小
    max_boxes:          预设值，图中最多有多少个真实框
    score_threshold:    score阈值， 小于阈值的框直接丢弃
    iou_threshold:      iou阈值，大于阈值的框的置信度标为-1，不计入loss计算过程


    Returns
    --------------------------------------------
    box_masked:         经过阈值过滤和非极大值抑制后得到的目标框坐标,     [?, 4]
    scores_masked:      经过阈值过滤和非极大值抑制后得到的目标框分数,     [?, 1]
    classes_nms:        经过阈值过滤和非极大值抑制后得到的目标框类别,     [?, 1] 数字的值代表类别编号

    """

    num_layers = len(yolo_outputs)                          # yolov3共有三组输出
    # 此为mask预设值， 可修改
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32        # downsample=32

    boxes = []
    box_scores = []

    # 分别对三组输出进行
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l], num_classes, input_shape, image_shape])
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)


    # 过滤小于score_threshold的目标框
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.concatenate(max_boxes, dtype='int32')

    box_masked = []
    scores_masked = []
    classes_nms = []

    # 对目标框进行mask过滤、非极大值抑制
    for c in range(num_classes):
        # tf.boolean_mask 会直接从张量里删除元素，而不是置为0
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, dtype='int32') * c

        box_masked.append(class_boxes)
        scores_masked.append(class_box_scores)
        classes_nms.append(classes)

    box_masked = K.concatenate(box_masked, axis=0)
    scores_masked = K.concatenate(scores_masked, axis=0)
    classes_nms = K.concatenate(classes_nms, axis=0)

    return box_masked, scores_masked, classes_nms



def box_iou(b1, b2):

    """
    计算IoU

    Parameters
    -------------------------------------

    b1, b2:         回归框，形状为[x, y, h, w]


    Returns
    -------------------------------------
    iou             float

    """

    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxs = b1_xy + b1_wh_half

    b2 = K.expand_dims(b2, -2)
    b2_xy = b2[..., :1]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b1_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxs = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxs = K.maximum(b1_maxs, b2_maxs)
    intersect_wh = K.maximum(intersect_maxs - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou



def yolo_loss(args, anchors, num_classes, iou_threshold=.5, print_loss=False):
    """
    Calculate yolo loss

    Parameters
    --------------------------------------------

    args:               yolo_body 的输出
    anchors:            锚框
    num_classes:        类别数量
    iou_threshold:      iou阈值，大于阈值的框的置信度标为-1，不计入loss计算过程


    Returns
    --------------------------------------------
    loss:               tensor, shape=(1,)

    """

    num_layers = len(anchors) // 3
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    # print('anchors: {}'.format(anchors))
    # print('args shape: {}'.format(args))

    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]


    loss = 0
    m = K.shape(yolo_outputs[0])[0]     # batch_size tensor, x 
    mf = K.cast(m, K.dtype(yolo_outputs[0]))


    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]   # box_confidence
        true_cls_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l], 
            anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # print('raw_pred shape: {}'.format(raw_pred.shape))
        # 
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))
        # scale 作加权参数，调节不同尺寸的真实框对loss的贡献
        box_loss_scale = 2.0 - y_true[l][..., 2:3] * y_true[l][..., 3:4]
 
        # 
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        # Reference: https://tensorflow.google.cn/api_docs/python/tf/TensorArray
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,..., 0:4], object_mask_bool[b,..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < iou_threshold, K.dtype(true_box)))
            return b+1, ignore_mask

        # Reference: https://tensorflow.google.cn/api_docs/python/tf/while_loop
        # param1: lambda b, *args: b<m -- 条件 ; param2: loop_body -- 满足条件后将loop_vars[0]传入loop_body更新
        # 更新loop_vars[0] 和 loop_vars[1] 传入b, *args
        # _, ignore_mask -> 0, ignore_mask
        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()   # 将tf.TensorArray转化为tensor
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # x y loss
        # print(raw_true_xy, raw_pred[0:2])
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, 
            raw_pred[..., 0:2], from_logits=True)
        # w h loss
        wh_loss = object_mask * box_loss_scale * .5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                    (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_cls_probs, raw_pred[..., 5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf

        loss = xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss



def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """
    将true_boxes 转变为 label形式, 
    y_true[l][batch, grid_x, grid_y, num_layers, 5 + num_classes]

    Parameters
    --------------------------------------------

    true_boxes:         [m, T, 5]
                               5 -> [x_min, y_min, x_max, y_max, class_id]
    input_shape:        hw, 32 的倍数
    anchors:            锚框
    num_classes:        类别数量


    Returns
    --------------------------------------------
    y_true:             和 yolo_output 形状一样
                        y_true[l][batch, grid_x, grid_y, num_layers, 5 + num_classes]

    """

    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='float32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    # print('true_boxes: {}'.format(true_boxes))
    # print('anchors: {},{}'.format(anchors.dtype, anchors))

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    # numpy 1.19 numpy.float32 cannot be interpreted as an integer
    # we need to manually convert them.
    y_true = [np.zeros((m, int(grid_shapes[l][0]), int(grid_shapes[l][1]), len(anchor_mask[l]), 5 + num_classes), 
        dtype='float32') for l in range(num_layers)]

    # 
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue

        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true
