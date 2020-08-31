"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from . import backend
import keras.backend as K
import tensorflow as tf
import math

def smooth_batch_labels(batch_labels, factor=0.1):
    """
        Args:
            batch_labels: array of shape [B, A, C+1]
                B: batch size, A: anchor count, C: class count
            factor: factor to smooth by

    """
    # remove anchor state
    labels = batch_labels[:, :, :-1]
    # compute smoothed labels and replace
    batch_labels[:, :, :-1] = smooth_labels(labels, factor)

    return batch_labels

def smooth_labels(labels, factor=0.1):
    labels *= (1 - factor)
    labels += (factor / labels.shape[2])
    return labels

def focal(alpha=0.25, gamma=2.0, cutoff=0.5):
    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.
        cutoff: Positive prediction cutoff for soft targets

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels         = y_true[:, :, :-1]
        anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices        = backend.where(keras.backend.not_equal(anchor_state, -1))
        labels         = backend.gather_nd(labels, indices)
        classification = backend.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = backend.where(keras.backend.greater(labels, cutoff), alpha_factor, 1 - alpha_factor)
        focal_weight = backend.where(keras.backend.greater(labels, cutoff), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = backend.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma=3.0):
    """ Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1


def diou_loss(image, anchors, use_ciou=False):
    """ Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    shape = keras.backend.cast(keras.backend.shape(image), keras.backend.floatx())
    if keras.backend.image_data_format() == 'channels_first':
        input_shape = backend.unstack(shape, axis=0)[2:4]# height, width
    else:
        input_shape = backend.unstack(shape, axis=0)[1:3]

    def _diou_loss(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        batch_size = K.shape(y_pred)[0]  # batch size, tensor
        batch_size_f = K.cast(batch_size, K.dtype(y_pred))
        # separate target and state
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        #regression
        regression = backend.bbox_transform_inv(anchors, regression)
        regression_target = backend.bbox_transform_inv(anchors, regression_target)
        # filter out "ignore" anchors
        indices = backend.where(keras.backend.equal(anchor_state, 1))
        regression = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)
        boxes_xy = (regression_target[..., 0:2] + regression_target[..., 2:4]) / 2
        boxes_xy /= input_shape[::-1]
        boxes_wh = (regression_target[..., 2:4] - regression_target[..., 0:2])/input_shape[::-1]
        regression_target = tf.concat([boxes_xy, boxes_wh],axis=-1)

        boxes_xy = (regression[..., 0:2] + regression[..., 2:4]) / 2
        boxes_xy /= input_shape[::-1]
        boxes_wh = (regression[..., 2:4] - regression[..., 0:2])/input_shape[::-1]
        regression = tf.concat([boxes_xy, boxes_wh],axis=-1)
        diou = box_diou(b_true=regression_target, b_pred = regression,use_ciou=use_ciou)
        box_loss_scale = 2 - regression_target[..., 2:3] * regression_target[..., 3:4]
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        diou_loss = box_loss_scale * (1 - diou)
        diou_loss = K.sum(diou_loss)/batch_size_f
        location_loss = diou_loss/normalizer
        return location_loss

    return _diou_loss

def box_diou(b_true, b_pred, use_ciou=False):
    """
    Calculate DIoU/CIoU loss on anchor boxes
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287
    Parameters
    ----------
    b_true: GT boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b_pred: predict boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    use_ciou: bool flag to indicate whether to use CIoU loss type
    Returns
    -------
    diou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    b_true_xy = b_true[..., :2]
    b_true_wh = b_true[..., 2:4]
    b_true_wh_half = b_true_wh/2.
    b_true_mins = b_true_xy - b_true_wh_half
    b_true_maxes = b_true_xy + b_true_wh_half

    b_pred_xy = b_pred[..., :2]
    b_pred_wh = b_pred[..., 2:4]
    b_pred_wh_half = b_pred_wh/2.
    b_pred_mins = b_pred_xy - b_pred_wh_half
    b_pred_maxes = b_pred_xy + b_pred_wh_half

    intersect_mins = K.maximum(b_true_mins, b_pred_mins)
    intersect_maxes = K.minimum(b_true_maxes, b_pred_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
    b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
    union_area = b_true_area + b_pred_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + K.epsilon())

    # box center distance
    center_distance = K.sum(K.square(b_true_xy - b_pred_xy), axis=-1)
    # get enclosed area
    enclose_mins = K.minimum(b_true_mins, b_pred_mins)
    enclose_maxes = K.maximum(b_true_maxes, b_pred_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    # get enclosed diagonal distance
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (enclose_diagonal + K.epsilon())

    if use_ciou:
        # calculate param v and alpha to extend to CIoU
        v = 4*K.square(tf.math.atan2(b_true_wh[..., 0], b_true_wh[..., 1]) - tf.math.atan2(b_pred_wh[..., 0], b_pred_wh[..., 1])) / (math.pi * math.pi)

        # a trick: here we add an non-gradient coefficient w^2+h^2 to v to customize it's back-propagate,
        #          to match related description for equation (12) in original paper
        #
        #
        #          v'/w' = (8/pi^2) * (arctan(wgt/hgt) - arctan(w/h)) * (h/(w^2+h^2))          (12)
        #          v'/h' = -(8/pi^2) * (arctan(wgt/hgt) - arctan(w/h)) * (w/(w^2+h^2))
        #
        #          The dominator w^2+h^2 is usually a small value for the cases
        #          h and w ranging in [0; 1], which is likely to yield gradient
        #          explosion. And thus in our implementation, the dominator
        #          w^2+h^2 is simply removed for stable convergence, by which
        #          the step size 1/(w^2+h^2) is replaced by 1 and the gradient direction
        #          is still consistent with Eqn. (12).
        v = v * tf.stop_gradient(b_pred_wh[..., 0] * b_pred_wh[..., 0] + b_pred_wh[..., 1] * b_pred_wh[..., 1])

        alpha = v / (1.0 - iou + v)
        diou = diou - alpha*v

    diou = K.expand_dims(diou, -1)
    return diou