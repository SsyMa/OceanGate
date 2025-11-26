import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import backend as K

@register_keras_serializable()
def iou_metric(y_true, y_pred, smooth=1e-6):
    # Ensure both are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.0, tf.float32)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

@register_keras_serializable()
def f2_metric(y_true, y_pred, beta=2, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.0, tf.float32)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum(y_pred_f) - tp
    fn = K.sum(y_true_f) - tp

    f2 = (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp + smooth)
    return f2