"""
Implements loss new loss functions
"""
import tensorflow as tf
from utils import squared_difference

def scale_labels(labels, margin=1):
    """
    Converts 0,1 labels to -margin,margin labels
    """
    return (2.0 * margin * labels) - margin


def hinge_loss(logits, labels, name=None):
    """
    Implements squared hinge loss
    """
    scaled_labels = scale_labels(labels)
    logits_labels = tf.mul(logits, scaled_labels)
    logits_labels_shifted = tf.minimum(logits_labels - 1.0, 0.0)
    squared_component_hinge_loss = tf.square(logits_labels_shifted)
    loss = tf.reduce_sum(squared_component_hinge_loss, 1)
    return loss


def smooth_l1_loss(x, scaling_factor=1.0, name=None):
    """
    Implements the smooth L1 loss from http://arxiv.org/pdf/1504.08083v2.pdf

    Computes f(x) = .5 * x**2 if |x| < 1 else |x| - .5
    """
    with tf.variable_scope("smooth_l1") as scope:
        scaled = scaling_factor * x
        abs_scaled = tf.abs(scaled)
        sqr = .5 * tf.square(abs_scaled)
        minus = abs_scaled - .5
        # get where the scaled value is < 1
        lt = tf.less(abs_scaled, 1)
        # where true use sqr
        scaled_out = tf.select(lt, sqr, minus)
        out = tf.identity(scaled_out / scaling_factor, name=name)
    return out


def NormalKLDivergence(mu1, var1, mu2=0.0, var2=1.0, scope_name="KL_Div"):
    """
    Computes the KL divergence between two normal distributions with means mu1 and m2 and variances var1 and var2
    """
    with tf.variable_scope(scope_name) as scope:
        num = var1 + squared_difference(mu1, mu2)
        t1 = tf.div(num, 2 * var2)
        std1 = tf.sqrt(var1)
        std2 = tf.sqrt(var2)
        t2 = tf.log(tf.div(std2, std1))
        out = t1 + t2 - .5
    return out
