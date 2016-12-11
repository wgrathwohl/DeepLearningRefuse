"""
Trains a convolutional adversarial autoencoder (https://arxiv.org/pdf/1511.05644v2.pdf)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

from tensorflow.python.platform import gfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from layers import *
from utils import _add_loss_summaries
import data_handler
import load_video_data
from networks import *
import cv2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("checkpoint_path", None, "if used will restore the model")
tf.app.flags.DEFINE_integer('batch_size', 64, "number of seperate models in a batch")
tf.app.flags.DEFINE_integer('encoding_dimension', 8, "number of features in the encodings")
tf.app.flags.DEFINE_string('train_dir', None, "the training directory")
tf.app.flags.DEFINE_integer('max_steps', 1000000, "total number of steps to run")
tf.app.flags.DEFINE_integer("iterations_per_decay", 10000, "num iterations to decay")
tf.app.flags.DEFINE_integer("iterations_per_valid", 1000, "num iterations to validation")
tf.app.flags.DEFINE_float('initial_learning_rate', .0001, "initial learning rate")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1, "learning rate decay factor")
tf.app.flags.DEFINE_integer("valid_iterations", 10, "number of batches to pass in validation step")
tf.app.flags.DEFINE_integer("patch_size", 8, "size of patches")
tf.app.flags.DEFINE_integer("scales", 3, "size of patches")
tf.app.flags.DEFINE_string("loss", "l2", "loss to use")
MOVING_AVERAGE_DECAY = 0.9999


dataset = data_handler.MultiScaleCifar(FLAGS.patch_size, FLAGS.scales)
NUM_CHANNELS = dataset.channels
# compute the shift we need to do to align image values in [-1, 1]
MIN_IMAGE_VALUE = 0.0
MAX_IMAGE_VALUE = 1.0
IMAGE_VALUE_RANGE = MAX_IMAGE_VALUE - MIN_IMAGE_VALUE
IMAGE_SHIFT = (MAX_IMAGE_VALUE + MIN_IMAGE_VALUE) / 2.0
IMAGE_SCALE = 2.0 / IMAGE_VALUE_RANGE

def shift_images(ims):
    return (ims - IMAGE_SHIFT) * IMAGE_SCALE
def unshift_images(ims):
    return (ims / IMAGE_SCALE) + IMAGE_SHIFT

def discriminator_network(features, is_training, scope_name="discriminator", reuse=False):
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(features.get_shape().as_list(), "feautres SHAPE")
        fc1 = batch_normalized_linear_layer(features, "fc1", 64, .01, 0.0 if reuse else .004)
        fc2 = batch_normalized_linear_layer(fc1, "fc2", 64, .01, 0.0 if reuse else .004)
        out = linear_layer(fc2, "out", 1, .01, 0.0 if reuse else .004, nonlinearity=None)
    return out

def patch_encoder_network(images, is_training, scope_name="encoder", reuse=False):
    """
    Builds the network that encodes the images

    Args:
        images: Images returned from distorted_inputs() or inputs().
        is_training: True if training, false if eval
        reuse: if true, will use previously allocated variables
    Returns:
        image features
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 32
        conv1 = batch_normalized_conv_layer(
            images, "conv1", n_filters_conv1,
            [3, 3], "MSFT", 0.0004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = 32
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv2,
            [3, 3], "MSFT", 0.0004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")

        # conv3
        n_filters_conv3 = 64
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv3,
            [3, 3], "MSFT", 0.0004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        # conv4
        n_filters_conv4 = 64
        conv4 = batch_normalized_conv_layer(
            conv3, "conv4", n_filters_conv4,
            [3, 3], "MSFT", 0.0004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")

        conv4_flat, dim = reshape_conv_layer(conv4)
        print(conv4_flat.get_shape().as_list(), "CONV 4 FLAT SHAPE")

        fc5 = batch_normalized_linear_layer(conv4_flat, "fc5", 64, .01, 0.0 if reuse else .0004, test=test)
        print(fc5.get_shape().as_list(), "FC5")

        # 1 fully connected layer to maintain spatial info
        out = linear_layer(
            fc5, "output", FLAGS.encoding_dimension,
            .01, .0004 if not reuse else 0.0, nonlinearity=tf.nn.tanh
        )
        print(out.get_shape().as_list(), "OUT SHAPE")

    return out

def patch_decoder_network(features, is_training, scope_name="decorder", reuse=False):
    """
    Simple convolutional decorder
    uses transpose convolutions (incorrectly called deconvolutions)
    """
    batch_size, num_features = features.get_shape().as_list()
    batch_size = -1 if batch_size is None else batch_size
    print(features.get_shape().as_list())
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        # project input into enough dims to reshape into conv layer of
        # the below shape
        fc0 = batch_normalized_linear_layer(features, "fc0", 64, .01, 0.0 if reuse else .0004, test=test)
        print(fc0.get_shape().as_list(), "FC0")
        conv1_shape = (batch_size, 2, 2, 64)
        num_outputs_fc1 = conv1_shape[1] * conv1_shape[2] * conv1_shape[3]
        fc1 = batch_normalized_linear_layer(
            fc0, "fc1", num_outputs_fc1,
            .01, .0004 if not reuse else 0.0, test=test
        )
        conv1 = tf.reshape(fc1, conv1_shape)
        print(conv1.get_shape().as_list(), "Conv1 shape")

        conv2_shape = (batch_size, 4, 4, 64)
        conv2 = batch_normalized_deconv_layer(
            conv1, "deconv2", conv2_shape,
            [3, 3], [2, 2], .01, .0004 if not reuse else 0.0, test=test
        )
        print(conv2.get_shape().as_list(), "Conv2 shape")

        conv3_shape = (batch_size, 4, 4, 32)
        conv3 = batch_normalized_deconv_layer(
            conv2, "deconv3", conv3_shape,
            [3, 3], [1, 1], .01, .0004 if not reuse else 0.0, test=test
        )
        print(conv3.get_shape().as_list(), "Conv3 shape")

        conv4_shape = (batch_size, 8, 8, 32)
        conv4 = batch_normalized_deconv_layer(
            conv3, "deconv4", conv4_shape,
            [3, 3], [2, 2], .01, .0004 if not reuse else 0.0, test=test
        )
        print(conv4.get_shape().as_list(), "Conv4 shape")

        conv5_shape = (batch_size, 8, 8, 3)
        conv5 = deconv_layer(
            conv4, "deconv5", conv5_shape,
            [3, 3], [1, 1], .01, .0004 if not reuse else 0.0,
            nonlinearity=tf.tanh
        )
        print(conv5.get_shape().as_list(), "Conv5 shape")

    return conv5

def patch_encoder_network_fc(images, is_training, scope_name="encoder", reuse=False):
    """
    Builds the network that encodes the images

    Args:
        images: Images returned from distorted_inputs() or inputs().
        is_training: True if training, false if eval
        reuse: if true, will use previously allocated variables
    Returns:
        image features
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        features_flat, dim = reshape_conv_layer(images)
        print(features_flat.get_shape().as_list(), "features FLAT SHAPE")

        fc1 = batch_normalized_linear_layer(features_flat, "fc1", 128, .01, 0.0 if reuse else .0004, test=test)
        print(fc1.get_shape().as_list(), "FC1")

        fc2 = batch_normalized_linear_layer(fc1, "fc2", 128, .01, 0.0 if reuse else .0004, test=test)
        print(fc2.get_shape().as_list(), "FC2")

        fc3 = batch_normalized_linear_layer(fc2, "fc3", 128, .01, 0.0 if reuse else .0004, test=test)
        print(fc3.get_shape().as_list(), "FC3")

        out = linear_layer(fc3, "out", FLAGS.encoding_dimension, .01, 0.0 if reuse else .0004, nonlinearity=tf.tanh)
        print(out.get_shape().as_list(), "out")
    return out

def patch_decoder_network_fc(features, is_training, scope_name="decorder", reuse=False, patch_size=FLAGS.patch_size):
    """
    Simple convolutional decorder
    uses transpose convolutions (incorrectly called deconvolutions)
    """
    batch_size, num_features = features.get_shape().as_list()
    batch_size = -1 if batch_size is None else batch_size
    print(features.get_shape().as_list())
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        # project input into enough dims to reshape into conv layer of
        # the below shape
        fc0 = batch_normalized_linear_layer(features, "fc0", 128, .01, 0.0 if reuse else .0004, test=test)
        print(fc0.get_shape().as_list(), "FC0")

        fc1 = batch_normalized_linear_layer(fc0, "fc1", 128, .01, 0.0 if reuse else .0004, test=test)
        print(fc1.get_shape().as_list(), "FC1")

        fc2 = batch_normalized_linear_layer(fc1, "fc2", 128, .01, 0.0 if reuse else .0004, test=test)
        print(fc2.get_shape().as_list(), "FC2")

        fc3 = linear_layer(fc2, "fc3", (patch_size**2)*3, .01, 0.0 if reuse else .0004, nonlinearity=tf.tanh)
        print(fc2.get_shape().as_list(), "FC2")

        out = tf.reshape(fc3, [batch_size, patch_size, patch_size, 3], name="out")

    return out

def get_train_op(R_loss, G_loss, D_loss, global_step):
    lr = tf.train.exponential_decay(
        FLAGS.initial_learning_rate,
        global_step,
        FLAGS.iterations_per_decay,
        FLAGS.learning_rate_decay_factor,
        staircase=True
    )
    tf.scalar_summary("learning_rate", lr)
    # get variables for D and non-D optimization
    trainable_vars = tf.trainable_variables()
    D_vars = [var for var in trainable_vars if "discriminator" in var.name]
    G_vars = [var for var in trainable_vars if not "discriminator" in var.name]
    weight_loss = tf.add_n(tf.get_collection("losses"), name="weight_loss")

    loss_averages_op = _add_loss_summaries([R_loss, G_loss, D_loss])
    with tf.control_dependencies([loss_averages_op]):
        D_opt = tf.train.AdamOptimizer(lr).minimize(D_loss+weight_loss, var_list=D_vars)
        R_opt = tf.train.AdamOptimizer(lr).minimize(R_loss+G_loss+weight_loss, var_list=G_vars)

    # Track the moving averages of the batch norm variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

    # average the batch norm variables
    variables_to_average = list(
        set(
            [v for v in tf.all_variables() if "_mean" in v.name or "_variance" in v.name]
        )
    )
    variables_averages_op = variable_averages.apply(variables_to_average)

    for v in tf.trainable_variables():
        tf.histogram_summary(v.name, v)

    with tf.control_dependencies([variables_averages_op]):
        with tf.control_dependencies([D_opt, R_opt]):
            train_op = tf.no_op("train")
    return train_op

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Setup placeholders
        # the videos that will be read by the encoder
        patches_placeholder = tf.placeholder("float",
            shape=[
                FLAGS.batch_size * FLAGS.scales, FLAGS.patch_size, FLAGS.patch_size, 3
            ], name="images_placeholder"
        )
        tf.image_summary("patches", patches_placeholder)
        shifted_patches = shift_images(patches_placeholder)
        patch_encoding = patch_encoder_network_fc(shifted_patches, True)
        tf.histogram_summary("patch_encoding", patch_encoding)
        predicted_shifted_patches = patch_decoder_network_fc(patch_encoding, True)
        predicted_patches = unshift_images(predicted_shifted_patches)
        tf.image_summary("pred_patches", predicted_patches)

        if FLAGS.loss == "l2":
            print("Using l2 loss")
            R_loss = tf.reduce_mean(tf.square(shifted_patches - predicted_shifted_patches), name="l2_loss")
        else:
            print("Using l1 loss")
            R_loss = tf.reduce_mean(tf.abs(shifted_patches - predicted_shifted_patches), name="l1_loss")

        noise_encoding = tf.random_uniform(patch_encoding.get_shape().as_list(), -1, 1, name="noise_encoding")
        tf.histogram_summary("noise_encoding", noise_encoding)
        d_logits_p = discriminator_network(patch_encoding, True)
        d_logits_n = discriminator_network(noise_encoding, True, reuse=True)

        with tf.variable_scope("D_loss"):
            d_loss_p = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(d_logits_p, tf.zeros_like(d_logits_p))
            )
            d_loss_n = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(d_logits_n, tf.ones_like(d_logits_n))
            )
            D_loss = tf.identity((d_loss_p + d_loss_n) / 2., name="D_loss")
        with tf.variable_scope("G_loss"):
            G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(d_logits_p, tf.ones_like(d_logits_p)),
                name="G_loss"
            )

        train_op = get_train_op(R_loss, G_loss, D_loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())
        best_saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)

        if FLAGS.checkpoint_path is not None:
            print("Loading Checkpoint {}".format(FLAGS.checkpoint_path))
            restorer = tf.train.Saver(tf.all_variables())
            restorer.restore(sess, FLAGS.checkpoint_path)
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)

        global dataset
        best_valid_score = np.inf
        for step in xrange(FLAGS.max_steps):
            start = time.time()

            # for each scale, get a batch and concat them together
            patch_batch = dataset.multiscale_batch("train", FLAGS.batch_size)

            fd = {patches_placeholder: patch_batch}
            assert patch_batch.shape == (FLAGS.scales*FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, NUM_CHANNELS)
            if step % 100 == 0:
                _, rl, gl, dl, sum_str = sess.run([train_op, R_loss, G_loss, D_loss, summary_op], feed_dict=fd)
                summary_writer.add_summary(sum_str, step)
            else:
                _, rl, gl, dl = sess.run([train_op, R_loss, G_loss, D_loss], feed_dict=fd)

            duration = time.time() - start
            for l in (rl, gl, dl):
                assert not np.isnan(l), "loss diverged"
            if step % 10 == 0:
                print("({}) R loss = {}, G loss = {}, D loss = {} | ({} secs, {} batch/sec)".format(step, rl, gl, dl, duration, FLAGS.batch_size / duration))

            # run validation
            if step % FLAGS.iterations_per_valid == 0:
                valid_score = 0.0
                for b in xrange(FLAGS.valid_iterations):
                    valid_batch = dataset.multiscale_batch("valid", FLAGS.batch_size)
                    fd = {patches_placeholder: valid_batch}
                    rl, gl, dl = sess.run([R_loss, G_loss, D_loss], feed_dict=fd)
                    valid_score += rl
                valid_score /= FLAGS.valid_iterations
                print("Validation Loss: {}".format(valid_score))
                if valid_score < best_valid_score:
                    print("    This is the best model")
                    best_valid_score = valid_score
                    checkpoint_path = os.path.join(FLAGS.train_dir, "best_model.ckpt")
                    best_saver.save(sess, checkpoint_path, global_step=step)
                else:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
    if gfile.Exists(FLAGS.train_dir):
        gfile.DeleteRecursively(FLAGS.train_dir)
    gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()






