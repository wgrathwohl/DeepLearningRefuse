"""
Will train a simple convolutional autoencoder using l2 loss
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

FLAGS = tf.app.flags.FLAGS
# A hack to stop conflicting batch sizes
try:
    print(FLAGS.batch_size)
except:
    tf.app.flags.DEFINE_integer(
        'batch_size', 64,
        """Number of images to process in a batch."""
    )
tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/conv_autoencoder_cifar_adv',
    """Directory where to write event logs and checkpoint."""
)
tf.app.flags.DEFINE_integer(
    'max_steps', 1000000,
    """Number of batches to run."""
)
tf.app.flags.DEFINE_boolean(
    'log_device_placement', False,
    """Whether to log device placement."""
)

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
USE_CHECKPOINT = False
CHECKPOINT_PATH = ''
NUM_EPOCHS_PER_DECAY = 10.0
NUM_ITERATIONS_PER_DECAY = 10000
INITIAL_LEARNING_RATE = 0.001     # Initial learning rate.


# read in dataset
# load data set
[train_set, valid_set, test_set, width, height, channels, num_labels] = data_handler.load_cifar_data()

FULL_IMAGE_SIZE = (height, width, channels)
IMAGE_SIZE = (height, width, channels)
NUM_CHANNELS = FULL_IMAGE_SIZE[-1]

# compute the shift we need to do to align image values in [-1, 1]
MIN_IMAGE_VALUE = np.min(train_set[0])
MAX_IMAGE_VALUE = np.max(train_set[0])

IMAGE_VALUE_RANGE = MAX_IMAGE_VALUE - MIN_IMAGE_VALUE

IMAGE_SHIFT = (MAX_IMAGE_VALUE + MIN_IMAGE_VALUE) / 2.0
IMAGE_SCALE = 2.0 / IMAGE_VALUE_RANGE


LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

def image_encoder_network(images, is_training, scope_name="encoder", reuse=False):
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
        n_filters_conv1 = 8
        conv1 = batch_normalized_conv_layer(
            images, "conv1", NUM_CHANNELS, n_filters_conv1,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = 16
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv1, n_filters_conv2,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")

        # conv3
        n_filters_conv3 = 32
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv2, n_filters_conv3,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        # conv4
        n_filters_conv4 = 64
        conv4 = batch_normalized_conv_layer(
            conv3, "conv4", n_filters_conv3, n_filters_conv4,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")

        # conv5
        n_filters_conv5 = 128
        conv5 = batch_normalized_conv_layer(
            conv4, "conv5", n_filters_conv4, n_filters_conv5,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        print(conv5.get_shape().as_list(), "CONV 5 SHAPE")

        # do global mean pooling
        out = global_pooling_layer(conv5, scope.name)
        print(out.get_shape().as_list(), "OUT SHAPE")

    return out


def discriminator_network(images, is_training, scope_name="discriminator", reuse=False):
    """
    Builds the network that is trainged to discriminate between generated images
    and images from the dataset

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
        n_filters_conv1 = 8
        conv1 = batch_normalized_conv_layer(
            images, "conv1", NUM_CHANNELS, n_filters_conv1,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = 16
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv1, n_filters_conv2,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")

        # conv3
        n_filters_conv3 = 32
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv2, n_filters_conv3,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        # conv4
        n_filters_conv4 = 64
        conv4 = batch_normalized_conv_layer(
            conv3, "conv4", n_filters_conv3, n_filters_conv4,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")

        # conv5
        n_filters_conv5 = 128
        conv5 = batch_normalized_conv_layer(
            conv4, "conv5", n_filters_conv4, n_filters_conv5,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        print(conv5.get_shape().as_list(), "CONV 5 SHAPE")

        # do global mean pooling
        top_feats = global_pooling_layer(conv5, scope.name+"_features")
        print(top_feats.get_shape().as_list(), "feats shape")

        out = batch_normalized_linear_layer(
            top_feats, scope.name, n_filters_conv5, 1,
            .01, .004 if not reuse else 0.0, test=test
        )
        print(out.get_shape().as_list(), "disc out shape")

    return out

def image_decoder_network(features, is_training, scope_name="decorder", reuse=False):
    """
    Simple convolutional decorder
    uses transpose convolutions (incorrectly called deconvolutions)
    """
    batch_size, num_features = features.get_shape().as_list()
    print(features.get_shape().as_list())
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        # project input into enough dims to reshape into conv layer of
        # the below shape
        conv1_shape = (batch_size, 4, 4, 128)
        num_outputs_fc1 = conv1_shape[1] * conv1_shape[2] * conv1_shape[3]
        fc1 = batch_normalized_linear_layer(
            features, "fc1", num_features, num_outputs_fc1,
            .01, .004 if not reuse else 0.0, test=test
        )
        conv1 = tf.reshape(fc1, conv1_shape)
        print(conv1.get_shape().as_list(), "Conv1 shape")

        conv2_shape = (batch_size, 4, 4, 64)
        conv2 = batch_normalized_deconv_layer(
            conv1, "deconv2", conv1_shape[-1], conv2_shape[-1], conv2_shape,
            [3, 3], [1, 1], .01, .004 if not reuse else 0.0, test=test
        )
        print(conv2.get_shape().as_list(), "Conv2 shape")

        conv3_shape = (batch_size, 8, 8, 32)
        conv3 = batch_normalized_deconv_layer(
            conv2, "deconv3", conv2_shape[-1], conv3_shape[-1], conv3_shape,
            [5, 5], [2, 2], .01, .004 if not reuse else 0.0, test=test
        )
        print(conv3.get_shape().as_list(), "Conv3 shape")

        conv4_shape = (batch_size, 16, 16, 16)
        conv4 = batch_normalized_deconv_layer(
            conv3, "deconv4", conv3_shape[-1], conv4_shape[-1], conv4_shape,
            [5, 5], [2, 2], .01, .004 if not reuse else 0.0, test=test
        )
        print(conv4.get_shape().as_list(), "Conv4 shape")

        conv5_shape = (batch_size, 32, 32, NUM_CHANNELS)
        conv5 = deconv_layer(
            conv4, "deconv5", conv4_shape[-1], conv5_shape[-1], conv5_shape,
            [5, 5], [2, 2], .01, .004 if not reuse else 0.0,
            nonlinearity=tf.tanh
        )
        print(conv5.get_shape().as_list(), "Conv5 shape")

    return conv5


def autoencoder_train(total_loss, global_step):
    """
    Train autoencoder

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
          processed.
    Returns:
        train_op: op for training.
    """
    # Variables that affect learning rate.
    decay_steps = NUM_ITERATIONS_PER_DECAY

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        global_step,
        decay_steps,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True
    )

    tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)


    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad:
            print(var.name)
            tf.histogram_summary(var.op.name + '/gradients', grad)

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

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def train():
    """Train a convolutional autoencoder for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Setup placeholders
        # the videos that will be read by the encoder
        images_placeholder = tf.placeholder("float",
            shape=[
                FLAGS.batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CHANNELS
            ], name="images_placeholder"
        )

        # take out images for viewing in tensorboard
        tf.image_summary("input_image", images_placeholder)

        # shift and scale images to be in tanh range
        shifted_images = (images_placeholder - IMAGE_SHIFT) * IMAGE_SCALE
        # encode images (batch_size, num_features)
        encoded = image_encoder_network(shifted_images, True)
        # decode images
        decoded = image_decoder_network(encoded, True)
        # shift and scale back into input space
        unshifted_images = (decoded / IMAGE_SCALE) + IMAGE_SHIFT
        # summary for reconstructed ims
        tf.image_summary("recons_image", unshifted_images)

        ##################################################
        # generate losses

        # reconstruction loss (minimize ||D(E(Im)) - Im||
        reconstruction_loss = tf.div(tf.nn.l2_loss(shifted_images - decoded), FLAGS.batch_size, name="R_loss")

        # feed real images and generated images into discriminator
        discriminator_logits_real = discriminator_network(shifted_images, True)
        discriminator_logits_fake = discriminator_network(decoded, True, reuse=True)
        # generate labels for discriminator batch
        ones = tf.ones((FLAGS.batch_size, 1))
        zeros = tf.zeros((FLAGS.batch_size, 1))

        # losses for discriminator
        # want to predict D(real) = 1, D(fake) = 0
        real_loss_v = tf.nn.sigmoid_cross_entropy_with_logits(discriminator_logits_real, ones)
        fake_loss_v = tf.nn.sigmoid_cross_entropy_with_logits(discriminator_logits_fake, zeros)
        real_loss = tf.reduce_mean(real_loss_v)
        fake_loss = tf.readuce_mean(fake_loss_v)
        # loss for discriminator training
        discriminator_loss = tf.div(real_loss + fake_loss, 2.0, name="D_loss")

        # generator loss
        # the generator wants to fool the discriminator
        # wants to maximize "fake_loss"
        generator_loss = tf.mul(fake_loss, -1.0, "G_loss")
        ##################################################


        tf.add_to_collection("losses", reconstruction_loss)

        total_loss = tf.add_n(tf.get_collection("losses"), name="total_loss")


        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = autoencoder_train(total_loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        if USE_CHECKPOINT:
            saver.restore(sess, CHECKPOINT_PATH)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                graph_def=sess.graph_def)

        # load data set
        #[train_set, valid_set, test_set, width, height, channels, num_labels] = data_handler.load_mnist_data()
        global train_set, valid_set, test_set
        train_data = train_set[0]
        # valid_data = valid_set[0]
        # test_data = test_set[0]

        # load first data
        arr = np.arange(len(train_data))
        generate_batch = lambda: train_data[np.random.choice(arr, FLAGS.batch_size)]


        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            # get the batch data
            image_batch = generate_batch()
            # load into feed dict
            fd = {
                images_placeholder: image_batch,
            }
            # run a step with summary generation
            if step % 100 == 0:
                _, loss_value, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict=fd)
                summary_writer.add_summary(summary_str, step)
            # run a step without summary generation
            else:
                _, loss_value = sess.run([train_op, total_loss], feed_dict=fd)

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))




            # Save the model checkpoint periodically.
            if step % 200 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
    if gfile.Exists(FLAGS.train_dir):
        gfile.DeleteRecursively(FLAGS.train_dir)
    gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
