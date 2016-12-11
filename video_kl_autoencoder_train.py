"""
Adversarial autoencoders are impossible to train. My desired video model may be too complex. Will sandbox convergence issues here. Simply trying to replicate results of http://arxiv.org/pdf/1511.05644v1.pdf

Will just load in moving mnist frames and attempt to constrain the distribution. Interested in the training difficulties of this simplified version of the problem
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import os.path
import time
import sys

from tensorflow.python.platform import gfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from layers import *
from utils import _add_loss_summaries, squared_difference, add_summary, moment_summary
import data_handler
from networks import *
import load_video_data
import losses

###################################################################################################
# Global variables and learning parameters
###################################################################################################
# The folder where checkpoints will be saved
TRAIN_DIR = '/tmp/frame_kl_autoencoder_test'


CHECKPOINT_PATH = '/tmp/video_kl_autoencoder_tiny_static_mean10/model.ckpt-999999'
LOG_DEVICE_PLACEMENT = False  # will display what devices are being used

# Learning parameters
BATCH_SIZE = 16
NUM_FRAMES = 16
NUM_EPOCHS_PER_DECAY = 10.0
NUM_ITERATIONS_PER_DECAY = 75000
MAX_STEPS = 1000000
VALID_ITERATIONS = 2000  # number of iterations to run validation set
NUM_VALID_BATCHES = 10
INITIAL_LEARNING_RATE = 0.0001     # Initial learning rate.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.


FULL_IMAGE_SIZE = (64, 64, 1)
IMAGE_SIZE = (64, 64, 1)
NUM_CHANNELS = FULL_IMAGE_SIZE[-1]

# Dataset generator
NUM_DIGITS=1
DATASET = data_handler.BouncingMNISTDataHandler(
    num_frames=NUM_FRAMES, batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE[0], num_digits=NUM_DIGITS
)

# compute the shift we need to do to align image values in [-1, 1]
MIN_IMAGE_VALUE = 0.0
MAX_IMAGE_VALUE = 1.0

IMAGE_VALUE_RANGE = MAX_IMAGE_VALUE - MIN_IMAGE_VALUE

IMAGE_SHIFT = (MAX_IMAGE_VALUE + MIN_IMAGE_VALUE) / 2.0
IMAGE_SCALE = 2.0 / IMAGE_VALUE_RANGE

# need to scale reconstruction loss to be in same range as other losses
RECONSTRUCTION_LOSS_SCALE = 1.0
DIFF_KL_LOSS_SCALE = 1.0
MEAN_KL_LOSS_SCALE = 1.0

NUM_ENCODING_FEATURES = 32

# Parameters of the distributions we are constraining the feautres to
ENCODING_MEAN_STD = 2.0
ENCODING_DIFF_STD=.1



LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

###################################################################################################


def autoencoder_train(diff_kl_loss, mean_kl_loss, reconstruction_loss, global_step):
    """
    Train adversarial autoencoder

    Create 2 optimizers and apply to generator and discriminator variables.

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
    loss_averages_op = _add_loss_summaries(
        [diff_kl_loss, mean_kl_loss, reconstruction_loss]
    )
    # Get total weight decay
    total_weight_loss = tf.add_n(tf.get_collection("losses"), name="total_weight_loss")

    # Get losses for each optimizer
    total_loss = reconstruction_loss +diff_kl_loss + mean_kl_loss + total_weight_loss

    # separate out the G and D variables
    trainable_vars = tf.trainable_variables()

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        # make opt
        opt = tf.train.AdamOptimizer(lr, beta1=.5, name="optimizer")
        # compute grads for both sets and both losses
        grads = opt.compute_gradients(total_loss, trainable_vars)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in trainable_vars:
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients for each optimizer
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + "/gradients", grad)

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

    # generate training op for reconstruction
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='R_train')

    return train_op


def train(testing=False, checkpoint=CHECKPOINT_PATH):
    """Train a convolutional autoencoder for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Setup placeholders
        # the videos that will be read by the encoder
        images_placeholder = tf.placeholder(
            "float",
            shape=[
                BATCH_SIZE * NUM_FRAMES, IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CHANNELS
            ], name="images_placeholder"
        )

        # take out images for viewing in tensorboard
        tf.image_summary("input_image", images_placeholder)

        # shift and scale images to be in tanh range
        shifted_images = (images_placeholder - IMAGE_SHIFT) * IMAGE_SCALE
        # encode images (batch_size, num_features)
        encoded = moving_mnist_image_encoder_network(
            shifted_images, True, base_filters=4, n_outputs=NUM_ENCODING_FEATURES
        )

        add_summary(encoded)
        moment_summary(encoded, "encoding")

        # compute the kl divergence loss
        time_diffs = []
        with tf.variable_scope("kl_loss") as scope:
            feats_for_each_video = tf.split(0, BATCH_SIZE, encoded)
            feat_means = [tf.reduce_mean(feats, 0, keep_dims=True) for feats in feats_for_each_video]
            for b in range(BATCH_SIZE):
                feats = feats_for_each_video[b]
                # slice from frame 0 to n-1 and from 1 to n then subtract to get
                # time differences
                feats_0_slice = tf.slice(feats, [0, 0], [NUM_FRAMES-1, -1])
                feats_1_slice = tf.slice(feats, [1, 0], [NUM_FRAMES-1, -1])
                time_differences = tf.identity(feats_1_slice - feats_0_slice, name="time_differences")
                time_diffs.append(time_differences)
            all_diffs = tf.concat(0, time_diffs, name="all_diffs")
            all_means = tf.concat(0, feat_means, name="all_means")
            moment_summary(all_means, "all_means")
            moment_summary(all_diffs, "all_diffs")

            # targets for time difference discriminator
            diff_mean, diff_var = tf.nn.moments(all_diffs, axes=[0])
            diff_kl_div = losses.NormalKLDivergence(diff_mean, diff_var, mu2=0.0, var2=ENCODING_DIFF_STD**2, scope_name="diff_kl_div")

            mean_mean, mean_var = tf.nn.moments(all_means, axes=[0])
            mean_kl_div = losses.NormalKLDivergence(mean_mean, mean_var, mu2=0.0, var2=ENCODING_MEAN_STD**2, scope_name="mean_kl_div")

            diff_kl_loss = tf.reduce_mean(
                (DIFF_KL_LOSS_SCALE * diff_kl_div),
                name="diff_kl_loss"
            )
            mean_kl_loss = tf.reduce_mean(
                (MEAN_KL_LOSS_SCALE * mean_kl_div),
                name="mean_kl_loss"
            )
        ##################################################

        # decoded images
        decoded = moving_mnist_image_decoder_network(encoded, True, base_filters=4)
        # shift and scale back into input space
        unshifted_images = (decoded / IMAGE_SCALE) + IMAGE_SHIFT
        # summary for reconstructed ims
        tf.image_summary("recons_image", unshifted_images)

        # generate losses
        # reconstruction loss (minimize ||D(E(Im)) - Im||
        with tf.variable_scope("R_loss") as scope:
            reconstruction_loss = tf.reduce_mean(
                RECONSTRUCTION_LOSS_SCALE * squared_difference(shifted_images, decoded),
                name=scope.name
            )

        ##################################################

        # Build a Graph that trains the model with one batch of examples and
        train_op = autoencoder_train(
            diff_kl_loss, mean_kl_loss, reconstruction_loss,
            global_step
        )

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())
        best_saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=LOG_DEVICE_PLACEMENT))
        sess.run(init)

        if checkpoint is not None:
            print("Loading Checkpoint {}".format(checkpoint))
            restorer = tf.train.Saver(tf.all_variables())
            restorer.restore(sess, CHECKPOINT_PATH)

        summary_writer = tf.train.SummaryWriter(TRAIN_DIR, graph_def=sess.graph_def)

        # setup data prefetching
        batch_generation_function = load_video_data.AsyncFunction(
            3, DATASET.GetBatch
        )
        generate_batch = lambda: batch_generation_function.run()
        generate_valid_batch = lambda: batch_generation_function.run()

        # get prefetch first batch
        batch_pref = generate_batch()
        if testing:
            return locals()

        best_validation_score = np.inf
        for step in xrange(MAX_STEPS):

            start_time = time.time()
            # get the batch data
            image_batch = np.concatenate(batch_pref.get()[0], 0)
            # prefetch the next batch
            batch_pref = generate_batch()

            # load into feed dict
            fd = {
                images_placeholder: image_batch,
            }

            # run a step with summary generation
            if step % 100 == 0:
                _, r_loss_value, diff_kl_loss_value, mean_kl_loss_value, summary_str = sess.run(
                    [
                        train_op,
                        reconstruction_loss, diff_kl_loss, mean_kl_loss,
                        summary_op
                    ],
                    feed_dict=fd
                )
                summary_writer.add_summary(summary_str, step)
            # run a step without summary generation
            else:
                _, r_loss_value, diff_kl_loss_value, mean_kl_loss_value = sess.run(
                    [
                        train_op,
                        reconstruction_loss, diff_kl_loss, mean_kl_loss
                    ],
                    feed_dict=fd
                )

            duration = time.time() - start_time

            nanned = any(np.isnan(l) for l in [r_loss_value, diff_kl_loss_value, mean_kl_loss_value])
            assert not nanned, 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = BATCH_SIZE
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, R_loss = %.2f, diff_KL_loss = %.2f, mean_KL_loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, r_loss_value, diff_kl_loss_value, mean_kl_loss_value, examples_per_sec, sec_per_batch))

            # Save the model checkpoint periodically.
            if step % VALID_ITERATIONS == 0 or (step + 1) == MAX_STEPS:
                print("Validating Model")
                # run the validation set
                total_valid_r_loss = 0
                total_valid_diff_kl_loss = 0
                total_valid_mean_kl_loss = 0
                e_std = 0
                d_std = 0
                for b in range(NUM_VALID_BATCHES):
                    image_batch = np.concatenate(batch_pref.get()[0], 0)
                    batch_pref = generate_batch()
                    v_fd = {
                        images_placeholder: image_batch,

                    }
                    [r_loss_value, diff_kl_loss_value, mean_kl_loss_value, enc, diffs] = sess.run(
                        [
                            reconstruction_loss, diff_kl_loss, mean_kl_loss,
                            encoded, all_diffs
                        ],
                        feed_dict=v_fd
                    )
                    total_valid_r_loss += r_loss_value
                    total_valid_diff_kl_loss += diff_kl_loss_value
                    total_valid_mean_kl_loss += mean_kl_loss_value
                    e_std += enc.std(axis=0).mean()
                    d_std += diffs.std(axis=0).mean()

                valid_r_loss = total_valid_r_loss / NUM_VALID_BATCHES
                valid_diff_kl_loss = total_valid_diff_kl_loss / NUM_VALID_BATCHES
                valid_mean_kl_loss = total_valid_mean_kl_loss / NUM_VALID_BATCHES
                valid_e_std = e_std / NUM_VALID_BATCHES
                valid_d_std = d_std / NUM_VALID_BATCHES
                print("Average Validation Reconstruction Score: {}".format(valid_r_loss))
                print("Average Validation diff KL Score: {}".format(valid_diff_kl_loss))
                print("Average Validation mean KL Score: {}".format(valid_mean_kl_loss))
                print("Average Validation Encoding STD: {}".format(valid_e_std))
                print("Average Validation Encoding Diff STD: {}".format(valid_d_std))
                if valid_r_loss < best_validation_score:
                    print("This is the best performing model!")
                    best_validation_score = valid_r_loss
                    checkpoint_path = os.path.join(TRAIN_DIR, "best_model.ckpt")
                    best_saver.save(sess, checkpoint_path)
                else:
                    checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
    if gfile.Exists(TRAIN_DIR):
        gfile.DeleteRecursively(TRAIN_DIR)
    gfile.MakeDirs(TRAIN_DIR)
    train()


if __name__ == '__main__':
    tf.app.run()
