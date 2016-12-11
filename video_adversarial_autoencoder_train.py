"""
Will train a simple convolutional autoencoder using l2 loss
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

###################################################################################################
# Global variables and learning parameters
###################################################################################################
# The folder where checkpoints will be saved
TRAIN_DIR = '/tmp/video_adversarial_autoencoder_test'


CHECKPOINT_PATH = '/tmp/video_adversarial_autoencoder_big_d_net/best_model.ckpt'
LOG_DEVICE_PLACEMENT = False # will display what devices are being used

# Learning parameters
BATCH_SIZE = 16
NUM_FRAMES = 16
NUM_EPOCHS_PER_DECAY = 10.0
NUM_ITERATIONS_PER_DECAY = 75000
MAX_STEPS = 1000000
VALID_ITERATIONS = 2000 # number of iterations to run validation set
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
GENERATOR_LOSS_SCALE = 1.0
DISCRIMINATOR_LOSS_SCALE = 1.0

NUM_STATIC_FEATURES = 32
NUM_TEMPORAL_FEATURES = 32

# Parameters of the distributions we are constraining the feautres to
STATIC_MEAN_VARIANCE = 1.0
TEMPORAL_VARIANCE = .1
STATIC_VARIANCE = .01


LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

###################################################################################################


def autoencoder_train(discriminator_loss, generator_loss, reconstruction_loss, global_step):
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
        [discriminator_loss, generator_loss, reconstruction_loss]
    )
    # Get total weight decay
    total_weight_loss = tf.add_n(tf.get_collection("losses"), name="total_weight_loss")

    # Get losses for each optimizer
    G_loss = generator_loss + total_weight_loss
    R_loss = reconstruction_loss + total_weight_loss
    D_loss = discriminator_loss + total_weight_loss

    # separate out the G and D variables
    trainable_vars = tf.trainable_variables()
    D_vars = [var for var in trainable_vars if "discriminator" in var.name]
    G_vars = [var for var in trainable_vars if not "discriminator" in var.name]

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        # optimizer for Discriminator
        D_opt = tf.train.AdamOptimizer(lr, beta1=.5, name="D_optimizer")
        D_grads = D_opt.compute_gradients(D_loss, D_vars)

        # optimizer for Reconstruction and generator
        R_opt = tf.train.AdamOptimizer(lr, name="R_optimizer")
        R_grads = R_opt.compute_gradients(R_loss+G_loss, G_vars)


    # Apply gradients.
    R_apply_gradient_op = R_opt.apply_gradients(R_grads, global_step=global_step)
    D_apply_gradient_op = D_opt.apply_gradients(D_grads, global_step=global_step)


    # Add histograms for trainable variables.
    for var in trainable_vars:
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients for each optimizer
    for grads, name in [(D_grads, '/D_gradients'), (R_grads, '/R_gradients')]:
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + name, grad)

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
    with tf.control_dependencies([R_apply_gradient_op, variables_averages_op]):
        R_train_op = tf.no_op(name='R_train')
    # generate training op for discriminator
    with tf.control_dependencies([D_apply_gradient_op, variables_averages_op]):
        D_train_op = tf.no_op(name='D_train')

    #return G_train_op, R_train_op, D_train_op
    return R_train_op, D_train_op

def train(testing=False, checkpoint=CHECKPOINT_PATH):
    """Train a convolutional autoencoder for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Setup placeholders
        # the videos that will be read by the encoder
        images_placeholder = tf.placeholder("float",
            shape=[
                BATCH_SIZE*NUM_FRAMES, IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CHANNELS
            ], name="images_placeholder"
        )
        static_mean_discriminator_input_gt = tf.placeholder(
            "float",
            shape=[BATCH_SIZE, NUM_STATIC_FEATURES],
            name="static_mean_discriminator_input_gt"
        )
        static_discriminator_input_gt = tf.placeholder(
            "float",
            shape=[BATCH_SIZE*(NUM_FRAMES-1), NUM_STATIC_FEATURES],
            name="static_discriminator_input_gt"
        )
        temporal_discriminator_input_gt = tf.placeholder(
            "float",
            shape=[BATCH_SIZE*(NUM_FRAMES-1), NUM_TEMPORAL_FEATURES],
            name="static_discriminator_input_gt"
        )

        # take out images for viewing in tensorboard
        tf.image_summary("input_image", images_placeholder)

        # shift and scale images to be in tanh range
        shifted_images = (images_placeholder - IMAGE_SHIFT) * IMAGE_SCALE
        # encode images (batch_size, num_features)
        encoded = moving_mnist_image_encoder_network(shifted_images, True, base_filters=8)

        with tf.variable_scope("adversarial_variables") as scope:
            # split out each feature set
            static_feats = tf.slice(
                encoded, [0, 0],
                [BATCH_SIZE*NUM_FRAMES, NUM_STATIC_FEATURES],
                name="static_feats"
            )
            add_summary(static_feats)
            temporal_feats = tf.slice(
                encoded, [0, NUM_STATIC_FEATURES],
                [BATCH_SIZE*NUM_FRAMES, NUM_TEMPORAL_FEATURES],
                name="temporal_feats"
            )
            add_summary(temporal_feats)
            # get feats for each video
            static_feats_for_each_video = tf.split(0, BATCH_SIZE, static_feats)
            temporal_feats_for_each_video = tf.split(0, BATCH_SIZE, temporal_feats)
            # get means of static feats for each video
            static_feats_mean_for_each_video = [tf.reduce_mean(feats, 0, keep_dims=True) for feats in static_feats_for_each_video]
            # shove means into tensor of size [BATCH_SIZE, NUM_STATIC_FEATURES] so can
            # be fed to discriminator to encourage means to be drawn from a normal
            # distribution
            static_mean_discriminator_input_gen = tf.concat(0, static_feats_mean_for_each_video, name="static_mean_discriminator_input_gen")
            assert static_mean_discriminator_input_gen.get_shape().as_list() == [BATCH_SIZE, NUM_STATIC_FEATURES]
            add_summary(static_mean_discriminator_input_gen)
            # get differences between neighboring frames
            # this is to encourage [f_s(t) - f_s(t-1)] to be drawn from almost a delta
            # distribution and to encourage [f_t(t) - f_t(t-1)] to be drawn from a zero
            # mean gaussian to enforce that they change smoothly with time
            static_diffs = []
            temporal_diffs = []
            for b in range(BATCH_SIZE):
                sfeats = static_feats_for_each_video[b]
                tfeats = temporal_feats_for_each_video[b]
                # slice from frame 0 to n-1 and from 1 to n then subtract to get
                # time differences
                static_0_slice = tf.slice(sfeats, [0, 0], [NUM_FRAMES-1, -1])
                static_1_slice = tf.slice(sfeats, [1, 0], [NUM_FRAMES-1, -1])

                temporal_0_slice = tf.slice(tfeats, [0, 0], [NUM_FRAMES-1, -1])
                temporal_1_slice = tf.slice(tfeats, [1, 0], [NUM_FRAMES-1, -1])

                assert static_0_slice.get_shape().as_list() == [NUM_FRAMES-1, NUM_STATIC_FEATURES]
                assert temporal_0_slice.get_shape().as_list() == [NUM_FRAMES-1, NUM_TEMPORAL_FEATURES]

                static_differences = tf.identity(static_1_slice - static_0_slice, name="static_differences")
                static_diffs.append(static_differences)
                temporal_differences = tf.identity(temporal_1_slice - temporal_0_slice, name="temporal_differences")
                temporal_diffs.append(temporal_differences)
            # stack these together to generate input tensors for discriminators
            static_discriminator_input_gen = tf.concat(0, static_diffs, name="static_discriminator_input")
            add_summary(static_discriminator_input_gen)
            assert static_discriminator_input_gen.get_shape().as_list() == [BATCH_SIZE*(NUM_FRAMES-1), NUM_STATIC_FEATURES]
            temporal_discriminator_input_gen = tf.concat(0, temporal_diffs, name="temporal_discriminator_input")
            add_summary(temporal_discriminator_input_gen)
            assert temporal_discriminator_input_gen.get_shape().as_list() == [BATCH_SIZE*(NUM_FRAMES-1), NUM_TEMPORAL_FEATURES]

            # track the moments of the tensors whose distribution we are trying to control
            with tf.variable_scope("encoding_moments") as scope:
                moment_summary(static_discriminator_input_gen, "static")
                moment_summary(static_mean_discriminator_input_gen, "static_mean")
                moment_summary(temporal_discriminator_input_gen, "temporal")

        # compute input variables for D network
        with tf.variable_scope("discriminator_inputs") as scope:
            static_batch_mean_discriminator_input = tf.concat(
                0,
                [static_mean_discriminator_input_gen, static_mean_discriminator_input_gt]
            )
            static_discriminator_input = tf.concat(
                0,
                [static_discriminator_input_gen, static_discriminator_input_gt]
            )
            temporal_discriminator_input = tf.concat(
                0,
                [temporal_discriminator_input_gen, temporal_discriminator_input_gt]
            )
        # compute labels for D network
        with tf.variable_scope("discriminator_labels") as scope:
            # targets for time difference discriminator
            ones = tf.ones((BATCH_SIZE*(NUM_FRAMES-1), 1))
            zeros = tf.zeros((BATCH_SIZE*(NUM_FRAMES-1), 1))
            # targets for mean discriminator
            ones_mean = tf.ones((BATCH_SIZE, 1))
            zeros_mean = tf.zeros((BATCH_SIZE, 1))

            discriminator_labels = tf.concat(0, [ones, zeros], name="discrim_labels")
            generator_labels = tf.concat(0, [zeros, ones], name="gen_labels")
            discriminator_labels_mean = tf.concat(0, [ones_mean, zeros_mean], name="discrim_labels_mean")
            generator_labels_mean = tf.concat(0, [zeros_mean, ones_mean], name="gen_labels_mean")
        ##################################################

        # decode images
        conced = tf.concat(1, [static_feats, temporal_feats])
        decoded = moving_mnist_image_decoder_network(conced, True, base_filters=8)
        # shift and scale back into input space
        unshifted_images = (decoded / IMAGE_SCALE) + IMAGE_SHIFT
        # summary for reconstructed ims
        tf.image_summary("recons_image", unshifted_images)

        # generate losses
        # reconstruction loss (minimize ||D(E(Im)) - Im||
        with tf.variable_scope("R_loss") as scope:
            reconstruction_loss = tf.reduce_mean(RECONSTRUCTION_LOSS_SCALE * squared_difference(shifted_images, decoded), name=scope.name)

        # feed real images and generated images into discriminator
        static_discriminator_batch = tf.concat(
            0, [static_discriminator_input_gt, static_discriminator_input_gen]
        )
        temporal_discriminator_batch = tf.concat(
            0, [temporal_discriminator_input_gt, temporal_discriminator_input_gen]
        )
        static_mean_discriminator_batch = tf.concat(
            0, [static_mean_discriminator_input_gt, static_mean_discriminator_input_gen]
        )
        # get logits
        static_discriminator_logits = discriminator_network(
            static_discriminator_batch, True, "static_discriminator"
        )
        temporal_discriminator_logits = discriminator_network(
            temporal_discriminator_batch, True, "temporal_discriminator"
        )
        static_mean_discriminator_logits = discriminator_network(
            static_mean_discriminator_batch, True, "static_mean_discriminator"
        )

        # loss for discriminator training
        with tf.variable_scope("D_loss") as scope:
            # want to predict D(real) = 1, D(fake) = 0
            static_discriminator_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                static_discriminator_logits, discriminator_labels
            )
            static_mean_discriminator_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                static_mean_discriminator_logits, discriminator_labels_mean
            )
            temporal_discriminator_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                temporal_discriminator_logits, discriminator_labels
            )
            sdl = tf.reduce_mean(static_discriminator_losses)
            tf.scalar_summary("D_loss/static_D_loss", sdl)
            tdl = tf.reduce_mean(temporal_discriminator_losses)
            tf.scalar_summary("D_loss/temporal_D_loss", tdl)
            smdl = tf.reduce_mean(static_mean_discriminator_losses)
            tf.scalar_summary("D_loss/static_mean_D_loss", smdl)
            d_l = sdl + tdl + smdl
            discriminator_loss = tf.identity(
                DISCRIMINATOR_LOSS_SCALE * d_l,
                name="D_loss"
            )

        # generator loss
        # the generator wants to fool the discriminator
        with tf.variable_scope("G_loss") as scope:
            # want to predict D(fake) = 1
            static_generator_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                static_discriminator_logits, generator_labels
            )
            # losses for real and decoded images, need to take back half
            _, static_generator_losses_v = tf.split(0, 2, static_generator_losses)
            static_mean_generator_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                static_mean_discriminator_logits, generator_labels_mean
            )
            # losses for real and decoded images, need to take back half
            _, static_mean_generator_losses_v = tf.split(0, 2, static_mean_generator_losses)
            temporal_generator_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                temporal_discriminator_logits, generator_labels
            )
            # losses for real and decoded images, need to take back half
            _, temporal_generator_losses_v = tf.split(0, 2, temporal_generator_losses)

            sgl = tf.reduce_mean(static_generator_losses_v)
            tf.scalar_summary("G_loss/static_G_loss", sgl)
            tgl = tf.reduce_mean(temporal_generator_losses_v)
            tf.scalar_summary("G_loss/temporal_G_loss", tgl)
            smgl = tf.reduce_mean(static_mean_generator_losses_v)
            tf.scalar_summary("G_loss/static_mean_G_loss", smgl)
            g_l = sgl + tgl + smgl
            generator_loss = tf.identity(
                GENERATOR_LOSS_SCALE * g_l,
                name="G_loss"
            )

        ##################################################

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters for each aspect of the loss
        # G_train_op, R_train_op, D_train_op = autoencoder_train(
        #     discriminator_loss, generator_loss, reconstruction_loss,
        #     global_step
        # )
        R_train_op, D_train_op = autoencoder_train(
            discriminator_loss, generator_loss, reconstruction_loss,
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
            # generate noise for adversarial losses
            static_mean_gt = np.random.normal(0, STATIC_MEAN_VARIANCE**.5, [BATCH_SIZE, NUM_STATIC_FEATURES])
            static_gt = np.random.normal(0, STATIC_VARIANCE**.5, [BATCH_SIZE*(NUM_FRAMES-1), NUM_STATIC_FEATURES])
            temporal_gt = np.random.normal(0, TEMPORAL_VARIANCE**.5, [BATCH_SIZE*(NUM_FRAMES-1), NUM_TEMPORAL_FEATURES])
            # load into feed dict
            fd = {
                images_placeholder: image_batch,
                static_discriminator_input_gt: static_gt,
                temporal_discriminator_input_gt: temporal_gt,
                static_mean_discriminator_input_gt: static_mean_gt
            }

            train_op = R_train_op if step % 5 == 0 else D_train_op

            # run a step with summary generation
            if step % 100 == 0:
                _, g_loss_value, r_loss_value, d_loss_value, summary_str = sess.run(
                    [train_op, generator_loss, reconstruction_loss, discriminator_loss, summary_op],
                    feed_dict=fd
                )
                summary_writer.add_summary(summary_str, step)
            # run a step without summary generation
            else:
                _, g_loss_value, r_loss_value, d_loss_value = sess.run(
                    [train_op, generator_loss, reconstruction_loss, discriminator_loss],
                    feed_dict=fd
                )

            duration = time.time() - start_time

            nanned = any(np.isnan(l) for l in [g_loss_value, r_loss_value, d_loss_value])
            assert not nanned, 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = BATCH_SIZE
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, G_loss = %.2f, R_loss = %.2f, D_loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, g_loss_value, r_loss_value, d_loss_value,
                                     examples_per_sec, sec_per_batch))

            # Save the model checkpoint periodically.
            if step % VALID_ITERATIONS == 0 or (step + 1) == MAX_STEPS:
                print("Validating Model")
                # run the validation set
                total_valid_r_loss = 0
                total_valid_g_loss = 0
                total_valid_d_loss = 0
                s_vars = 0
                t_vars = 0
                m_vars = 0
                for b in range(NUM_VALID_BATCHES):
                    image_batch = np.concatenate(batch_pref.get()[0], 0)
                    batch_pref = generate_batch()
                    static_mean_gt = np.random.normal(0, STATIC_MEAN_VARIANCE, [BATCH_SIZE, NUM_STATIC_FEATURES])
                    static_gt = np.random.normal(0, STATIC_VARIANCE, [BATCH_SIZE*(NUM_FRAMES-1), NUM_STATIC_FEATURES])
                    temporal_gt = np.random.normal(0, TEMPORAL_VARIANCE, [BATCH_SIZE*(NUM_FRAMES-1), NUM_TEMPORAL_FEATURES])
                    v_fd = {
                        images_placeholder: image_batch,
                        static_discriminator_input_gt: static_gt,
                        temporal_discriminator_input_gt: temporal_gt,
                        static_mean_discriminator_input_gt: static_mean_gt
                    }
                    [r_loss_value, g_loss_value, d_loss_value, s_diffs, t_diffs, means] = sess.run([reconstruction_loss, generator_loss, discriminator_loss, static_discriminator_input_gen, temporal_discriminator_input_gen, static_mean_discriminator_input_gen], feed_dict=v_fd)
                    total_valid_r_loss += r_loss_value
                    total_valid_g_loss += g_loss_value
                    total_valid_d_loss += d_loss_value
                    s_vars += s_diffs.var(axis=0).mean()
                    t_vars += t_diffs.var(axis=0).mean()
                    m_vars += means.var(axis=0).mean()

                valid_r_loss = total_valid_r_loss / NUM_VALID_BATCHES
                valid_g_loss = total_valid_g_loss / NUM_VALID_BATCHES
                valid_d_loss = total_valid_d_loss / NUM_VALID_BATCHES
                valid_s_vars = s_vars / NUM_VALID_BATCHES
                valid_t_vars = t_vars / NUM_VALID_BATCHES
                valid_m_vars = m_vars / NUM_VALID_BATCHES
                print("Average Validation Reconstruction Score: {}".format(valid_r_loss))
                print("Average Validation Generator Score: {}".format(valid_g_loss))
                print("Average Validation Discriminator Score: {}".format(valid_d_loss))
                print("Average Validation Static Vars: {}".format(valid_s_vars))
                print("Average Validation Temporal Vars: {}".format(valid_t_vars))
                print("Average Validation Static Mean Vars: {}".format(valid_m_vars))
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
