"""
A binary to train captcha using a single GPU or cpu.
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
from tensorflow.models.rnn import rnn_cell
import tensorflow.models.rnn as rnn
import load_video_data
from utils import split_tensor_to_list

FLAGS = tf.app.flags.FLAGS
# A hack to stop conflicting batch sizes
try:
    print(FLAGS.batch_size)
except:
    tf.app.flags.DEFINE_integer(
        'batch_size', 8,
        """Number of videos to process in a batch."""
    )
tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/video_training5',
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
NUM_ITERATIONS_PER_DECAY = 1000
INITIAL_LEARNING_RATE = 0.001     # Initial learning rate.

TRAIN_FNAMES = os.listdir('/Users/grathwohl1/Code/ulfv_main/grathwohl/data/video_data/')
TRAIN_FNAMES = ['/Users/grathwohl1/Code/ulfv_main/grathwohl/data/video_data/' + f for f in TRAIN_FNAMES if ".mp4" in f]
TRAIN_FNAMES = np.array(TRAIN_FNAMES)
TEST_FNAMES = []

FULL_IMAGE_SHAPE = (3, 128, 128)
NUM_CHANNELS = FULL_IMAGE_SHAPE[0]
FRAME_STEP = 4
TOTAL_CLIP_LENGTH = 14
NUM_READING_FRAMES = 10
NUM_PREDICTING_FRAMES = 4
GLIMPSE_SIZE = (64, 64)
NUM_VIDEO_FEATURES = 1024

TIME_COST_SCALING_FACTOR = 6.0
LOCATION_COST_SCALING_FACTOR = 8.0
LOCATION_TARGET_SCALING_FACTOR = .01

LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

####### !!!!!HACK!!!!! ############
IMAGE_NETWORK_ALLOCATED = False


def image_processing_network(images, is_training, scope_name="im_feats"):
    """
    Builds the network that processes the glimpses

    Args:
        images: Images returned from distorted_inputs() or inputs().
        is_training: True if training, false if eval

    Returns:
        image features
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=IMAGE_NETWORK_ALLOCATED) as scope:
        # conv1
        n_filters_conv1 = 64
        conv1 = batch_normalized_conv_layer(
            images, "conv1", NUM_CHANNELS, n_filters_conv1,
            [3, 3], "MSFT", 0.004 if not IMAGE_NETWORK_ALLOCATED else 0.0, test=test
        )

        # pool1
        pool1 = tf.nn.max_pool(
            conv1, ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding='SAME', name='pool1'
        )

        # conv2
        n_filters_conv2 = 128
        conv2 = batch_normalized_conv_layer(
            pool1, "conv2", n_filters_conv1, n_filters_conv2,
            [3, 3], "MSFT", 0.004 if not IMAGE_NETWORK_ALLOCATED else 0.0, test=test
        )

        # pool2
        pool2 = tf.nn.max_pool(
            conv2, ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding='SAME', name='pool2'
        )

        # conv3
        n_filters_conv3 = 256
        conv3 = batch_normalized_conv_layer(
            pool2, "conv3", n_filters_conv2, n_filters_conv3,
            [3, 3], "MSFT", 0.004 if not IMAGE_NETWORK_ALLOCATED else 0.0, test=test
        )

        # pool3
        pool3 = tf.nn.max_pool(
            conv3, ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding='SAME', name='pool3'
        )

        # reshape pool3 to linear
        reshape, dim = reshape_conv_layer(pool3)

        # fc4
        n_outputs_fc_4 = 1024
        fc4 = batch_normalized_linear_layer(
            reshape, "fc4", dim, n_outputs_fc_4,
            .01, 0.004 if not IMAGE_NETWORK_ALLOCATED else 0.0, test=test
        )

        # local4
        n_outputs_fc_5 = 1024
        fc5 = batch_normalized_linear_layer(
            fc4, scope.name, n_outputs_fc_4, n_outputs_fc_5,
            .01, 0.004 if not IMAGE_NETWORK_ALLOCATED else 0.0, test=test
        )
        ##### !!!!!! HACK !!!!!!
        global IMAGE_NETWORK_ALLOCATED
        IMAGE_NETWORK_ALLOCATED = True
    return fc5


def image_feature_lstm(image_feature_batch, state_size, scope_name="LSTM"):
    """
    Takes a list of frame batches in shape: (batch_size, num_vis_features + 2), the two comes from the x,y position of the glimpse that we will read in and length is num_frames

    num_frames is the length of our video sequence

    outputs a batch of lstm features (MIGHT NEED TO JUST TAKE THE LAST OUTPUT)
    """
    assert type(image_feature_batch) == list
    # get image feature size for lstm input
    sizes = image_feature_batch[0].get_shape().as_list()
    batch_size = sizes[0]
    with tf.variable_scope(scope_name):
        lstm_cell = rnn_cell.BasicLSTMCell(state_size)
        initial_state = lstm_cell.zero_state(batch_size, tf.float32)
    outputs, states = rnn.rnn.rnn(lstm_cell, image_feature_batch, initial_state=initial_state)
    return outputs, states


def concat_glimps_locs(image_feature_batch, glimpse_loc_batch):
    """
    takes a list of frame batches of video features (length num_freatures and shape (batch_size, vis_features) and a list of frame batches of glimpse locations (length num_features and shape (batch_size, 2)) and concats the glimpse locations onto the image features returning a list of length num_frames of tensors of shape (batch_size, vis_features + 2)
    """
    concated = []
    for image_frame_batch, glimpse_frame_batch in zip(image_feature_batch, glimpse_loc_batch):
        feats_locs = tf.concat(1, [image_frame_batch, glimpse_frame_batch])
        concated.append(feats_locs)
    return concated


def compute_vis_features(video_batch, is_training):
    """
    Takes a video batch tensor (num_frames, batch_size, channels, height, width)
    and runs the conv net on each image producing a list of tensors of length num_frames where each tensor has size (num_frames, batch_size, num_vis_features)
    """
    # split frames across batches
    frames = split_tensor_to_list(video_batch)

    # now have array of tensors of shape (batch_size, channels, height, width)
    # apply the conv net
    # MAY NEED TO FIGURE OUT VARIABLE REUSE STUFF HERE
    #conv_features = [image_processing_network(frame_batch, is_training) for frame_batch in frames]
    conv_features = []
    conv_features.append(image_processing_network(frames[0], is_training))
    for frame_batch in frames[1:]:
        conv_features.append(image_processing_network(frame_batch, is_training))
    return conv_features


def compute_transformation_prediction_cost(video_feature_tensor, future_glimpse_conv_features, future_glimpse_locations, future_glimpse_times, time_scaling_factor, location_scaling_factor, is_training):
    """
    Takes batch of video features (from lstm), a batch of conv features from the future video frames of size (num_frames_in_future, batch_size, num_visual_features) and computes the error in predicting the transformation (in space and time) given the video features and the glimpse features

        time scaling factor is a factor that we multiply the cost of the time portion to balance the losses
    """
    assert len(future_glimpse_conv_features) == len(future_glimpse_locations)
    assert len(future_glimpse_locations) == len(future_glimpse_times)

    time_costs = []
    location_costs = []
    z = zip(future_glimpse_conv_features, future_glimpse_locations, future_glimpse_times)

    # for each visual feature, we need to concat it with the video feature
    for i, (conv_feature, location_gt, time_gt) in enumerate(z):
        reuse = (i > 0)
        with tf.variable_scope("outputs", reuse=reuse):
            input_feature_for_last_layer = tf.concat(
                1,
                [video_feature_tensor, conv_feature]
            )
            last_feature_size = input_feature_for_last_layer.get_shape().as_list()[1]
            # predicts the x, y location of future glimpse
            location_prediction = linear_layer(
                input_feature_for_last_layer, "l_pred",
                last_feature_size, 2,
                .01, 0.004 if not reuse else 0.0, use_nonlinearity=False
            )
            # predicts time location of the future glimpse
            time_prediction = linear_layer(
                input_feature_for_last_layer, "t_pred",
                last_feature_size, 1,
                .01, 0.004 if not reuse else 0.0, use_nonlinearity=False
            )
            # these are per frame costs across batches
            # REMOVE THE L2 LOSS WITH A SQUARE OP IT SUMS OVER EVERTYHIGN!!!!!!!
            location_cost_per_example_per_frame_per_dim = tf.square(location_prediction - (LOCATION_TARGET_SCALING_FACTOR * location_gt))
            location_cost_per_example_per_frame = tf.reduce_sum(location_cost_per_example_per_frame_per_dim, 1)

            time_cost_per_example_per_frame = tf.square(time_prediction - time_gt)

            # get mean across batch
            location_cost_per_frame = tf.reduce_mean(location_cost_per_example_per_frame)
            time_cost_per_frame = tf.reduce_mean(time_cost_per_example_per_frame)

            # add to cost lists
            time_costs.append(time_scaling_factor * time_cost_per_frame)
            location_costs.append(location_scaling_factor * location_cost_per_frame)


    total_time_cost = tf.add_n(time_costs, "time_loss")
    total_location_cost =  tf.add_n(location_costs, "location_loss")

    tf.add_to_collection("losses", total_time_cost)
    tf.add_to_collection("losses", total_location_cost)

    total_loss = tf.add_n(tf.get_collection("losses"), name="total_loss")
    return total_loss


def video_train(total_loss, global_step):
    """
    Train captcha model.

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
        pass
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad:
            print(var.name)
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of the batch norm variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

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
    """Train video model for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Setup placeholders
        # the videos that will be read by the encoder
        video_placeholder = tf.placeholder("float",
            shape=[
                NUM_READING_FRAMES, FLAGS.batch_size,
                GLIMPSE_SIZE[0], GLIMPSE_SIZE[1], NUM_CHANNELS
            ], name="video_placeholder"
        )
        # the locations that will be read by the encoder
        loc_placeholder = tf.placeholder("float",
            shape=[NUM_READING_FRAMES, FLAGS.batch_size, 2],
            name="loc_placeholder"
        )
        # the videos that will be used for prediction
        future_video_placeholder = tf.placeholder("float",
            shape=[
                NUM_PREDICTING_FRAMES, FLAGS.batch_size,
                GLIMPSE_SIZE[0], GLIMPSE_SIZE[1], NUM_CHANNELS
            ], name="future_video_placeholder"
        )
        # the locations that will be predicted by the output layer
        future_loc_placeholder = tf.placeholder("float",
            shape=[NUM_PREDICTING_FRAMES, FLAGS.batch_size, 2],
            name="future_loc_placeholder"
        )
        # the times that will be predicted by the output layer
        future_time_placeholder = tf.placeholder("float",
            shape=[NUM_PREDICTING_FRAMES, FLAGS.batch_size, 1],
            name="future_time_placeholder"
        )

        # take out images for viewing in tensorboard
        for frame_num in range(NUM_READING_FRAMES):
            read_im = video_placeholder[frame_num, :, :, :, :]
            tf.image_summary("input_image_{}".format(frame_num), read_im)
        for frame_num in range(NUM_PREDICTING_FRAMES):
            pred_im = future_video_placeholder[frame_num, :, :, :, :]
            tf.image_summary("pred_image_{}".format(frame_num), pred_im)

        # compute conv_features from the input videos
        video_conv_features = compute_vis_features(video_placeholder, True)
        # compute conv_features from future_videos
        future_video_conv_features = compute_vis_features(future_video_placeholder, True)

        # concat locs onto visual features
        loc_list = split_tensor_to_list(loc_placeholder)
        video_conv_features_plus_locs = concat_glimps_locs(video_conv_features, loc_list)

        # computes a batch of video features
        outputs, states = image_feature_lstm(video_conv_features_plus_locs, NUM_VIDEO_FEATURES)
        # lstm outputs the total outputs and the current states
        # we only want the last output to feed into our output layers
        video_features = outputs[-1]

        # split the future locs and future times
        future_loc_list = split_tensor_to_list(future_loc_placeholder)
        future_time_list = split_tensor_to_list(future_time_placeholder)
        # compute loss
        loss = compute_transformation_prediction_cost(video_features, future_video_conv_features, future_loc_list, future_time_list, TIME_COST_SCALING_FACTOR, LOCATION_COST_SCALING_FACTOR, True)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = video_train(loss, global_step)

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

        # create async video loading function
        video_loading_function = load_video_data.AsyncFunction(3, load_video_data.load_videos_and_generate_targets)
        # load first data
        generate_batch = lambda: video_loading_function.run(
            TRAIN_FNAMES, np.array([0] * len(TRAIN_FNAMES)),
            FLAGS.batch_size, FULL_IMAGE_SHAPE, FRAME_STEP,
            TOTAL_CLIP_LENGTH, NUM_READING_FRAMES, NUM_PREDICTING_FRAMES,
            GLIMPSE_SIZE, 2
        )

        r_val = generate_batch()

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            # get the batch data
            X, Y = r_val.get()
            # print to see how much time we are wasting waiting for data
            print("Data wait time: {}".format(time.time() - start_time))
            # unload the data
            reading_glimpse_ims, reading_glimpse_locs = X
            prediction_glimpse_ims, prediction_glimpse_locs, prediction_glimpse_times = Y

            # load into feed dict
            fd = {
                video_placeholder: reading_glimpse_ims,
                loc_placeholder: reading_glimpse_locs,
                future_video_placeholder: prediction_glimpse_ims,
                future_loc_placeholder: prediction_glimpse_locs,
                future_time_placeholder: prediction_glimpse_times
            }

            # asychronously fetch the next batch
            r_val = generate_batch()
            # run a step with summary generation
            if step % 1 == 0:
                _, loss_value, summary_str = sess.run([train_op, loss, summary_op], feed_dict=fd)
                summary_writer.add_summary(summary_str, step)
            # run a step without summary generation
            else:
                _, loss_value = sess.run([train_op, loss], feed_dict=fd)

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 1 == 0:
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
