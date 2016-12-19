"""
Video Heirarchical Variational Autoencoder

Trains an variational autoencoder to take in a video, V = [I_1, I_2, ..., I_T]
and will produce a set of encodings
H = [(s_1, t_1), (s_2, t_2), ..., (s_T, t_T)] where the s and t variables obey the prior:

t_0 ~ N(0, sigma_t0_sqr)
t_i ~ N(s_{i-1}, sigma_t_sqr)

s_i ~ N(s_V, sigma_s_sqr)
s_V ~ N(0, sigma_s0_sqr)

by Will Grathwohl
"""
from datetime import datetime
import os.path
import time

from tensorflow.python.platform import gfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import data_handler
import load_video_data
from networks import *
import magic_init
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 64, "number of videos in a batch")
tf.app.flags.DEFINE_string('train_dir', None, "the training directory")
tf.app.flags.DEFINE_integer('max_steps', 10000, "total number of steps to run")
LOG_DEVICE_PLACEMENT = False
tf.app.flags.DEFINE_string("checkpoint_path", None, "if used will restore checkpointed model")
tf.app.flags.DEFINE_integer("iterations_per_decay", 100000, "num iterations to decay")
tf.app.flags.DEFINE_integer("iterations_per_valid", 1000, "num iterations to validation")
tf.app.flags.DEFINE_integer("iterations_per_display", 10, "num iterations to decay")
tf.app.flags.DEFINE_integer("iterations_per_summary", 100, "num iterations to decay")
tf.app.flags.DEFINE_float('initial_learning_rate', .001, "initial learning rate")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1, "learning rate decay factor")
tf.app.flags.DEFINE_integer("num_features", 32, "number of encoder features")
tf.app.flags.DEFINE_integer("num_frames", 16, "number of frames in a video")
tf.app.flags.DEFINE_integer("im_size", 64, "size of frames")
tf.app.flags.DEFINE_float('mu0', 0.0, "prior static feature mean")
tf.app.flags.DEFINE_float('sigma_s_sqr', 0.01, "prior static feature variance")
tf.app.flags.DEFINE_float('sigma_s0_sqr', 1.0, "prior static feature mean variance")
tf.app.flags.DEFINE_float('sigma_t_sqr', 0.01, "prior temporal feature variance")
tf.app.flags.DEFINE_float('sigma_t0_sqr', 1.0, "prior temporal feature mean variance")
tf.app.flags.DEFINE_float('kl_weight', 1.0, "weight on kl divergence term")
tf.app.flags.DEFINE_boolean("temporal_only", False, "if true, only use temporal kl loss and don't factor features, this is slow-feaure-like benchmark")
tf.app.flags.DEFINE_string("dataset", "bouncing_MNIST", "the dataset")
tf.app.flags.DEFINE_string("data_folder", None, "the folder with the data")
assert FLAGS.dataset in ("bouncing_MNIST", "chairs")

ENCODER_NETWORK = video_variational_encoder_large
DECODER_NETWORK = video_variational_decoder_large

class DATASET:
    def __init__(self, type):
        self.type = type
        if type == "bouncing_MNIST":
            self.dataset = data_handler.BouncingMNISTDataHandler(
            num_frames=FLAGS.num_frames, batch_size=FLAGS.batch_size,
            image_size=FLAGS.im_size, num_digits=1
            )
            # setup data prefetching
            self.batch_generation_function = load_video_data.AsyncFunction(
                3, self.dataset.GetBatch
            )
            self.generate_batch = lambda: self.batch_generation_function.run()
            self.batch_pref = self.generate_batch()
        elif type == "chairs":
            self.train_dataset = data_handler.ImageSequenceDataHandler(
                FLAGS.batch_size,
                os.path.join(FLAGS.data_folder, "train"),
                FLAGS.im_size, True, reverse=True, greyscale=True
            )
            self.test_dataset = data_handler.ImageSequenceDataHandler(
                FLAGS.batch_size,
                os.path.join(FLAGS.data_folder, "test"),
                FLAGS.im_size, True, reverse=True, greyscale=True
            )
        else:
            assert False

    def GET_BATCH(self):
        if self.type == "bouncing_MNIST":
            video_batch = self.batch_pref.get()[0]
            # prefetch the next batch
            self.batch_pref = self.generate_batch()
            return video_batch
        elif self.type == "chairs":
            return self.train_dataset.get_result()

    def GET_TEST_BATCH(self):
        if self.type == "bouncing_MNIST":
            return self.dataset.GetTestBatch()[0]
        elif self.type == "chairs":
            return self.test_dataset.get_result()


    def DisplayData(self, vdm):
        return self.dataset.DisplayData(vdm)

def Ex_log_q(mu, log_sigma_sqr):
    """
    mu: tensor [batch_size, n_frames, n_features]
    log_sigma_sqr: tensor [batch_size, n_frames, n_features]

    returns integral(log(q(x))q(x)dx)
    """

    batch_size, N, F = mu.get_shape().as_list()
    c1 = -.5 * N * F * np.log(2. * np.pi)
    c2 = -.5 * tf.reduce_sum(1. + log_sigma_sqr, reduction_indices=[1, 2])

    sequence_entropy = c1 +  c2
    return tf.reduce_sum(sequence_entropy)

def Ex_log_p_s(mu, log_sigma_sqr, sigma_s_sqr, eps=1e-10):
    """
    mu: tensor [batch_size, n_frames, n_features]
    log_sigma_sqr: tensor [batch_size, n_frames, n_features]

    returns integral(log(p(x))q(x)dx)
    """
    batch_size, N, F = mu.get_shape().as_list()

    ex_h_sqr_hat = tf.reduce_mean(tf.square(mu) + tf.exp(log_sigma_sqr), reduction_indices=[1])

    mean_sum = 0.
    for i in range(N-1):
        mu_i = mu[:, i, :]
        mu_i_plus_1 = mu[:, i+1:, :]
        sum_mu_i_plus_1 = tf.reduce_sum(mu_i_plus_1, reduction_indices=[1])
        prod = mu_i * sum_mu_i_plus_1
        mean_sum = mean_sum + prod

    ex_h_hat_sqr = (ex_h_sqr_hat / N) + (2. / (N**2)) * mean_sum

    v0 = -1. * N / (2. * sigma_s_sqr) * tf.reduce_sum(ex_h_sqr_hat, reduction_indices=[1])
    v1 = -1. * N * ((1. / (2. * (sigma_s_sqr + N))) - (1. / (2. * sigma_s_sqr))) * tf.reduce_sum(
        ex_h_hat_sqr, reduction_indices=[1]
    )

    c1 = -.5 * (N - 1) * np.log(2. * np.pi)
    c2 = .5 * (np.log(sigma_s_sqr) - np.log(N)) - .5 * N * np.log(sigma_s_sqr)
    c3 = -.5 * np.log(2. * np.pi * (sigma_s_sqr + 1))
    logC = c1 + c2 + c3

    return tf.reduce_sum(v0 + v1 + F * logC)


def D_kl_s(mu, log_sigma_sqr, sigma_s_sqr):
    e_log_q = Ex_log_q(mu, log_sigma_sqr)
    e_log_p = Ex_log_p_s(mu, log_sigma_sqr, sigma_s_sqr)

    return e_log_q - e_log_p

def D_kl_t(mu, log_sigma_sqr, sigma_t_sqr, eps=1e-10):
    """
    mu: tensor [batch_size, n_frames, n_features]
    log_sigma_sqr: tensor [batch_size, n_frames, n_features]

    returns integral(log(p(x)) - log(q(x))q(x)dx)
    """
    batch_size, N, F = mu.get_shape().as_list()

    mu_0 = mu[:, 0, :]
    mu_delta = mu[:, 1:, :] - mu[:, :-1, :]
    sigma_sqr_0 = tf.exp(log_sigma_sqr[:, 0, :])
    sigma_sqr_delta = tf.exp(log_sigma_sqr[:, 1:, :]) + tf.exp(log_sigma_sqr[:, :-1, :])

    c0 = -.5 * F * np.log(2. * np.pi)
    c1 = -.5 * (N - 1) * F * np.log(2. * np.pi * sigma_t_sqr)
    v0 = -.5 * tf.reduce_sum(tf.square(mu_0) + sigma_sqr_0, reduction_indices=[1])
    v_mu = - 1. / (2. * sigma_t_sqr) * tf.reduce_sum(
        tf.square(mu_delta),
        reduction_indices=[1, 2]
    )
    v_sigma = - 1. / (2. * sigma_t_sqr) * tf.reduce_sum(sigma_sqr_delta, reduction_indices=[1, 2])

    e_log_p = tf.reduce_sum(c0 + v0 + c1 + v_mu + v_sigma)
    e_log_q = Ex_log_q(mu, log_sigma_sqr)

    return e_log_q - e_log_p





def bernouli_neg_log_likelihood(x, mu):
    """
    Our "reconstruction" loss: -log(pr(x | z))

    x: [batch_size, height, width, n_channels]
    mu: [batch_size, height, width, 1]

    Returns the log prob for the entire batch
    """

    # unwrap all to vectors
    x_v = tf.reshape(x, [-1] + [np.prod(x.get_shape().as_list()[1:])])
    mu_v = tf.reshape(mu, [-1] + [np.prod(mu.get_shape().as_list()[1:])])

    negative_log_prob = tf.nn.sigmoid_cross_entropy_with_logits(mu_v, x_v)

    return tf.reduce_sum(negative_log_prob)

def frames_2_video(t):
    """
    reshapes a tensor of shape [batch_size * num_frames, num_features] into a tensor of shape
        [batch_size, num_frames, num_features]
    """
    video_t = tf.reshape(t, [-1, FLAGS.num_frames, t.get_shape().as_list()[1]])
    return video_t

def split_videos(t):
    """
    reshapes a tensor [batch_size, num_frames, *dims] to batch_size tensors [num_frames, *dims]
    """
    assert t.get_shape().as_list()[1] == FLAGS.num_frames
    ts = tf.split(0, FLAGS.batch_size, t)
    tss = [tf.squeeze(_ts, squeeze_dims=[0]) for _ts in ts]
    return tss

def autoencoder_train(total_loss, global_step):
    """
    Train modelnet autoencoder

    Create an optimizer and apply to all trainable variables. Add moving
    average for all batch norm variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
          processed.
    Returns:
        train_op: op for training.
    """
    # Variables that affect learning rate.
    decay_steps = FLAGS.iterations_per_decay

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(
        FLAGS.initial_learning_rate,
        global_step,
        decay_steps,
        FLAGS.learning_rate_decay_factor,
        staircase=True
    )

    tf.scalar_summary('learning_rate', lr)

    # Compute gradients.
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(total_loss)


    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    return apply_gradient_op


def train(test=False):
    """Train a convolutional autoencoder for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Setup placeholders
        # voxels for static feature learning batches
        videos_placeholder = tf.placeholder("float",
            shape=[
                FLAGS.batch_size,
                FLAGS.num_frames,
                FLAGS.im_size, FLAGS.im_size, 1
            ], name="videos_placeholder"
        )
        # take out images for viewing in tensorboard
        tf.image_summary("images", videos_placeholder[0, :, :, :])

        # reshape video placeholder into one big batch
        all_frames = tf.reshape(videos_placeholder, [-1, FLAGS.im_size, FLAGS.im_size, 1])
        # encode all frames
        frames_mu, frames_log_sigma_sq = ENCODER_NETWORK(
            all_frames, n_outputs=FLAGS.num_features, train=not test
        )
        # draw samples for each feature
        eps = tf.random_normal(tf.shape(frames_mu), 0, 1, dtype=tf.float32)
        # convert them to samples from q(z | x) with z = mu + sigma * e
        frames_z = frames_mu + tf.exp(frames_log_sigma_sq / 2.0) * eps
        tf.histogram_summary("z", frames_z)
        # produce the decoder distribution
        dec_mu = DECODER_NETWORK(frames_z, train=not test)
        videos_dec_mu = tf.reshape(
            dec_mu, [-1, FLAGS.num_frames, FLAGS.im_size, FLAGS.im_size, 1]
        )

        # take out decoded images for viewing in tensorboard
        summary_videos = tf.sigmoid(videos_dec_mu)
        tf.image_summary("decoded_images", summary_videos[0, :, :, :])

        if FLAGS.temporal_only:
            videos_mu_t = frames_2_video(frames_mu)
            videos_log_sigma_sqr_t = frames_2_video(frames_log_sigma_sq)

            temporal_D_kl = D_kl_t(videos_mu_t, videos_log_sigma_sqr_t, FLAGS.sigma_t_sqr)
            tf.scalar_summary("temporal_kl_div", temporal_D_kl)

            negative_D_kl = -1. * temporal_D_kl / FLAGS.batch_size / FLAGS.num_frames

        else:
            # split feature distributions and samples into [static, temporal]
            frames_mu_s, frames_mu_t = tf.split(1, 2, frames_mu)
            frames_log_sigma_sq_s, frames_log_sigma_sq_t = tf.split(1, 2, frames_log_sigma_sq)

            # reshape all model outputs and activation back into video tensor form
            videos_mu_s = frames_2_video(frames_mu_s)
            videos_mu_t = frames_2_video(frames_mu_t)
            videos_log_sigma_sqr_s = frames_2_video(frames_log_sigma_sq_s)
            videos_log_sigma_sqr_t = frames_2_video(frames_log_sigma_sq_t)
            #videos_z_s = frames_2_video(frames_z_s)
            #videos_z_t = frames_2_video(frames_z_t)


            static_D_kl = D_kl_s(videos_mu_s, videos_log_sigma_sqr_s, FLAGS.sigma_s_sqr)
            tf.scalar_summary("static_kl_div", static_D_kl)
            temporal_D_kl = D_kl_t(videos_mu_t, videos_log_sigma_sqr_t, FLAGS.sigma_t_sqr)
            tf.scalar_summary("temporal_kl_div", temporal_D_kl)

            negative_D_kl = -1. * (static_D_kl + temporal_D_kl) / FLAGS.batch_size / FLAGS.num_frames
            tf.scalar_summary("total_kl_div", static_D_kl + temporal_D_kl)

        nl_p_x = bernouli_neg_log_likelihood(all_frames, dec_mu)
        Ex_p_x = -1 * nl_p_x / FLAGS.batch_size / FLAGS.num_frames
        tf.scalar_summary("log_p_x", Ex_p_x)

        lower_bound = Ex_p_x + FLAGS.kl_weight * negative_D_kl
        loss = -1. * lower_bound
        tf.scalar_summary("loss", loss)
        tf.add_to_collection("losses", loss)

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
            log_device_placement=LOG_DEVICE_PLACEMENT))
        sess.run(init)

        if FLAGS.checkpoint_path is not None:
            saver.restore(sess, FLAGS.checkpoint_path)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                graph_def=sess.graph_def)

        # load data set
        dataset = DATASET(FLAGS.dataset)

        if test:
            return locals(), globals()


        best_valid_score = np.inf

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            video_batch = dataset.GET_BATCH()

            # load into feed dict
            fd = {
                videos_placeholder: video_batch
            }

            # run a step with summary generation
            if step % FLAGS.iterations_per_summary == 0:
                _, loss_value, kl_div_value, p_x_value, summary_str = sess.run(
                    [train_op, total_loss, -1.*negative_D_kl, -1.*Ex_p_x, summary_op],
                    feed_dict=fd
                )
                summary_writer.add_summary(summary_str, step)
            # run a step without summary generation
            else:
                _, loss_value, kl_div_value, p_x_value = sess.run(
                    [train_op, total_loss, -1.*negative_D_kl, -1.*Ex_p_x],
                    feed_dict=fd
                )

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % FLAGS.iterations_per_display == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, loss = %2f, kl div = %2f, neg_log_p_x = %2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                     kl_div_value, p_x_value, examples_per_sec, sec_per_batch))

            # Save the model checkpoint periodically.
            if step > 0 and step % FLAGS.iterations_per_valid == 0 or (step + 1) == FLAGS.max_steps:
                n_test_batches = 100
                score = 0.0
                for b in range(n_test_batches):
                    video_batch = dataset.GET_TEST_BATCH()
                    fd = {
                        videos_placeholder: video_batch
                    }
                    l, = sess.run([total_loss], feed_dict=fd)

                    score += l
                score /= n_test_batches
                print("Valid loss =  {}".format(score))
                if score < best_valid_score:
                    best_valid_score = score
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'best-model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                if (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'last-model.ckpt')
                    saver.save(sess, checkpoint_path)


def main(argv=None):
    if gfile.Exists(FLAGS.train_dir):
        gfile.DeleteRecursively(FLAGS.train_dir)
    gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()




