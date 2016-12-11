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
from utils import _add_loss_summaries, squared_difference, abs_difference, _create_variable, moment_summary
import data_handler
import load_video_data
from networks import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 64*16, "number of seperate models in a batch")
tf.app.flags.DEFINE_string('train_dir', None, "the training directory")
tf.app.flags.DEFINE_integer('max_steps', 30000, "total number of steps to run")
LOG_DEVICE_PLACEMENT = False

tf.app.flags.DEFINE_string("checkpoint_path", None, "if used will restore checkpointed model")
tf.app.flags.DEFINE_integer("iterations_per_decay", 10000, "num iterations to decay")
tf.app.flags.DEFINE_integer("iterations_per_valid", 1000, "num iterations to validation")
tf.app.flags.DEFINE_float('initial_learning_rate', .001, "initial learning rate")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1, "learning rate decay factor")
tf.app.flags.DEFINE_integer("num_features", 4, "number of encoder features")
tf.app.flags.DEFINE_float('kl_weight', 1.0, "weight on kl divergence term")
tf.app.flags.DEFINE_integer("im_size", 64, "size of frames")
tf.app.flags.DEFINE_string("dataset", "bouncing_MNIST", "which dataset")
tf.app.flags.DEFINE_string("data_folder", None, "data folder")

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
            self.train_dataset = data_handler.ImageDataHandler(
                FLAGS.batch_size,
                os.path.join(FLAGS.data_folder, "train"),
                FLAGS.im_size, True, greyscale=True
            )
            self.test_dataset = data_handler.ImageDataHandler(
                FLAGS.batch_size,
                os.path.join(FLAGS.data_folder, "test"),
                FLAGS.im_size, True, greyscale=True
            )
        else:
            assert False

    def GET_BATCH(self):
        if self.type == "bouncing_MNIST":
            image_batch = np.concatenate(batch_pref.get()[0], 0)
            # prefetch the next batch
            self.batch_pref = self.generate_batch()
            return image_batch
        elif self.type == "chairs":
            return self.train_dataset.get_result()

    def GET_TEST_BATCH(self):
        if self.type == "bouncing_MNIST":
            return np.concatenate(self.dataset.GetTestBatch()[0], 0)
        elif self.type == "chairs":
            return self.test_dataset.get_result()


    def DisplayData(self, vdm):
        return self.dataset.DisplayData(vdm)

# Dataset generator
dataset = data_handler.BouncingMNISTDataHandler(
    num_frames=1, batch_size=FLAGS.batch_size,
    image_size=FLAGS.im_size, num_digits=1
)
dataset = DATASET(FLAGS.dataset)

ENCODER_NETWORK = video_variational_encoder_large
DECODER_NETWORK = video_variational_decoder_large



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

    return tf.reduce_sum(negative_log_prob) / FLAGS.batch_size

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

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries([total_loss])

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
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


def train():
    """Train a convolutional autoencoder for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Setup placeholders
        # voxels for static feature learning batches
        images_placeholder = tf.placeholder("float",
            shape=[
                FLAGS.batch_size,
                FLAGS.im_size, FLAGS.im_size, 1
            ], name="images_placeholder"
        )
        # take out images for viewing in tensorboard
        tf.image_summary("images", images_placeholder)

        encoding_mu, encoding_log_sigma_sq = ENCODER_NETWORK(images_placeholder, train=True, n_outputs=FLAGS.num_features)

        eps = tf.random_normal((FLAGS.batch_size, FLAGS.num_features), 0, 1, dtype=tf.float32)
        tf.histogram_summary("eps", eps)
        z = tf.add(encoding_mu, tf.mul(tf.exp(encoding_log_sigma_sq / 2.0), eps))
        moment_summary(z, "z")
        tf.histogram_summary("z", z)


        decoded_images = DECODER_NETWORK(z, train=True)
        tf.image_summary("decoded_images", tf.sigmoid(decoded_images))

        r_loss = bernouli_neg_log_likelihood(images_placeholder, decoded_images)
        tf.add_to_collection("losses", r_loss)

        kl_loss = FLAGS.kl_weight * tf.reduce_mean(
            -0.5 * tf.reduce_sum(1 + encoding_log_sigma_sq - tf.square(encoding_mu) - tf.exp(encoding_log_sigma_sq), 1),
            name="kl_loss"
        )
        tf.add_to_collection("losses", kl_loss)

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
        global dataset

        best_valid_score = np.inf
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            image_batch = dataset.GET_BATCH()

            # load into feed dict
            fd = {
                images_placeholder: image_batch,
            }

            # run a step with summary generation
            if step % 100 == 0:
                _, r_loss_value, kl_loss_value, summary_str = sess.run([train_op, r_loss, kl_loss, summary_op], feed_dict=fd)
                summary_writer.add_summary(summary_str, step)
            # run a step without summary generation
            else:
                _, r_loss_value, kl_loss_value = sess.run([train_op, r_loss, kl_loss], feed_dict=fd)

            duration = time.time() - start_time

            assert not (np.isnan(r_loss_value) or np.isnan(kl_loss_value)), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, r loss = %2f, kl loss = %2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
                print (format_str % (datetime.now(), step, r_loss_value,
                                     kl_loss_value, examples_per_sec, sec_per_batch))


            # Save the model checkpoint periodically.
            if step % FLAGS.iterations_per_valid == 0 or (step + 1) == FLAGS.max_steps:
                n_test_batches = 100
                score = 0.0
                for b in range(n_test_batches):
                    batch = dataset.GET_TEST_BATCH()
                    fd = {
                        images_placeholder: batch
                    }
                    l, = sess.run([total_loss], feed_dict=fd)

                    score += l
                score /= n_test_batches
                print("Valid loss =  {}".format(score))
                if score < best_valid_score:
                    print(score, best_valid_score)
                    best_valid_score = score
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'best-model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
    if gfile.Exists(FLAGS.train_dir):
        gfile.DeleteRecursively(FLAGS.train_dir)
    gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
