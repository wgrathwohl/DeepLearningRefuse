"""
Tests an adversarial model by training on the cifar10 dataset
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
from utils import _add_loss_summaries, squared_difference
import data_handler
from networks import *
from utils import _activation_summary, up_down_weighter, delayed_down_weighter, constant_weighter, clip_grads
import load_video_data
import cv2
import cv
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier

###################################################################################################
# Global variables and learning parameters
###################################################################################################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_gpus", 1, "number of gpus to use")
tf.app.flags.DEFINE_string("dataset", "cifar10", "dataset to use")
tf.app.flags.DEFINE_string("train_dir", "/tmp/gan_transfer", "summary directory")
tf.app.flags.DEFINE_string("checkpoint_path", '', "the path to load")
tf.app.flags.DEFINE_string("feature_method", "discriminator", "whether to use features from the D or G network")
tf.app.flags.DEFINE_bool("save_ims", False, "whether or not to save the generated images")
tf.app.flags.DEFINE_string("curriculum", "none", "curriculum type")
tf.app.flags.DEFINE_integer("batch_size", 1024, "size of batches")
tf.app.flags.DEFINE_integer("iters", 1000, "number of iterations to run features")
assert FLAGS.dataset in ["cifar10"], "invalid dataset"

if FLAGS.dataset == "cifar10":
    [train_set, valid_set, test_set, width, height, channels, num_labels] = data_handler.load_cifar_data()
    IMAGE_SIZE = (height, width, channels)
    NUM_CHANNELS = channels
    GENERATOR_NETWORK = imagenet_generator_network_large
    if FLAGS.curriculum  == "none":
        DISCRIMINATOR_NETWORK = imagenet_discriminator_network_large
    else:
        1/0
        DISCRIMINATOR_NETWORK = imagenet_discriminator_network_multiout
    NUM_LOSSES = 4
else:
    assert False, "Invalid dataset"

# Learning parameters
NUM_ENCODING_FEATURES = 100
INITIAL_LEARNING_RATE = 0.1     # Initial learning rate.
MOVING_AVERAGE_DECAY = 0.9
NUM_ITERATIONS_PER_DECAY = 1000



# compute the shift we need to do to align image values in [-1, 1]
MIN_IMAGE_VALUE = train_set[0].min()
MAX_IMAGE_VALUE = train_set[0].max()

IMAGE_VALUE_RANGE = MAX_IMAGE_VALUE - MIN_IMAGE_VALUE

IMAGE_SHIFT = (MAX_IMAGE_VALUE + MIN_IMAGE_VALUE) / 2.0
IMAGE_SCALE = 2.0 / IMAGE_VALUE_RANGE

MIN_OUT_IMAGE_VALUE = 0.
MAX_OUT_IMAGE_VALUE = 255.
OUT_IMAGE_VALUE_RANGE = MAX_OUT_IMAGE_VALUE - MIN_OUT_IMAGE_VALUE

OUT_IMAGE_SHIFT = (MAX_OUT_IMAGE_VALUE + MIN_OUT_IMAGE_VALUE) / 2.0
OUT_IMAGE_SCALE = 2.0 / OUT_IMAGE_VALUE_RANGE



# need to scale reconstruction loss to be in same range as other losses
GENERATOR_LOSS_SCALE = 1.0
DISCRIMINATOR_LOSS_SCALE = 1.0

LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

def get_generator_features(data, batch_ind, decay_iters=1000):
    """
    computes generator features
    """
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Setup placeholders
        # the images that will be judged the the discriminators
        images_placeholder = tf.placeholder("float",
            shape=[
                FLAGS.batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CHANNELS
            ], name="images_placeholder"
        )

        # shift and scale images to be in tanh range
        shifted_images = tf.identity((images_placeholder - IMAGE_SHIFT) * IMAGE_SCALE, name="shifted_images")

        initializer = tf.truncated_normal_initializer(mean=.5, stddev=.25)
        encodings = tf.get_variable("image_encodings", (FLAGS.batch_size, NUM_ENCODING_FEATURES), initializer=initializer, trainable=True)
        generated_ims = GENERATOR_NETWORK(encodings, True, scope_name="generator")
        unshifted_generated_ims = (generated_ims / OUT_IMAGE_SCALE) + OUT_IMAGE_SHIFT
        tf.image_summary("real_images", images_placeholder)
        tf.image_summary("generated_ims", unshifted_generated_ims)
        loss = tf.reduce_mean(squared_difference(generated_ims, shifted_images), name="l2_loss")
        tf.scalar_summary("loss", loss)
        tf.histogram_summary("encodings", encodings)
        # set up learning parameters
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(
            INITIAL_LEARNING_RATE,
            global_step,
            decay_iters,
            LEARNING_RATE_DECAY_FACTOR,
            staircase=True
        )
        opt = tf.train.AdamOptimizer(lr, name="optimizer")
        # optimize only with respect to the encodings
        grads = opt.compute_gradients(loss, var_list=[encodings])
        for var, grad in grads:
            print(var.name, grad.name)
            tf.histogram_summary(var.op.name, grad)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        with tf.control_dependencies([apply_gradient_op]):
            train_op = tf.no_op(name='train')

        summary_op = tf.merge_all_summaries()
        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=False))
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                            graph_def=sess.graph_def)
        sess.run(init)
        restore_vars = [var for var in tf.trainable_variables() if "generator" in var.name]
        restorer = tf.train.Saver(restore_vars)
        restorer.restore(sess, FLAGS.checkpoint_path)

        real_ims = data[batch_ind*FLAGS.batch_size:(batch_ind+1)*FLAGS.batch_size]
        for step in range(FLAGS.iters):
            start_time = time.time()
            fd = {
                images_placeholder: real_ims
            }
            if step % 100 == 0:
                _, summary_str, current_loss = sess.run([train_op, summary_op, loss], feed_dict=fd)
                summary_writer.add_summary(summary_str, step)
            else:
                _, current_loss = sess.run([train_op, loss], feed_dict=fd)
            if step % 10 == 0:
                print("Step {} took {} with loss {}".format(step, time.time() - start_time, current_loss))
            if step % 100 == 0:
                gen_ims, _, current_loss = sess.run([unshifted_generated_ims, train_op, loss], feed_dict=fd)
                if FLAGS.save_ims:
                    save_image_batch(gen_ims, "/tmp/gen_ims/", 0)
                    save_image_batch(255. * real_ims, "/tmp/real_ims", 0)
        # get final encodings
        final_encodings, = sess.run([encodings], feed_dict=fd)
        return final_encodings


def save_image_batch(ims, folder, start):
    for i, im in enumerate(ims):
        im_f = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(folder, "im_{}.jpg".format(start + i)), im_f)



def train(training=True):
    """Train a convolutional autoencoder for a number of steps."""

    if FLAGS.feature_method == "discriminator":
        with tf.Graph().as_default():
            global_step = tf.Variable(0, trainable=False)

            # Setup placeholders
            # the images that will be judged the the discriminators
            images_placeholder = tf.placeholder("float",
                shape=[
                    FLAGS.batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CHANNELS
                ], name="images_placeholder"
            )

            # shift and scale images to be in tanh range
            shifted_images = tf.identity((images_placeholder - IMAGE_SHIFT) * IMAGE_SCALE, name="shifted_images")
            """
            Get features from discriminator via pooling
            """
            with tf.variable_scope("discriminator_logits"):
                features = DISCRIMINATOR_NETWORK(shifted_images, False, return_activations=True)
            pooled = [pool_to_shape(feats, (4, 4), "pooled", pool_type="max") for feats in features[:-1]]
            reshaped = [tf.reshape(pool, [FLAGS.batch_size, -1]) for pool in pooled]
            conced = tf.concat(1, reshaped)
            # Build an initialization operation to run below.
            init = tf.initialize_all_variables()

            # Start running operations on the Graph.
            sess = tf.Session(config=tf.ConfigProto(
                log_device_placement=False))
            sess.run(init)

            restorer = tf.train.Saver(tf.all_variables())
            restorer.restore(sess, FLAGS.checkpoint_path)



            # load data set
            global train_set, valid_set, test_set

            train_feats = []
            valid_feats = []
            test_feats = []

            for tset, feats in zip([train_set, valid_set, test_set], [train_feats, valid_feats, test_feats]):
                max_steps = len(tset[0]) // FLAGS.batch_size
                for step in xrange(max_steps):
                    start_time = time.time()
                    # get the batch data
                    image_batch = tset[0][step*FLAGS.batch_size:(step+1)*FLAGS.batch_size]


                    # load into feed dict
                    fd = {
                        images_placeholder: image_batch,
                    }

                    transfer_feats, = sess.run([conced], feed_dict=fd)
                    feats.extend(transfer_feats)

                    duration = time.time() - start_time

                    if step % 10 == 0:
                        num_examples_per_step = FLAGS.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)

                        format_str = ('step %d (%.1f examples/sec; %.3f sec/batch)')
                        print(format_str % (step, examples_per_sec, sec_per_batch))

            train_feats = np.array(train_feats)
            valid_feats = np.array(valid_feats)
            test_feats = np.array(test_feats)

            print("Training SVM")
            #clf = LinearSVC(penalty="l2")
            for c in [100, 10, 1, .1, .01, 001, .0001]:
                print("C = {}".format(c))
                clf = SGDClassifier(penalty="l2", alpha=c)
                clf.fit(train_feats, train_set[1][:len(train_feats)])
                print("score: {}".format(clf.score(test_feats, test_set[1][:len(test_feats)])))

    elif FLAGS.feature_method == "generator":
        """
        Get features from generator by gradient descent
        """
        global train_set, valid_set, test_set
        all_encodings = []
        all_labels = []
        for dset, dset_name in [(train_set, "train"), (valid_set, "valid"), (test_set, "test")]:
            print("Running {} set".format(dset_name))
            num_ims = len(dset[0])
            num_batches = num_ims // FLAGS.batch_size
            encodings_list = [get_generator_features(dset[0], i) for i in range(num_batches)]
            encodings = np.concatenate(encodings_list)
            labels = dset[1][:num_batches*FLAGS.batch_size]
            print(encodings.shape, labels.shape)
            np.save(FLAGS.train_dir+"/{}_encodings".format(dset_name), encodings)
            np.save(FLAGS.train_dir+"/{}_labels".format(dset_name), labels)
            all_encodings.append(encodings)
            all_labels.append(labels)

        train_encodings = all_encodings[0]
        train_labels = all_labels[0]
        valid_encodings = all_encodings[1]
        valid_labels = all_labels[1]
        test_encodings = all_encodings[2]
        test_labels = all_labels[2]

        print("Training SVM")
        svm = LinearSVC(penalty="l2")
        svm.fit(train_encodings, train_labels)
        print("score: {}".format(svm.score(test_encodings, test_labels)))







    else:
        assert False, "invalid method"

def main(argv=None):
    if gfile.Exists(FLAGS.train_dir):
        gfile.DeleteRecursively(FLAGS.train_dir)
    gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
