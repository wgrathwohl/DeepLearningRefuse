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
from utils import _add_loss_summaries
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
tf.app.flags.DEFINE_string("curriculum", "none", "type of curriculum to use")
tf.app.flags.DEFINE_string("dataset", "lsun", "dataset to use")
assert FLAGS.dataset in ["lsun", "imagenet"], "invalid dataset"
assert FLAGS.curriculum in ["none", "constant", "updown"], "invalid curriculum"

# The folder where checkpoints will be saved
TRAIN_DIR = '/tmp/test_transfer_validation_curriculum_updown'

USE_CHECKPOINT = False
CHECKPOINT_PATH = ''
LOG_DEVICE_PLACEMENT = False # will display what devices are being used

# Learning parameters
BATCH_SIZE = 128
NUM_ENCODING_FEATURES = 100
NUM_TRAINING_EPOCHS = 15
NUM_EPOCHS_PER_DECAY = 10.0

# load data set
if FLAGS.dataset == "lsun":
    DATA_PATH = "data/bedroom_train_lmdb"
    dataset = data_handler.LSUN_data_handler(DATA_PATH, BATCH_SIZE * FLAGS.num_gpus)
    IMAGE_SIZE = (64, 64, 3)
    NUM_CHANNELS = IMAGE_SIZE[-1]
    GENERATOR_NETWORK = lsun_generator_network
    if FLAGS.curriculum  == "none":
        DISCRIMINATOR_NETWORK = lsun_discriminator_network
    else:
        DISCRIMINATOR_NETWORK = lsun_discriminator_network_multiout
    NUM_LOSSES = 5
elif FLAGS.dataset == "imagenet":
    DATA_PATH = "data/imagenet_resized"
    dataset = data_handler.imagenet_data_handler(BATCH_SIZE * FLAGS.num_gpus)
    validation_dataset = data_handler.load_cifar_data()
    IMAGE_SIZE = (32, 32, 3)
    NUM_CHANNELS = IMAGE_SIZE[-1]
    GENERATOR_NETWORK = imagenet_generator_network_large
    if FLAGS.curriculum  == "none":
        DISCRIMINATOR_NETWORK = imagenet_discriminator_network_large
    else:
        DISCRIMINATOR_NETWORK = imagenet_discriminator_network_large_multiout
    NUM_LOSSES = 4

# the number of steps that the dqiscriminator takes before the generator takes a set
NUM_DISCRIMINATOR_STEPS = 1
NUM_STEPS_PER_EPOCH = int((dataset.num_ims // BATCH_SIZE) / FLAGS.num_gpus)
print("Dataset has {} ims".format(dataset.num_ims))
print("{} steps in an epoch".format(NUM_STEPS_PER_EPOCH))
NUM_ITERATIONS_PER_DECAY = 1000000 #NUM_STEPS_PER_EPOCH * NUM_EPOCHS_PER_DECAY


if FLAGS.curriculum == "constant":
    weights = [1.0 for i in range(NUM_LOSSES)]
    G_WEIGHTER = constant_weighter(weights)
    D_WEIGHTER = constant_weighter(weights)
elif FLAGS.curriculum == "updown":
    G_WEIGHTER = up_down_weighter(NUM_STEPS_PER_EPOCH // 10, NUM_LOSSES)
    D_WEIGHTER = delayed_down_weighter(NUM_STEPS_PER_EPOCH // 10, NUM_LOSSES)

VALID_ITERATIONS = NUM_STEPS_PER_EPOCH // 10
VALID_IMAGES = 10000 # number of images to save for each vaidation set



INITIAL_LEARNING_RATE = 0.0002     # Initial learning rate.
MOVING_AVERAGE_DECAY = 0.9




# compute the shift we need to do to align image values in [-1, 1]
MIN_IMAGE_VALUE = 0.
MAX_IMAGE_VALUE = 255.

IMAGE_VALUE_RANGE = MAX_IMAGE_VALUE - MIN_IMAGE_VALUE

IMAGE_SHIFT = (MAX_IMAGE_VALUE + MIN_IMAGE_VALUE) / 2.0
IMAGE_SCALE = 2.0 / IMAGE_VALUE_RANGE

# need to scale reconstruction loss to be in same range as other losses
GENERATOR_LOSS_SCALE = 1.0
DISCRIMINATOR_LOSS_SCALE = 1.0

LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

def gan_grads(generator_loss, discriminator_loss, G_opt, D_opt, global_step):
    """
    Computes gradients for a gan
    """
    # Get total weight decay
    weight_decay_vars = [l for l in tf.get_collection("losses") if "weight_loss" in l.name]
    assert len(weight_decay_vars) > 0
    total_weight_loss = tf.add_n(weight_decay_vars, name="total_weight_loss")

    # create losses the the optimizers will optimize
    D_loss = discriminator_loss + total_weight_loss
    G_loss = generator_loss + total_weight_loss

    # separate out the G and D variables
    trainable_vars = tf.trainable_variables()
    D_vars = [var for var in trainable_vars if "discriminator" in var.name]
    G_vars = [var for var in trainable_vars if "generator" in var.name]

    # Add histograms for trainable variables.
    for var in trainable_vars:
        with tf.device("/cpu:0"):
            tf.histogram_summary(var.op.name, var)

    # Compute gradients.
    # optimizer for Discriminator
    D_grads = D_opt.compute_gradients(D_loss, D_vars)

    # optimizer for Generator
    G_grads = G_opt.compute_gradients(G_loss, G_vars)


    return G_grads, D_grads

def average_gradients(grads):
    """
    Takes a list of lists of grad-var pairs for each gpu
    and returns a list of grad-var pairs of the gradients averaged across the devices
    """
    average_grads = []
    # for each variable
    for grad_and_vars in zip(*grads):
        grads = []
        # for each device
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def apply_gan_grads(global_step, G_grads, G_losses, G_opt, D_grads, D_losses, D_opt):
    mean_D_loss = tf.identity(tf.add_n(D_losses) / len(D_losses), "mean_D_loss")
    mean_G_loss = tf.identity(tf.add_n(G_losses) / len(G_losses), "mean_G_loss")

    tf.add_to_collection("losses", mean_G_loss)
    tf.add_to_collection("losses", mean_D_loss)
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries([])

    D_apply_gradient_op = D_opt.apply_gradients(D_grads, global_step=global_step)
    G_apply_gradient_op = G_opt.apply_gradients(G_grads, global_step=global_step)

    # Add histograms for gradients for each optimizer
    for grads, name in [(D_grads, '/D_gradients'), (G_grads, '/G_gradients')]:
        for grad, var in grads:
            if grad is not None:
                with tf.device("/cpu:0"):
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

    # generate training op for generator
    with tf.control_dependencies([G_apply_gradient_op, variables_averages_op, loss_averages_op]):
        G_train_op = tf.no_op(name='G_train')
    # generate training op for discriminator
    with tf.control_dependencies([D_apply_gradient_op, variables_averages_op, loss_averages_op]):
        D_train_op = tf.no_op(name='D_train')

    return G_train_op, mean_G_loss, D_train_op, mean_D_loss

def shape(v):
    print(v.get_shape().as_list(), v.name)

def gan_loss(reuse, device, training):
    """
    creates a gan loss given the images and encodings
    """
    name_tail = device.split(":")[-1]
    # Setup placeholders
    # the images that will be judged the the discriminators
    with tf.device("/cpu:0"):
        images_placeholder = tf.placeholder("float",
            shape=[
                BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CHANNELS
            ], name="images_placeholder_{}".format(name_tail)
        )
        # the random vectors that will be used by the generator to make images
        encoding_placeholder = tf.placeholder("float",
            shape=[BATCH_SIZE, NUM_ENCODING_FEATURES],
            name="encoding_placeholder_{}".format(name_tail)
        )
    # shift and scale images to be in tanh range
    shifted_images = (images_placeholder - IMAGE_SHIFT) * IMAGE_SCALE
    # generate images
    generated_ims = GENERATOR_NETWORK(encoding_placeholder, training, scope_name="generator", reuse=reuse)

    # shift and scale back into input space
    unshifted_generated_ims = (generated_ims / IMAGE_SCALE) + IMAGE_SHIFT

    if not reuse:
        with tf.device("/cpu:0"):
            # take out images for viewing in tensorboard if this is the first gpu
            tf.image_summary("real_images", images_placeholder)
            tf.image_summary("generated_ims", unshifted_generated_ims)

    # get logits for each discriminator
    with tf.variable_scope("discriminator_logits"):
        # feed real images and generated images into discriminator
        D_activations = DISCRIMINATOR_NETWORK(shifted_images, training, reuse=reuse, return_activations=True)
        pooled = [pool_to_shape(feats, (4, 4), "pooled", pool_type="max") for feats in D_activations[:-1]]
        reshaped = [tf.reshape(pool, [BATCH_SIZE, -1]) for pool in pooled]
        transfer_feats = tf.concat(1, reshaped)
        D_logits_real = D_activations[-1]
        D_logits_fake = DISCRIMINATOR_NETWORK(generated_ims, training, reuse=True)
        if not reuse:
            with tf.device("/cpu:0"):
                tf.histogram_summary("D_logits_real", D_logits_real)
                tf.histogram_summary("D_logits_fake", D_logits_fake)

    # losses for training
    with tf.variable_scope("losses") as scope:
        # want to predict D(real) = 1, D(fake) = 0
        real_D_loss_v = tf.nn.sigmoid_cross_entropy_with_logits(D_logits_real, tf.ones_like(D_logits_real))
        fake_D_loss_v = tf.nn.sigmoid_cross_entropy_with_logits(D_logits_fake, tf.zeros_like(D_logits_fake))
        real_d_loss = tf.reduce_mean(real_D_loss_v)
        fake_d_loss = tf.reduce_mean(fake_D_loss_v)
        D_loss = tf.identity(
           DISCRIMINATOR_LOSS_SCALE * (real_d_loss + fake_d_loss),
           name="D_loss_{}".format(name_tail)
        )
        gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(D_logits_fake, tf.ones_like(D_logits_fake))
        G_loss = tf.identity(
           GENERATOR_LOSS_SCALE * tf.reduce_mean(gen_loss),
           name="G_loss_{}".format(name_tail)
        )
    return images_placeholder, encoding_placeholder, G_loss, D_loss, unshifted_generated_ims, transfer_feats

def gan_loss_smooth_curriculum(reuse, device, training):
    """
    creates a gan loss given the images and encodings
    """
    name_tail = device.split(":")[-1]
    # Setup placeholders
    # the images that will be judged the the discriminators
    with tf.device("/cpu:0"):
        images_placeholder = tf.placeholder("float",
            shape=[
                BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CHANNELS
            ], name="images_placeholder_{}".format(name_tail)
        )
        # the random vectors that will be used by the generator to make images
        encoding_placeholder = tf.placeholder("float",
            shape=[BATCH_SIZE, NUM_ENCODING_FEATURES],
            name="encoding_placeholder_{}".format(name_tail)
        )
                # the annealed weights for the losses
        g_weights_placeholder = tf.placeholder("float",
            shape=NUM_LOSSES,
            name="g_weights_placeholder"
        )
        d_weights_placeholder = tf.placeholder("float",
            shape=NUM_LOSSES,
            name="d_weights_placeholder"
        )
        # add summaries for loss weightings
        if not reuse:
            with tf.variable_scope("loss_weights") as scope:
                for i in range(NUM_LOSSES):
                    tf.scalar_summary("G_weights/G_weight_{}".format(i), g_weights_placeholder[i])
                    tf.scalar_summary("D_weights/D_weight_{}".format(i), d_weights_placeholder[i])
    # shift and scale images to be in tanh range
    shifted_images = (images_placeholder - IMAGE_SHIFT) * IMAGE_SCALE
    # generate images
    generated_ims = GENERATOR_NETWORK(encoding_placeholder, training, scope_name="generator", reuse=reuse)

    # shift and scale back into input space
    unshifted_generated_ims = (generated_ims / IMAGE_SCALE) + IMAGE_SHIFT

    if not reuse:
        with tf.device("/cpu:0"):
            # take out images for viewing in tensorboard if this is the first gpu
            tf.image_summary("real_images", images_placeholder)
            tf.image_summary("generated_ims", unshifted_generated_ims)

    # get logits for each discriminator
    with tf.variable_scope("discriminator_logits"):
        # feed real images and generated images into discriminator
        D_activations, D_logits_reals = DISCRIMINATOR_NETWORK(shifted_images, training, reuse=reuse, return_activations=True)
        pooled = [pool_to_shape(feats, (4, 4), "pooled", pool_type="max") for feats in D_activations]
        reshaped = [tf.reshape(pool, [BATCH_SIZE, -1]) for pool in pooled]
        transfer_feats = tf.concat(1, reshaped)
        print(transfer_feats.get_shape().as_list())
        D_logits_fakes = DISCRIMINATOR_NETWORK(generated_ims, training, reuse=True)
        if not reuse:
            with tf.device("/cpu:0"):
                tf.histogram_summary("D_logits_real", D_logits_reals[-1])
                tf.histogram_summary("D_logits_fake", D_logits_fakes[-1])

    # losses for training
    with tf.variable_scope("losses") as scope:
        # want to predict D(real) = 1, D(fake) = 0
        D_losses = []
        G_losses = []
        for i, (D_logits_real, D_logits_fake) in enumerate(zip(D_logits_reals, D_logits_fakes)):
            real_D_loss_v = tf.nn.sigmoid_cross_entropy_with_logits(D_logits_real, tf.ones_like(D_logits_real))
            fake_D_loss_v = tf.nn.sigmoid_cross_entropy_with_logits(D_logits_fake, tf.zeros_like(D_logits_fake))
            real_d_loss = tf.reduce_mean(real_D_loss_v)
            fake_d_loss = tf.reduce_mean(fake_D_loss_v)
            D_loss = tf.identity(
               DISCRIMINATOR_LOSS_SCALE * (real_d_loss + fake_d_loss),
               name="D_loss_{}_{}".format(name_tail, i)
            )
            D_losses.append(D_loss)
            tf.add_to_collection("losses", D_loss)
            gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(D_logits_fake, tf.ones_like(D_logits_fake))
            G_loss = tf.identity(
               GENERATOR_LOSS_SCALE * tf.reduce_mean(gen_loss),
               name="G_loss_{}_{}".format(name_tail, i)
            )
            G_losses.append(G_loss)
            tf.add_to_collection("losses", G_loss)
        D_loss = tf.add_n(
                [d_weights_placeholder[i] * D_losses[i] for i in range(NUM_LOSSES)],
                name="Weighted_D_loss"
            )
        G_loss = tf.add_n(
            [g_weights_placeholder[i] * G_losses[i] for i in range(NUM_LOSSES)],
            name="Weighted_G_loss"
        )
    return images_placeholder, encoding_placeholder, G_loss, D_loss, unshifted_generated_ims, g_weights_placeholder, d_weights_placeholder, transfer_feats


def save_image_batch(ims, folder, start):
    for i, im in enumerate(ims):
        im_f = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(folder, "im_{}.jpg".format(start + i)), im_f)



def train(training=True):
    """Train a convolutional autoencoder for a number of steps."""
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        global_step = tf.Variable(0, trainable=False)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(
            INITIAL_LEARNING_RATE,
            global_step,
            NUM_ITERATIONS_PER_DECAY,
            LEARNING_RATE_DECAY_FACTOR,
            staircase=True
        )
        tf.scalar_summary('learning_rate', lr)

        # create optimizers
        D_opt = tf.train.AdamOptimizer(lr, name="D_optimizer", beta1=.5)
        G_opt = tf.train.AdamOptimizer(lr, name="G_optimizer", beta1=.5)
        im_placeholders = []
        enc_placeholders = []
        G_losses = []
        D_losses = []
        G_grads_list = []
        D_grads_list = []
        gen_ims_vars = []
        g_weights_placeholders = []
        d_weights_placeholders = []
        trans_feats = []
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                if FLAGS.curriculum == "none":
                    ims, encs, G_loss, D_loss, gen_ims, t_feats = gan_loss(i != 0,  '/gpu:%d' % i, training)
                else:
                    ims, encs, G_loss, D_loss, gen_ims, g_w, d_w, t_feats = gan_loss_smooth_curriculum(i != 0,  '/gpu:%d' % i, training)
                    g_weights_placeholders.append(g_w)
                    d_weights_placeholders.append(d_w)
                im_placeholders.append(ims)
                enc_placeholders.append(encs)
                G_losses.append(G_loss)
                D_losses.append(D_loss)
                gen_ims_vars.append(gen_ims)
                trans_feats.append(t_feats)

                tower_G_grads, tower_D_grads = gan_grads(G_loss, D_loss, G_opt, D_opt, global_step)
                G_grads_list.append(tower_G_grads)
                D_grads_list.append(tower_D_grads)

        G_grads = average_gradients(G_grads_list)
        D_grads = average_gradients(D_grads_list)

        G_train_op, G_loss, D_train_op, D_loss = apply_gan_grads(
            global_step,
            G_grads, G_losses, G_opt,
            D_grads, D_losses, D_opt
        )

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=LOG_DEVICE_PLACEMENT))
        sess.run(init)

        if USE_CHECKPOINT:
            restorer = tf.train.Saver(tf.all_variables())
            restorer.restore(sess, CHECKPOINT_PATH)

        summary_writer = tf.train.SummaryWriter(TRAIN_DIR,
                                                graph_def=sess.graph_def)

        # load data set
        global dataset
        global validation_dataset

        # load first data
        batch_generation_function = load_video_data.AsyncFunction(
            3, dataset.load_batch
        )

        generate_batch = lambda: batch_generation_function.run(IMAGE_SIZE[0])
        generate_encoding_batch = lambda: np.random.uniform(0., 1., [BATCH_SIZE * FLAGS.num_gpus, NUM_ENCODING_FEATURES])
        batch_pref = generate_batch()
        image_batch, _ = batch_pref.get()

        MAX_STEPS = NUM_STEPS_PER_EPOCH * NUM_TRAINING_EPOCHS
        if not training:
            all_vars = tf.all_variables()
            return locals()

        for step in xrange(MAX_STEPS):

            # get the batch data
            image_batch, _ = batch_pref.get()
            batch_pref = generate_batch()


            start_time = time.time()
            encoding_batch = generate_encoding_batch()

            if FLAGS.curriculum != "none":
                g_weights = G_WEIGHTER.step()
                d_weights = D_WEIGHTER.step()

            # load into feed dict
            fd = {}
            for g in range(FLAGS.num_gpus):
                fd[im_placeholders[g]] = image_batch[g*BATCH_SIZE:(g+1)*BATCH_SIZE]
                fd[enc_placeholders[g]] = encoding_batch[g*BATCH_SIZE:(g+1)*BATCH_SIZE]
                if FLAGS.curriculum != "none":
                    fd[g_weights_placeholders[g]] = g_weights
                    fd[d_weights_placeholders[g]] = d_weights

            # run a step with summary generation
            if step % 50 == 0:
                _, _, g_loss_value, d_loss_value, summary_str = sess.run(
                    [D_train_op, G_train_op, G_loss, D_loss, summary_op],
                    feed_dict=fd
                )
                summary_writer.add_summary(summary_str, step)
            # run a step without summary generation
            else:
                _, _, g_loss_value, d_loss_value= sess.run(
                    [D_train_op, G_train_op, G_loss, D_loss],
                    feed_dict=fd
                )

            duration = time.time() - start_time

            nanned = any(np.isnan(l) for l in [g_loss_value, d_loss_value])
            assert not nanned, 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = BATCH_SIZE
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('step %d, G_loss = %.2f, D_loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (
                    step,
                    g_loss_value,
                    d_loss_value,
                    examples_per_sec, sec_per_batch)
                )

            # Save the model checkpoint periodically.
            if step % VALID_ITERATIONS == 0 or (step + 1) == MAX_STEPS:
                print("Validating and Saving Model...")
                checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                [train_set, valid_set, test_set, width, height, channels, num_labels] = validation_dataset

                train_feats = []
                valid_feats = []
                test_feats = []
                for tset, feats in zip([train_set, valid_set, test_set], [train_feats, valid_feats, test_feats]):
                    num_valid_steps = len(tset[0]) // BATCH_SIZE
                    for v_step in xrange(num_valid_steps):
                        start_time = time.time()
                        # get the batch data
                        image_batch = 255. * tset[0][v_step*BATCH_SIZE:(v_step+1)*BATCH_SIZE]

                        # load into feed dict
                        fd = {
                            im_placeholders[0]: image_batch,
                        }

                        transfer_feats, = sess.run([trans_feats[0]], feed_dict=fd)
                        feats.extend(transfer_feats)

                        duration = time.time() - start_time

                        if v_step % 10 == 0:
                            num_examples_per_step = BATCH_SIZE
                            examples_per_sec = num_examples_per_step / duration
                            sec_per_batch = float(duration)

                            format_str = ('step %d (%.1f examples/sec; %.3f sec/batch)')
                            print(format_str % (v_step, examples_per_sec, sec_per_batch))

                train_feats = np.array(train_feats)
                valid_feats = np.array(valid_feats)
                test_feats = np.array(test_feats)

                print("Training SVM")
                #Cs = [10, 1, .1, .01, 001, .0001]
                #scores = []
                #for c in Cs:
                #    print("C = {}".format(c))
                #    clf = SGDClassifier(penalty="l2", alpha=c, n_jobs=-1)
                #    clf.fit(train_feats, train_set[1][:len(train_feats)])
                #    score = clf.score(valid_feats, valid_set[1][:len(valid_feats)])
                #    print("score: {}".format(score))
                #    scores.append(score)
                #print("Best score: {}".format(max(scores)))
                best_c = .1 #Cs[np.argmax(scores)]
                #print("Best C: {}".format(best_c))
                clf = SGDClassifier(penalty="l2", alpha=best_c, n_jobs=-1)
                X = np.concatenate([train_feats, valid_feats])
                Y = np.concatenate([train_set[1][:len(train_feats)], valid_set[1][:len(valid_feats)]])
                clf.fit(X, Y)
                test_score = clf.score(test_feats, test_set[1][:len(test_feats)])
                print("Test score: {}".format(test_score))
                with open(os.path.join(TRAIN_DIR, "valid_scores.txt"), "a") as f:
                    f.write("step: {}, score: {}\n".format(step, test_score))

            if step % NUM_STEPS_PER_EPOCH == 0:
                print("Completed epoch!")


def main(argv=None):
    if gfile.Exists(TRAIN_DIR):
        gfile.DeleteRecursively(TRAIN_DIR)
    gfile.MakeDirs(TRAIN_DIR)
    os.mkdir(os.path.join(TRAIN_DIR, "validation_images"))

    train()


if __name__ == '__main__':
    tf.app.run()
