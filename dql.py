"""
Generic Implementation of deep determanistic policy gradient
http://arxiv.org/pdf/1509.02971v5.pdf
"""

import tensorflow as tf
from tensorflow.python.platform import gfile
from data_handler import ReplayMemory, SequenceReplayMemory, ActionSequenceReplayMemory
import networks
import gym
import gym_ple
import gym_soccer
import gym_pull
from gym import spaces
import random
from utils import _add_loss_summaries, get_weight_loss
import cv2, cv
import numpy as np
import time
import thread_utils
from layers import *
import magic_init
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean("vis", True, "will visualize")
tf.app.flags.DEFINE_string("norm", None, "will use wn")
tf.app.flags.DEFINE_string("train_dir", None, "directory to save training stuff")
tf.app.flags.DEFINE_integer("max_steps", 1000, "maximum number of steps in an episode")
tf.app.flags.DEFINE_boolean("image", False, "will use images")
tf.app.flags.DEFINE_boolean("siamese", True, "will use siamese image networks")
tf.app.flags.DEFINE_string("env", None, "env to use")
tf.app.flags.DEFINE_boolean("reproject", False, "if true, traing image net with reprojection loss")
tf.app.flags.DEFINE_float("target", None, "target reward for early stopping")
tf.app.flags.DEFINE_integer("buffer_size", 1000000, "size of replay buffer")
tf.app.flags.DEFINE_integer("max_length", 10, "timesteps")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch_size")
tf.app.flags.DEFINE_integer("r_loss_decay", None, "decay r loss weight from 1.0 to 0.0 in this many steps")
tf.app.flags.DEFINE_string("checkpoint_path", None, "path to checkpoint file")
tf.app.flags.DEFINE_boolean("action_compression", False, "whether or not to use action compression")
tf.app.flags.DEFINE_boolean("state_prop", False, "whether or not to use state propagation")

def resize_image(img, im_size):
        # resize to smallest side is im_size
        h, w, d = img.shape
        r = float(im_size) / min(h, w)
        new_size = min(w, int(r * w)), min(h, int(r * h))
        im_r = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return im_r / 255.

class VisualEnv:
    """
    Class that wraps around an OpenAI env that changes its actions to act over multiple iterations
    """
    def __init__(self, env, timesteps, im_size):
        self.env = gym.make(env)
        self.timesteps = timesteps
        self.im_size = im_size
        self.action_space = self.env.action_space
        obs_shape = self.reset().shape
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=obs_shape)
        self.frame_buffer = []

        # obs = []
        # obs.append(self.reset(sub_mean=False))
        # for i in range(1000):
        #     ob, r, done, info = self.step(self.env.action_space.sample(), sub_mean=False)
        #     obs.append(ob)
        #     if done:
        #         obs.append(self.reset(sub_mean=False))
        # np_obs = np.array(obs)
        # self.obs_mean = np_obs.mean(axis=0)

    def reset(self):
        obs = self.env.reset()
        ob_im = self.resize_image(obs)
        self.frame_buffer = [ob_im for i in range(self.timesteps)]
        obs_c = np.concatenate(self.frame_buffer, axis=2)
        return obs_c

    def render(self):
        self.env.render()

    def resize_image(self, img):
        # resize to smallest side is im_size
        h, w, d = img.shape
        r = float(self.im_size) / min(h, w)
        new_size = min(w, int(r * w)), min(h, int(r * h))
        im_r = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return im_r / 255.

    def step(self, action):
        new_obs, reward, done, info = self.env.step(action)
        self.frame_buffer = self.frame_buffer[1:] + [self.resize_image(new_obs)]

        new_obs = np.concatenate(self.frame_buffer, axis=2)
        info = {}
        return new_obs, reward, done, info

def get_trainable_var_subset(name, non_trainable_tag=None):
    """
    Gets all trainable variables under scope "name"
    """
    vs = [v for v in tf.trainable_variables() if "{}/".format(name) in v.name]
    if non_trainable_tag is None:
        return vs
    else:
        ntvs = [v for v in tf.all_variables() if "{}/".format(name) in v.name and non_trainable_tag in v.name]
        return vs + ntvs

def get_corresponding_vars(scope_a, scope_b, non_trainable_tag=None):
    a_names = {
        v.name.replace(scope_a+'/', ''): v for v in get_trainable_var_subset(scope_a, non_trainable_tag=non_trainable_tag)
    }
    b_names = {
        v.name.replace(scope_b+'/', ''): v for v in get_trainable_var_subset(scope_b, non_trainable_tag=non_trainable_tag)
    }
    assert set(a_names.keys()).difference(set(b_names)) == set(), \
        "to and from sets must perfectly intersect"
    out = {n: (a_names[n], b_names[n]) for n in a_names.keys()}
    for k, v in out.items():
        print(k, v[0].name, v[1].name)
    return out

def assign_set(to_scope, from_scope):
    """
    Generates an op that assigns all trainable variables under to_scope
    from variables under from_scope
    """
    matches = get_corresponding_vars(to_scope, from_scope)
    pairs = matches.values()
    assigns = [a.assign(b) for a, b in pairs]
    with tf.control_dependencies(assigns):
        assign_op = tf.no_op()
    return assign_op

class DQL:
    def __init__(self, env, train_dir, epsilon, Q_func,
                 buffer_size=1e6, batch_size=64,
                 lr=.001, discount_factor=.99, tau=.001,
                 checkpoint_path=None):
        """
        All networks fed as input need not be scoped, that will be done internally so we can grab the variables

        tau is the moving average decay factor on the target network parameters should be like .001
        noise_process should be a class that can be instantiated and called with a timestep to
            produce a noise vector that is added to actions

        mu should return inputs scaled to -1, 1 and we will scale them automatically based on the
            env spec
        """
        self.Q_func = Q_func
        self.lr = lr
        self.discount_factor = discount_factor
        self.tau = tau
        self.buffer_size = buffer_size
        self.env = env
        self.batch_size = batch_size
        self.last_length = 0

        self._set_replay_buffer()



        self._set_action_space()
        self.epsilon = epsilon
        # placeholder for feeding observations into the algorithm
        self.obs_placeholder = tf.placeholder(
            "float", [None] + list(env.observation_space.shape), "obs_placeholder"
        )
        if len(env.observation_space.shape) == 3:
            tf.image_summary("observation", self.obs_placeholder[:, :, :, :3])
        print("Observation shape: {}".format(env.observation_space.shape))
        self.next_obs_placeholder = tf.placeholder(
            "float", [None] + list(env.observation_space.shape), "next_obs_placeholder"
        )

        self.action_placeholder = tf.placeholder(tf.int32, [None], "action_placeholder")

        self.step = 0
        self.global_step = tf.Variable(0, trainable=False)
        self.reward_placeholder = tf.placeholder("float", [None], "reward_placeholder")
        self.done_placeholder = tf.placeholder("float", [None], "done_placeholder")

        self._initialize_Q_vars()
        self._initialize_Q_loss()

        # ops for updating target networks
        self.update_target_networks_op = self._update_target_networks_op()
        self.target_networks_init_op = self._target_networks_init_op()
        self._train_op()

        self.saver = tf.train.Saver(tf.trainable_variables())
        # Build the summary operation based on the TF collection of Summaries.
        self.summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        # create the session
        self.sess = tf.Session()
        self.sess.run(init)
        # assign target networks to the training networks
        self.sess.run(self.target_networks_init_op)

        if checkpoint_path is not None:
            self.saver.restore(self.sess, checkpoint_path)

        self.summary_writer = tf.train.SummaryWriter(train_dir,
                                                graph_def=self.sess.graph_def)

    def get_Q(self, obs):
        # add actions to placeholders
        fd = {self.obs_placeholder: [obs]}
        q, = self.sess.run([self.Q], feed_dict=fd)
        return q

    def get_action(self, Q_values):
        # with probability self.epsilon return a random action
        if random.random() < self.epsilon:
            return random.choice(range(len(Q_values)))
        # otherwise, return the max Q valued action
        else:
            return np.argmax(Q_values)

    def step_env(self, action):
        return self.env.step(action)


    def training_step(self, max_steps, train=True, render=False):
        observation = self.env.reset()
        total_reward = 0.0
        for i in xrange(max_steps):
            if render:
                self.env.render()
            Q_values = self.get_Q(observation)
            action = self.get_action(Q_values)
            # step the env
            new_obs, reward, done, info = self.step_env(action)

            total_reward += reward
            transition = (observation, action, reward, new_obs, float(done))
            # add tranition to the replay buffer
            self.replay_buffer.add_transition(transition)

            if train:
                t = time.time()
                obs, acts, r, next_obs, d = self.replay_buffer.get_batch()
                assert (np.array(d) >= 0).all()
                batch_time = time.time() - t
                fd = {
                    self.obs_placeholder: np.array(obs),
                    self.next_obs_placeholder: np.array(next_obs),
                    self.reward_placeholder: np.array(r),
                    self.done_placeholder: np.array(d),
                    self.action_placeholder: np.array(acts)
                }

                if self.step % 100 == 0:
                    _, q_loss, sum_str = self.sess.run(
                        [self.train_op, self.Q_loss, self.summary_op],
                        feed_dict=fd
                    )
                    self.summary_writer.add_summary(sum_str, self.step)

                else:
                    _, q_loss = self.sess.run(
                        [self.train_op, self.Q_loss],
                        feed_dict=fd
                    )
                if (self.step % 100 == 0):
                    print("Global Step {}: Q loss = {}, Last Steps = {}, reward = {} | batch load time = {}".format(self.step, q_loss, self.last_length, reward, batch_time))
                    print("    Action {}".format(action))
                self.step += 1

            if done:
                self.last_length = i + 1
                return total_reward
            observation = new_obs

        self.last_length = i + 1
        return total_reward

    def set_epsilon(self, eps):
        self.epsilon = eps

    def _set_replay_buffer(self):
        self.replay_buffer = ReplayMemory(
            self.batch_size, self.buffer_size, os.path.join(FLAGS.train_dir, "tmp")
        )

    def _set_action_space(self):
        """
        sets action space, always a discrete space
        """
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        self.action_space = self.env.action_space


    def _initialize_Q_vars(self):
        with tf.variable_scope("Q") as scope:
            self.Q, self.Q_params = self.Q_func(self.obs_placeholder, self.action_space.n, True)
        with tf.variable_scope("Q_prime") as scope:
            self.Q_prime, _ = self.Q_func(self.next_obs_placeholder, self.action_space.n, False)

    def _initialize_Q_loss(self):
        with tf.variable_scope("Q_loss") as scope:
            # generate loss for learning q parameters
            # slice the Q values for the chosen actions
            one_hot_actions = tf.one_hot(self.action_placeholder, self.action_space.n)
            Q_values = tf.reduce_sum(self.Q * one_hot_actions, [1])

            Q_prime_values = tf.reduce_max(self.Q_prime, [1])

            # the term for the non terminal values
            self.Q_non_term = tf.mul(
                self.reward_placeholder + self.discount_factor * Q_prime_values,
                -1. * (self.done_placeholder - 1.), name="Q_targets_non_term"
            )
            # the term for the terminal values
            self.Q_term = tf.mul(
                self.reward_placeholder,
                self.done_placeholder, name="Q_targets_term"
            )
            self.Q_targets = tf.add(self.Q_non_term, self.Q_term, name="Q_targets")
            tf.histogram_summary("Q_targets", self.Q_targets)
            self.Q_loss = tf.reduce_mean(tf.square(Q_values - self.Q_targets), name="Q_loss")
            tf.add_to_collection("losses", self.Q_loss)

    def _update_target_networks_op(self):
        """
        Generates op for updating the target networks after each iteration
        """
        with tf.variable_scope("update_target_networks") as scope:
            Qp_Q = get_corresponding_vars("Q_prime", "Q", non_trainable_tag="_av")

            Q_tups = Qp_Q.values()
            Q_updates = [
                self.tau * Q_var + (1 - self.tau) * Qp_var for Qp_var, Q_var in Q_tups
            ]

            Q_assign_ops = [
                Qp_var.assign(Q_update) for Q_update, (Qp_var, Q_var) in zip(Q_updates, Q_tups)
            ]
            with tf.control_dependencies(Q_assign_ops):
                update_target_networks_op = tf.no_op("update_target_networks")
        return update_target_networks_op

    def _target_networks_init_op(self):
        """
        Generates op for seting initial values of the main and target networks to be the same
        """
        with tf.variable_scope("target_networks_init") as scope:
            Q_op = assign_set("Q_prime", "Q")
            with tf.control_dependencies([Q_op]):
                target_networks_init_op = tf.no_op("target_networks_init")
        return target_networks_init_op

    def _train_op(self):
        """
        Generates op for one iteration of training
        """
        #weight_loss = get_weight_loss()

        Q_vars = get_trainable_var_subset("Q")
        for v in Q_vars:
            print("Q:", v.name)

        #ql = tf.add_n([weight_loss, self.Q_loss], "total_Q_loss")

        loss_averages_op = _add_loss_summaries([])

        with tf.control_dependencies([loss_averages_op]):
            Q_opt = tf.train.AdamOptimizer(self.lr)
            Q_grads = Q_opt.compute_gradients(self.Q_loss, Q_vars)
            for grad, var in Q_grads:
                if grad is not None:
                    tf.histogram_summary(var.op.name + '/gradients', grad)

            self.Q_train_op = Q_opt.minimize(self.Q_loss, var_list=Q_vars)

        for var in Q_vars:
            tf.histogram_summary(var.op.name, var)

        # create a single op that
        #   1) updates Q network
        #   2) updates mu network
        #   3) updates target networks
        with tf.control_dependencies([self.Q_train_op]):
            with tf.control_dependencies([self.update_target_networks_op]):
                train_op = tf.no_op("train_op")
        self.train_op = train_op

class ActionCompressionDQL:
    def __init__(self, env, train_dir, epsilon,
                 feats_func, Q_func, comp_func, dec_func,
                 max_prediction_steps,
                 buffer_size=1e6, batch_size=64,
                 lr=.001, discount_factor=.99, tau=.001,
                 checkpoint_path=None):
        """
        All networks fed as input need not be scoped, that will be done internally so we can grab the variables

        tau is the moving average decay factor on the target network parameters should be like .001
        noise_process should be a class that can be instantiated and called with a timestep to
            produce a noise vector that is added to actions

        mu should return inputs scaled to -1, 1 and we will scale them automatically based on the
            env spec
        """
        self.Q_func = Q_func
        self.feats_func = feats_func
        self.dec_func = dec_func
        self.comp_func = comp_func
        self.lr = lr
        self.discount_factor = discount_factor
        self.tau = tau
        self.buffer_size = buffer_size
        self.env = env
        self.batch_size = batch_size
        self.last_length = 0
        self.max_prediction_steps = max_prediction_steps

        self._set_replay_buffer()



        self._set_action_space()
        self.epsilon = epsilon
        # placeholder for feeding observations into the algorithm
        self.obs_placeholder = tf.placeholder(
            "float", [None] + list(env.observation_space.shape), "obs_placeholder"
        )
        if len(env.observation_space.shape) == 3:
            tf.image_summary("observation", self.obs_placeholder[:, :, :, :3])
        print("Observation shape: {}".format(env.observation_space.shape))
        self.next_obs_placeholder = tf.placeholder(
            "float", [None] + list(env.observation_space.shape), "next_obs_placeholder"
        )

        # placeholders for start and end of action sequence
        self.action_sequence_start_obs_placeholder = tf.placeholder(
            "float", [None] + list(env.observation_space.shape),
            "action_sequence_start_obs_placeholder"
        )
        self.action_sequence_end_obs_placeholder = tf.placeholder(
            "float", [None] + list(env.observation_space.shape),
            "action_sequence_end_obs_placeholder"
        )
        self.action_sequence_placeholder = tf.placeholder(
            tf.int32, [None, self.max_prediction_steps], "action_sequence_placeholder"
        )
        self.action_sequence_lengths_placeholder = tf.placeholder(
            tf.int32, [None], "action_sequence_lengths_placeholder"
        )

        self.action_placeholder = tf.placeholder(tf.int32, [None], "action_placeholder")

        self.step = 0
        self.global_step = tf.Variable(0, trainable=False)
        self.reward_placeholder = tf.placeholder("float", [None], "reward_placeholder")
        self.done_placeholder = tf.placeholder("float", [None], "done_placeholder")

        self._initialize_Q_vars()
        self._initialize_Q_loss()

        # ops for updating target networks
        self.update_target_networks_op = self._update_target_networks_op()
        self.target_networks_init_op = self._target_networks_init_op()
        self._train_op()

        self.saver = tf.train.Saver(tf.trainable_variables())
        # Build the summary operation based on the TF collection of Summaries.
        self.summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        # create the session
        self.sess = tf.Session()
        self.sess.run(init)
        # assign target networks to the training networks
        self.sess.run(self.target_networks_init_op)

        if checkpoint_path is not None:
            self.saver.restore(self.sess, checkpoint_path)

        self.summary_writer = tf.train.SummaryWriter(train_dir,
                                                graph_def=self.sess.graph_def)

    def get_Q(self, obs):
        # add actions to placeholders
        fd = {self.obs_placeholder: [obs]}
        q, = self.sess.run([self.Q], feed_dict=fd)
        return q

    def get_action(self, Q_values):
        # with probability self.epsilon return a random action
        if random.random() < self.epsilon:
            return random.choice(range(len(Q_values)))
        # otherwise, return the max Q valued action
        else:
            return np.argmax(Q_values)

    def step_env(self, action):
        return self.env.step(action)


    def training_step(self, max_steps, train=True, render=False):
        observation = self.env.reset()
        total_reward = 0.0
        # choose an action sequence length to pick
        action_sequence_length = None
        action_sequence = []
        action_sequence_start_ob = None
        action_sequence_end_ob = None
        for i in xrange(max_steps):
            if render:
                self.env.render()
            Q_values = self.get_Q(observation)
            action = self.get_action(Q_values)
            # step the env
            new_obs, reward, done, info = self.step_env(action)

            total_reward += reward
            transition = (observation, action, reward, new_obs, float(done))
            # add tranition to the replay buffer
            self.replay_buffer.add_transition(transition)

            # update current actioni_sequence
            # if we are starting a new sequence
            if len(action_sequence) == 0:
                action_sequence_length = random.randint(1, self.max_prediction_steps)
                action_sequence_start_ob = observation
            # add the current action to the sequence
            action_sequence.append(action)
            # if the sequence is complete
            if len(action_sequence) == action_sequence_length:
                action_sequence_end_ob = new_obs
                obs_act_seq = (action_sequence_start_ob, action_sequence, action_sequence_end_ob)
                # save the action sequence
                self.action_replay_buffer.add_sequence(obs_act_seq)
                action_sequence = []
                action_sequence_start_ob = None
                action_sequence_end_ob = None
                action_sequence_length = None

            if train:
                t = time.time()
                obs, acts, r, next_obs, d = self.replay_buffer.get_batch()
                act_obs, act_seq, act_lens, act_next_obs = self.action_replay_buffer.get_batch()
                batch_time = time.time() - t
                # print("act obs", act_obs)
                # print("act seq", act_seq)
                # print("act lens", act_lens)
                # print('next act ob', act_next_obs)
                fd = {
                    self.obs_placeholder: np.array(obs),
                    self.next_obs_placeholder: np.array(next_obs),
                    self.reward_placeholder: np.array(r),
                    self.done_placeholder: np.array(d),
                    self.action_placeholder: np.array(acts),
                    self.action_sequence_start_obs_placeholder: np.array(act_obs),
                    self.action_sequence_end_obs_placeholder: np.array(act_next_obs),
                    self.action_sequence_placeholder: np.array(act_seq),
                    self.action_sequence_lengths_placeholder: np.array(act_lens)
                }

                if self.step % 100 == 0:
                    _, q_loss, r_loss, sum_str = self.sess.run(
                        [self.train_op, self.Q_loss, self.R_loss, self.summary_op],
                        feed_dict=fd
                    )
                    self.summary_writer.add_summary(sum_str, self.step)

                else:
                    _, q_loss, r_loss = self.sess.run(
                        [self.train_op, self.Q_loss, self.R_loss],
                        feed_dict=fd
                    )
                if (self.step % 100 == 0):
                    print("Global Step {}: Q loss = {}, R loss = {}, Last Steps = {}, reward = {} | batch load time = {}".format(self.step, q_loss, r_loss, self.last_length, reward, batch_time))
                    print("    Action {}".format(action))
                self.step += 1

            if done:
                self.last_length = i + 1
                return total_reward
            observation = new_obs

        self.last_length = i + 1
        return total_reward

    def set_epsilon(self, eps):
        self.epsilon = eps

    def _set_replay_buffer(self):
        self.replay_buffer = ReplayMemory(
            self.batch_size, self.buffer_size, os.path.join(FLAGS.train_dir, "tmp")
        )
        self.action_replay_buffer = ActionSequenceReplayMemory(
            self.batch_size, self.buffer_size, os.path.join(FLAGS.train_dir, "action_tmp")
        )

    def _set_action_space(self):
        """
        sets action space, always a discrete space
        """
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        self.action_space = self.env.action_space


    def _initialize_Q_vars(self):
        with tf.variable_scope("Q") as scope:
            # get the Q values for the observation batches
            self.obs_feats, fp = self.feats_func(self.obs_placeholder, True)
            self.Q, qp = self.Q_func(self.obs_feats, self.action_space.n, True)
            self.Q_params = fp + qp
        with tf.variable_scope("Q", reuse=True) as scope:
            # get the features for the action sequence initial observations
            self.action_start_obs_feats, _ = self.feats_func(
                self.action_sequence_start_obs_placeholder, True, reuse=True
            )
        with tf.variable_scope("Q") as scope:
            # get the compressed action sequence features
            self.compressed_action_sequence_features = self.comp_func(
                tf.one_hot(self.action_sequence_placeholder, self.env.action_space.n),
                self.action_sequence_lengths_placeholder
            )
            self.decoded_action_sequence_end_obs = self.dec_func(
                self.action_start_obs_feats, self.compressed_action_sequence_features,
                self.env.observation_space
            )

        with tf.variable_scope("Q_prime") as scope:
            self.Q_prime_feats, _ = self.feats_func(self.next_obs_placeholder, False)
            self.Q_prime, _ = self.Q_func(self.Q_prime_feats, self.action_space.n, False)

    def _initialize_Q_loss(self):
        with tf.variable_scope("Q_loss") as scope:
            # generate loss for learning q parameters
            # slice the Q values for the chosen actions
            one_hot_actions = tf.one_hot(self.action_placeholder, self.action_space.n)
            Q_values = tf.reduce_sum(self.Q * one_hot_actions, [1])

            Q_prime_values = tf.reduce_max(self.Q_prime, [1])

            # the term for the non terminal values
            self.Q_non_term = tf.mul(
                self.reward_placeholder + self.discount_factor * Q_prime_values,
                -1. * (self.done_placeholder - 1.), name="Q_targets_non_term"
            )
            # the term for the terminal values
            self.Q_term = tf.mul(
                self.reward_placeholder,
                self.done_placeholder, name="Q_targets_term"
            )
            self.Q_targets = tf.add(self.Q_non_term, self.Q_term, name="Q_targets")
            tf.histogram_summary("Q_targets", self.Q_targets)
            self.Q_loss = tf.reduce_mean(tf.square(Q_values - self.Q_targets), name="Q_loss")
            tf.add_to_collection("losses", self.Q_loss)

            # compute reprojection loss
            self.R_loss = tf.reduce_mean(
                tf.square(self.action_sequence_end_obs_placeholder - self.decoded_action_sequence_end_obs),
                name="R_loss"
            )
            tf.add_to_collection("losses", self.R_loss)

    def _update_target_networks_op(self):
        """
        Generates op for updating the target networks after each iteration
        """
        with tf.variable_scope("update_target_networks") as scope:
            Qp_Q = get_corresponding_vars("Q_prime", "Q", non_trainable_tag="_av")

            Q_tups = Qp_Q.values()
            for Qp_var, Q_var in Q_tups:
                print(Qp_var.name, Q_var.name)
            Q_updates = [
                self.tau * Q_var + (1 - self.tau) * Qp_var for Qp_var, Q_var in Q_tups
            ]

            Q_assign_ops = [
                Qp_var.assign(Q_update) for Q_update, (Qp_var, Q_var) in zip(Q_updates, Q_tups)
            ]
            with tf.control_dependencies(Q_assign_ops):
                update_target_networks_op = tf.no_op("update_target_networks")
        return update_target_networks_op

    def _target_networks_init_op(self):
        """
        Generates op for seting initial values of the main and target networks to be the same
        """
        with tf.variable_scope("target_networks_init") as scope:
            Q_op = assign_set("Q_prime", "Q")
            with tf.control_dependencies([Q_op]):
                target_networks_init_op = tf.no_op("target_networks_init")
        return target_networks_init_op

    def _train_op(self):
        """
        Generates op for one iteration of training
        """

        total_loss = self.Q_loss + 1000*self.R_loss #tf.add_n(tf.get_collection("losses"), "total_Q_loss")

        loss_averages_op = _add_loss_summaries([total_loss])

        Q_vars = [v for v in tf.trainable_variables() if not "Q_prime" in v.name]
        for v in Q_vars:
            print("Q var", v.name)

        with tf.control_dependencies([loss_averages_op]):
            Q_opt = tf.train.AdamOptimizer(self.lr * .1)
            Q_grads = Q_opt.compute_gradients(self.Q_loss, var_list=Q_vars)
            D_opt = tf.train.AdamOptimizer(self.lr * 10)
            D_grads = D_opt.compute_gradients(self.R_loss, var_list=Q_vars)
            for grad, var in Q_grads:# + D_grads:
                if grad is not None:
                    tf.histogram_summary(var.op.name + '/gradients', grad)

            self.Q_train_op = Q_opt.minimize(self.Q_loss, var_list=Q_vars)
            self.R_train_op = D_opt.minimize(self.R_loss, var_list=Q_vars)

        for var in Q_vars:
            tf.histogram_summary(var.op.name, var)

        # create a single op that
        #   1) updates Q network
        #   3) updates target networks
        with tf.control_dependencies([self.Q_train_op, self.R_train_op]):
            with tf.control_dependencies([self.update_target_networks_op]):
                train_op = tf.no_op("train_op")
        self.train_op = train_op

class RDQL:
    def __init__(self, env, train_dir, epsilon, Q_func,
                 buffer_size=1e5, batch_size=16, max_length=10,
                 lr=.001, discount_factor=.99, tau=.001, im_size=64,
                 checkpoint_path=None):
        """
        All networks fed as input need not be scoped, that will be done internally so we can grab the variables

        tau is the moving average decay factor on the target network parameters should be like .001
        noise_process should be a class that can be instantiated and called with a timestep to
            produce a noise vector that is added to actions

        mu should return inputs scaled to -1, 1 and we will scale them automatically based on the
            env spec
        """
        self.Q_func = Q_func
        self.lr = lr
        self.max_length = max_length
        self.discount_factor = discount_factor
        self.tau = tau
        self.buffer_size = buffer_size
        self.env = env
        self.batch_size = batch_size
        self.last_length = 0
        self.im_size = im_size
        obs_shape = list(self.reset().shape)

        self._set_replay_buffer()
        self._set_action_space()

        self.epsilon = epsilon
        # placeholder for feeding observations into the algorithm
        self.obs_placeholder = tf.placeholder(
            "float", [None, self.max_length] + obs_shape, "obs_placeholder"
        )
        self.used_frames = networks.used_frames(self.obs_placeholder)

        if len(env.observation_space.shape) == 3:
            tf.image_summary("observation", self.obs_placeholder[:, 0, :, :, :])
        print("Observation shape: {}".format(obs_shape))
        self.next_obs_placeholder = tf.placeholder(
            "float", [None, self.max_length] + obs_shape,
            "next_obs_placeholder"
        )

        self.action_placeholder = tf.placeholder(
            tf.int32, [None, self.max_length], "action_placeholder"
        )

        self.step = 0
        self.global_step = tf.Variable(0, trainable=False)
        self.reward_placeholder = tf.placeholder(
            "float", [None, self.max_length], "reward_placeholder"
        )
        self.done_placeholder = tf.placeholder(
            "float", [None, self.max_length], "done_placeholder"
        )

        self._initialize_Q_vars()
        self._initialize_Q_loss()

        # ops for updating target networks
        self.update_target_networks_op = self._update_target_networks_op()
        self.target_networks_init_op = self._target_networks_init_op()
        self._train_op()

        self.saver = tf.train.Saver(tf.trainable_variables())
        # Build the summary operation based on the TF collection of Summaries.
        self.summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        # create the session
        self.sess = tf.Session()
        self.sess.run(init)
        # assign target networks to the training networks
        self.sess.run(self.target_networks_init_op)

        if checkpoint_path is not None:
            self.saver.restore(self.sess, checkpoint_path)

        self.summary_writer = tf.train.SummaryWriter(train_dir,
                                                graph_def=self.sess.graph_def)

    def get_Q(self, live_obs):
        # add actions to placeholders
        fd = {self.obs_placeholder: [np.array(live_obs)]}
        q_values, uf = self.sess.run([self.Q, self.used_frames], feed_dict=fd)
        #print(uf)
        return q_values[0]

    def get_action(self, Q_values):
        # with probability self.epsilon return a random action
        if random.random() < self.epsilon:
            return random.choice(range(len(Q_values)))
        # otherwise, return the max Q valued action
        else:
            return np.argmax(Q_values)

    def step_env(self, action):
        new_obs, reward, done, info = self.env.step(action)
        return resize_image(new_obs, self.im_size), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return resize_image(obs, self.im_size)


    def training_step(self, max_steps, train=True, render=False):
        observation = self.reset()
        total_reward = 0.0
        sequence = []
        live_obs = []
        # fill with zeros for the rest of timesteps
        for i in range(self.max_length):
            live_obs.append(np.zeros_like(observation))

        for i in xrange(max_steps):
            if render:
                self.env.render()
            t = time.time()

            if i < self.max_length:
                live_obs[i] = observation
            else:
                live_obs = live_obs[1:] + [observation]
            # this outputs the Q values for each timestep in the sequence of observed frames
            Q_values = self.get_Q(live_obs)
            # print(Q_values.shape)
            # for t, q in enumerate(Q_values):
            #     print(t, q)
            # get the q values that correspond to the frame we are in
            this_timestep_q_values = Q_values[i] if i < self.max_length else Q_values[-1]

            action = self.get_action(this_timestep_q_values)
            # step the env
            new_obs, reward, done, info = self.step_env(action)

            total_reward += reward
            transition = (observation, action, reward, new_obs, float(done))
            sequence.append(transition)
            as_time = time.time() - t
            if train:
                t = time.time()
                obs, acts, r, next_obs, d = self.replay_buffer.get_batch()
                batch_time = time.time() - t
                fd = {
                    self.obs_placeholder: np.array(obs),
                    self.next_obs_placeholder: np.array(next_obs),
                    self.reward_placeholder: np.array(r),
                    self.done_placeholder: np.array(d),
                    self.action_placeholder: np.array(acts)
                }
                t = time.time()
                if self.step % 100 == 0:
                    _, q_loss, sum_str = self.sess.run(
                        [self.train_op, self.Q_loss, self.summary_op],
                        feed_dict=fd
                    )
                    self.summary_writer.add_summary(sum_str, self.step)

                else:
                    _, q_loss = self.sess.run(
                        [self.train_op, self.Q_loss],
                        feed_dict=fd
                    )
                wu_time = time.time() - t
                if (self.step % 100 == 0):
                    print("Global Step {}: Q loss = {}, Last Steps = {}, reward = {} | batch load time = {}, weight update time = {}, action selection time = {}".format(self.step, q_loss, self.last_length, reward, batch_time, wu_time, as_time))
                    print("    Action {}".format(action))
                self.step += 1

            if done:
                self.last_length = i + 1
                self.replay_buffer.add_sequence(sequence)
                return total_reward
            observation = new_obs


        self.last_length = i + 1
        self.replay_buffer.add_sequence(sequence)
        return total_reward

    def set_epsilon(self, eps):
        self.epsilon = eps

    def _set_replay_buffer(self):
        self.replay_buffer = SequenceReplayMemory(
            self.batch_size, self.buffer_size, os.path.join(FLAGS.train_dir, "tmp"),
            max_length=self.max_length
        )

    def _set_action_space(self):
        """
        sets action space, always a discrete space
        """
        assert isinstance(self.env.action_space, gym.spaces.Discrete), self.env.action_space
        self.action_space = self.env.action_space


    def _initialize_Q_vars(self):
        with tf.variable_scope("Q") as scope:
            self.Q, _ = self.Q_func(
                self.obs_placeholder, self.action_space.n, self.max_length,
            )
        with tf.variable_scope("Q_prime") as scope:
            self.Q_prime, _ = self.Q_func(
                self.next_obs_placeholder, self.action_space.n, self.max_length
            )

    def _initialize_Q_loss(self):
        with tf.variable_scope("Q_loss") as scope:
            # generate loss for learning q parameters
            # slice the Q values for the chosen actions
            one_hot_actions = tf.one_hot(self.action_placeholder, self.action_space.n)
            Q_values = tf.reduce_sum(self.Q * one_hot_actions, [2])

            Q_prime_values = tf.reduce_max(self.Q_prime, [2])

            # the term for the non terminal values
            self.Q_non_term = tf.mul(
                self.reward_placeholder + self.discount_factor * Q_prime_values,
                -1. * (self.done_placeholder - 1.), name="Q_targets_non_term"
            )
            # the term for the terminal values
            self.Q_term = tf.mul(
                self.reward_placeholder,
                self.done_placeholder, name="Q_targets_term"
            )
            self.Q_targets = tf.add(self.Q_non_term, self.Q_term, name="Q_targets")
            tf.histogram_summary("Q_targets", self.Q_targets)
            self.Q_loss = tf.reduce_mean(
                tf.square(Q_values - self.Q_targets) * self.used_frames,
                name="Q_loss"
            )
            tf.add_to_collection("losses", self.Q_loss)

    def _update_target_networks_op(self):
        """
        Generates op for updating the target networks after each iteration
        """
        with tf.variable_scope("update_target_networks") as scope:
            Qp_Q = get_corresponding_vars("Q_prime", "Q", non_trainable_tag="_av")

            Q_tups = Qp_Q.values()
            Q_updates = [
                self.tau * Q_var + (1 - self.tau) * Qp_var for Qp_var, Q_var in Q_tups
            ]

            Q_assign_ops = [
                Qp_var.assign(Q_update) for Q_update, (Qp_var, Q_var) in zip(Q_updates, Q_tups)
            ]
            with tf.control_dependencies(Q_assign_ops):
                update_target_networks_op = tf.no_op("update_target_networks")
        return update_target_networks_op

    def _target_networks_init_op(self):
        """
        Generates op for seting initial values of the main and target networks to be the same
        """
        with tf.variable_scope("target_networks_init") as scope:
            Q_op = assign_set("Q_prime", "Q")
            with tf.control_dependencies([Q_op]):
                target_networks_init_op = tf.no_op("target_networks_init")
        return target_networks_init_op

    def _train_op(self):
        """
        Generates op for one iteration of training
        """
        weight_loss = get_weight_loss()

        Q_vars = get_trainable_var_subset("Q")

        tf.scalar_summary("Q_loss", self.Q_loss)

        ql = weight_loss + self.Q_loss

        Q_opt = tf.train.AdamOptimizer(self.lr)
        Q_grads = Q_opt.compute_gradients(ql, Q_vars)
        for grad, var in Q_grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        self.Q_train_op = Q_opt.minimize(ql, var_list=Q_vars)

        for var in Q_vars:
            tf.histogram_summary(var.op.name, var)

        # create a single op that
        #   1) updates Q network
        #   2) updates mu network
        #   3) updates target networks
        with tf.control_dependencies([self.Q_train_op]):
            with tf.control_dependencies([self.update_target_networks_op]):
                train_op = tf.no_op("train_op")
        self.train_op = train_op

class StatePropDQL:
    def __init__(self, env, train_dir, epsilon, feats_func, Q_func, state_prop_func,
                 buffer_size=1e5, batch_size=16, max_length=10,
                 lr=.001, discount_factor=.99, tau=.001,
                 checkpoint_path=None):
        """
        All networks fed as input need not be scoped, that will be done internally so we can grab the variables

        tau is the moving average decay factor on the target network parameters should be like .001
        noise_process should be a class that can be instantiated and called with a timestep to
            produce a noise vector that is added to actions

        mu should return inputs scaled to -1, 1 and we will scale them automatically based on the
            env spec
        """
        self.Q_func = Q_func
        self.feats_func = feats_func
        self.state_prop_func = state_prop_func
        self.lr = lr
        self.max_length = max_length
        self.discount_factor = discount_factor
        self.tau = tau
        self.buffer_size = buffer_size
        self.env = env
        self.batch_size = batch_size
        self.last_length = 0

        obs_shape = list(self.reset().shape)
        self.max_length = max_length

        self._set_replay_buffer()
        self._set_action_space()

        self.epsilon = epsilon
        # placeholder for feeding observations into the algorithm
        self.obs_placeholder = tf.placeholder(
            "float", [None] + list(env.observation_space.shape), "obs_placeholder"
        )

        self.next_obs_placeholder = tf.placeholder(
            "float", [None] + list(env.observation_space.shape),
            "next_obs_placeholder"
        )
        self.next_obs_sequence_placeholder = tf.placeholder(
            "float", [None, self.max_length] + obs_shape,
            "next_obs_sequence_placeholder"
        )
        self.action_sequence_length = networks.sequence_length(self.next_obs_sequence_placeholder)

        self.actions_placeholder = tf.placeholder(
            tf.int32, [None, self.max_length], "actions_placeholder"
        )
        self.this_act_placeholder = tf.placeholder(
            tf.int32, [None], "this_act_placeholder"
        )

        self.step = 0
        self.global_step = tf.Variable(0, trainable=False)
        self.reward_placeholder = tf.placeholder(
            "float", [None], "reward_placeholder"
        )
        self.done_placeholder = tf.placeholder(
            "float", [None], "done_placeholder"
        )

        self._initialize_Q_vars()
        self._initialize_Q_loss()

        # ops for updating target networks
        self.update_target_networks_op = self._update_target_networks_op()
        self.target_networks_init_op = self._target_networks_init_op()
        self._train_op()

        self.saver = tf.train.Saver(tf.trainable_variables())
        # Build the summary operation based on the TF collection of Summaries.
        self.summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        # create the session
        self.sess = tf.Session()
        self.sess.run(init)
        # assign target networks to the training networks
        self.sess.run(self.target_networks_init_op)

        if checkpoint_path is not None:
            self.saver.restore(self.sess, checkpoint_path)

        self.summary_writer = tf.train.SummaryWriter(train_dir,
                                                graph_def=self.sess.graph_def)

    def get_Q(self, live_obs):
        # add actions to placeholders
        fd = {self.obs_placeholder: [np.array(live_obs)]}
        q_values = self.sess.run([self.Q], feed_dict=fd)
        return q_values[0]

    def get_action(self, Q_values):
        # with probability self.epsilon return a random action
        if random.random() < self.epsilon:
            return random.choice(range(len(Q_values)))
        # otherwise, return the max Q valued action
        else:
            return np.argmax(Q_values)

    def step_env(self, action):
        return self.env.step(action)

    def reset(self):
        obs = self.env.reset()
        return obs


    def training_step(self, max_steps, train=True, render=False):
        observation = self.reset()
        total_reward = 0.0
        sequence = []

        for i in xrange(max_steps):
            if render:
                self.env.render()
            t = time.time()

            Q_values = self.get_Q(observation)

            action = self.get_action(Q_values)
            # step the env
            new_obs, reward, done, info = self.step_env(action)

            total_reward += reward
            transition = (observation, action, reward, new_obs, float(done))
            sequence.append(transition)
            as_time = time.time() - t
            if train:
                t = time.time()
                obs, acts, r, next_obs, d = self.replay_buffer.get_batch()
                ob = np.array(obs)[:, 0, :]
                act = np.array(acts)[:, 0]
                r = np.array(r)[:, 0]
                next_ob = np.array(next_obs)[:, 0, :]
                d = np.array(d)[:, 0]
                batch_time = time.time() - t
                fd = {
                    self.obs_placeholder: ob,
                    self.next_obs_placeholder: next_ob,
                    self.next_obs_sequence_placeholder: next_obs,
                    self.reward_placeholder: r,
                    self.done_placeholder: d,
                    self.this_act_placeholder: act,
                    self.actions_placeholder: np.array(acts)
                }
                t = time.time()
                if self.step % 100 == 0:
                    _, q_loss, p_loss, sum_str, q, qp, pf, pt = self.sess.run(
                        [self.train_op, self.Q_loss, self.prop_loss, self.summary_op, self.Q, self.Q_prime, self.prop_states, self.prop_states_targets],
                        feed_dict=fd
                    )
                    # print(uf, "used frames")
                    # print(q, q.shape, "q")
                    # print(qp, qp.shape, "qp")
                    # print(pf, "prop_feats")
                    # print(pt, "prop_targets")
                    self.summary_writer.add_summary(sum_str, self.step)

                else:
                    _, q_loss = self.sess.run(
                        [self.train_op, self.Q_loss],
                        feed_dict=fd
                    )
                wu_time = time.time() - t
                if (self.step % 100 == 0):
                    print("Global Step {}: Q loss = {}, Last Steps = {}, reward = {} | batch load time = {}, weight update time = {}, action selection time = {}".format(self.step, q_loss, self.last_length, reward, batch_time, wu_time, as_time))
                    print("    Action {}".format(action))
                    print("Prop loss {}".format(p_loss))
                self.step += 1

            if done:
                self.last_length = i + 1
                self.replay_buffer.add_sequence(sequence)
                return total_reward
            observation = new_obs


        self.last_length = i + 1
        self.replay_buffer.add_sequence(sequence)
        return total_reward

    def set_epsilon(self, eps):
        self.epsilon = eps

    def _set_replay_buffer(self):
        self.replay_buffer = SequenceReplayMemory(
            self.batch_size, self.buffer_size, os.path.join(FLAGS.train_dir, "tmp"),
            max_length=self.max_length
        )

    def _set_action_space(self):
        """
        sets action space, always a discrete space
        """
        assert isinstance(self.env.action_space, gym.spaces.Discrete), self.env.action_space
        self.action_space = self.env.action_space


    def _initialize_Q_vars(self):
        with tf.variable_scope("Q") as scope:
            self.cur_feats, fp = self.feats_func(self.obs_placeholder, True)
            self.Q, qp = self.Q_func(self.cur_feats, self.action_space.n, True)
            self.prop_states = self.state_prop_func(
                tf.one_hot(self.actions_placeholder, self.action_space.n),
                self.action_sequence_length,
                self.cur_feats
            )
            self.Q_params = fp + qp

        with tf.variable_scope("Q_prime") as scope:
            prime_feats, _= self.feats_func(self.next_obs_placeholder, False)
            self.Q_prime, _ = self.Q_func(prime_feats, self.action_space.n, False)

        with tf.variable_scope("Q_prime", reuse=True) as scope:
            linear_next_obs = tf.reshape(
                self.next_obs_sequence_placeholder,
                [-1] + self.obs_placeholder.get_shape().as_list()[1:]
            )
            linear_next_obs_feats, _ = self.feats_func(linear_next_obs, False)

            self.prop_states_targets = tf.reshape(
                linear_next_obs_feats,
                [-1, self.max_length, linear_next_obs_feats.get_shape().as_list()[-1]]
            )


    def _initialize_Q_loss(self):
        with tf.variable_scope("Q_loss") as scope:
            # generate loss for learning q parameters
            # slice the Q values for the chosen actions
            one_hot_actions = tf.one_hot(self.this_act_placeholder, self.action_space.n)
            Q_values = tf.reduce_sum(self.Q * one_hot_actions, [1])

            Q_prime_values = tf.reduce_max(self.Q_prime, [1])

            # the term for the non terminal values
            self.Q_non_term = tf.mul(
                self.reward_placeholder + self.discount_factor * Q_prime_values,
                -1. * (self.done_placeholder - 1.), name="Q_targets_non_term"
            )
            # the term for the terminal values
            self.Q_term = tf.mul(
                self.reward_placeholder,
                self.done_placeholder, name="Q_targets_term"
            )
            self.Q_targets = tf.add(self.Q_non_term, self.Q_term, name="Q_targets")
            tf.histogram_summary("Q_targets", self.Q_targets)
            self.Q_loss = tf.reduce_mean(tf.square(Q_values - self.Q_targets), name="Q_loss")
            tf.add_to_collection("losses", self.Q_loss)

            self.prop_loss = 0.0*tf.nn.l2_loss(
                self.prop_states - self.prop_states_targets, name="prop_loss"
            )

    def _update_target_networks_op(self):
        """
        Generates op for updating the target networks after each iteration
        """
        with tf.variable_scope("update_target_networks") as scope:
            Qp_Q = get_corresponding_vars("Q_prime", "Q", non_trainable_tag="_av")

            Q_tups = Qp_Q.values()
            Q_updates = [
                self.tau * Q_var + (1 - self.tau) * Qp_var for Qp_var, Q_var in Q_tups
            ]

            Q_assign_ops = [
                Qp_var.assign(Q_update) for Q_update, (Qp_var, Q_var) in zip(Q_updates, Q_tups)
            ]
            with tf.control_dependencies(Q_assign_ops):
                update_target_networks_op = tf.no_op("update_target_networks")
        return update_target_networks_op

    def _target_networks_init_op(self):
        """
        Generates op for seting initial values of the main and target networks to be the same
        """
        with tf.variable_scope("target_networks_init") as scope:
            Q_op = assign_set("Q_prime", "Q")
            with tf.control_dependencies([Q_op]):
                target_networks_init_op = tf.no_op("target_networks_init")
        return target_networks_init_op

    def _train_op(self):
        """
        Generates op for one iteration of training
        """
        #weight_loss = get_weight_loss()

        Q_vars = get_trainable_var_subset("Q")

        tf.scalar_summary("Q_loss", self.Q_loss)
        tf.scalar_summary("S_loss", self.prop_loss)

        #ql = weight_loss + self.Q_loss

        Q_opt = tf.train.AdamOptimizer(self.lr)
        Q_grads = Q_opt.compute_gradients(self.Q_loss + self.prop_loss, Q_vars)
        for grad, var in Q_grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        self.Q_train_op = Q_opt.minimize(self.Q_loss + self.prop_loss, var_list=Q_vars)

        for var in Q_vars:
            tf.histogram_summary(var.op.name, var)

        # create a single op that
        #   1) updates Q network
        #   2) updates mu network
        #   3) updates target networks
        with tf.control_dependencies([self.Q_train_op]):
            with tf.control_dependencies([self.update_target_networks_op]):
                train_op = tf.no_op("train_op")
        self.train_op = train_op


if __name__ == "__main__":

    if gfile.Exists(FLAGS.train_dir):
        gfile.DeleteRecursively(FLAGS.train_dir)
    gfile.MakeDirs(FLAGS.train_dir)
    logfile = open("{}/log.txt".format(FLAGS.train_dir), 'w')

    epsilon = 1.
    env_n = FLAGS.env
    if FLAGS.image:

        env = gym.make(env_n)
        env.monitor.start(os.path.join(FLAGS.train_dir, "monitor"))
        dql = RDQL(
            env, FLAGS.train_dir, epsilon, networks.DQL_net_conv_recurrent,
            max_length=FLAGS.max_length, batch_size=FLAGS.batch_size,
            checkpoint_path=FLAGS.checkpoint_path
        )
    else:
        env = gym.make(env_n)
        env.monitor.start(os.path.join(FLAGS.train_dir, "monitor"))
        if FLAGS.action_compression:
            dql = ActionCompressionDQL(
                env, FLAGS.train_dir, epsilon,
                networks.DQL_net_feats,
                networks.DQL_net_Q,
                networks.DQL_net_comp,
                networks.DQL_net_dec, FLAGS.max_length,
                buffer_size=FLAGS.buffer_size, batch_size=FLAGS.batch_size,
                checkpoint_path=FLAGS.checkpoint_path
            )
        elif FLAGS.state_prop:
            dql = StatePropDQL(
                env, FLAGS.train_dir, epsilon,
                networks.DQL_net_feats,
                networks.DQL_net_Q,
                networks.DQL_net_state_prop,
                max_length=FLAGS.max_length,
                buffer_size=FLAGS.buffer_size, batch_size=FLAGS.batch_size,
                checkpoint_path=FLAGS.checkpoint_path
            )
        else:
            dql = DQL(
                env, FLAGS.train_dir, epsilon, networks.DQL_net,
                buffer_size=FLAGS.buffer_size, batch_size=FLAGS.batch_size,
                checkpoint_path=FLAGS.checkpoint_path
            )

    while len(dql.replay_buffer.fnames) < dql.batch_size * 10:
        dql.training_step(FLAGS.max_steps, False, False)
        print(len(dql.replay_buffer.fnames))
    _ = dql.replay_buffer.get_batch()
    # if not FLAGS.image:
    #     print("Running magic init")
    #     def batch_func():
    #         obs, acts, r, next_obs, d = dql.replay_buffer.get_batch()
    #         return [obs]
    #     magic_init.magic_init(
    #         dql.Q_params, "pca", dql.sess,
    #         batch_func,
    #         [dql.obs_placeholder]
    #     )
    #     print("done")
    #     dql.sess.run(dql.target_networks_init_op)

    rewards = []
    for i in range(5000):
        if i % 100 == 0:
            epsilon /= 10
            dql.set_epsilon(epsilon)
            print("reducing noise to {}".format(epsilon))
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            dql.saver.save(dql.sess, checkpoint_path, global_step=i)
        if i % 10 == 0:
            print("Completed {} Episodes".format(i))
            if FLAGS.vis:
                #dql.set_epsilon(0.0)
                r = dql.training_step(FLAGS.max_steps, False, FLAGS.vis)
                print("Reward: {}".format(r))
                #dql.set_epsilon(epsilon)
        r = dql.training_step(FLAGS.max_steps, True, False)
        logfile.write("{}, {}, {}\n".format(dql.step, i, r))
        logfile.flush()
        if len(rewards) < 100:
            rewards.append(r)
        else:
            rewards = rewards[1:]
            rewards.append(r)
            ave_r = sum(rewards) / 100.
            print(FLAGS.train_dir, "ave_r", ave_r, "r", r)
            if ave_r > FLAGS.target:
                print("Solved in {} episodes".format(i))
                break
    dql.env.monitor.close()
    gfile.DeleteRecursively("{}/tmp".format(FLAGS.train_dir))
