"""
Generic Implementation of deep determanistic policy gradient
http://arxiv.org/pdf/1509.02971v5.pdf
"""
import tensorflow as tf
from tensorflow.python.platform import gfile
import data_handler
import networks
import gym
from gym import spaces
import acrobot_continuous
import pole_env
import random
from utils import _add_loss_summaries, get_weight_loss
import cv2, cv
import numpy as np
import time
import thread_utils
from layers import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean("vis", True, "will visualize")
tf.app.flags.DEFINE_boolean("canvas", False, "will use canvas")
tf.app.flags.DEFINE_string("norm", None, "will use wn")
tf.app.flags.DEFINE_string("train_dir", None, "directory to save training stuff")
tf.app.flags.DEFINE_integer("max_steps", 1000, "maximum number of steps in an episode")
tf.app.flags.DEFINE_boolean("image", False, "will use images")
tf.app.flags.DEFINE_boolean("siamese", True, "will use siamese image networks")
tf.app.flags.DEFINE_string("env", None, "env to use")
tf.app.flags.DEFINE_boolean("reproject", False, "if true, traing image net with reprojection loss")
tf.app.flags.DEFINE_float("target", None, "target reward for early stopping")
tf.app.flags.DEFINE_integer("buffer_size", 1000000, "size of replay buffer")
tf.app.flags.DEFINE_integer("r_loss_decay", 100000, "decay r loss weight from 1.0 to 0.0 in this many steps")


def softmax(w):
    e = np.exp(w)
    dist = e / np.sum(e)
    return dist

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


class GaussianNoiseProcess:
    def __init__(self, std):
        self.std = std
    def sample(self, shape):
        return np.random.normal(np.zeros(shape), self.std)

class RandomSampleProcess:
    def __init__(self, eps):
        self.eps = eps
    def sample(self):
        return random.random() < self.eps

class VisualEnv:
    """
    Class that wraps around an OpenAI env that changes its observations to be images
    and changes its actions to act over multiple iterations
    """
    def __init__(self, env, timesteps, im_size):
        self.env = env
        self.timesteps = timesteps
        self.im_size = im_size
        self.action_space = self.env.action_space
        shape = [im_size, im_size, timesteps * 3]
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=shape)

        obs = []
        obs.append(self.reset(sub_mean=False))
        for i in range(1000):
            ob, r, done, info = self.step(self.env.action_space.sample(), sub_mean=False)
            obs.append(ob)
            if done:
                obs.append(self.reset(sub_mean=False))
        np_obs = np.array(obs)
        self.obs_mean = np_obs.mean(axis=0)

    def reset(self, sub_mean=True):
        _ = self.env.reset()
        ob_im = self.resize_image(self.env.render("rgb_array"))
        obs = [ob_im for i in range(self.timesteps)]
        obs_c = np.concatenate(obs, axis=2)
        if sub_mean:
            return obs_c - self.obs_mean
        else:
            return obs_c

    def render(self):
        self.env.render()

    def resize_image(self, img):
        # resize to smallest side is im_size
        h, w, d = img.shape
        r = float(self.im_size) / min(h, w)
        new_size = min(w, int(r * w)), min(h, int(r * h))
        im_r = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        h, w, d = im_r.shape
        if h <= w:
            buf = (w - self.im_size) / 2
            im_c = im_r[:, buf:buf+self.im_size]
        else:
            buf = (h - self.im_size) / 2
            im_c = im_r[buf:buf+self.im_size, :]
        assert im_c.shape[0] == self.im_size and im_c.shape[1] == self.im_size
        return im_c / 255.

    def step(self, action, sub_mean=True):
        new_obs_v = []
        rewards = []
        dones = []
        for t in range(self.timesteps):
            new_obs, reward, done, info = self.env.step(action)
            rewards.append(reward)
            dones.append(done)
            new_obs_v.append(self.env.render("rgb_array"))
        new_obs_vd = [self.resize_image(im) for im in new_obs_v]
        done = any(dones)
        reward = sum(rewards)
        new_obs = np.concatenate(new_obs_vd, axis=2)
        info = {}
        if sub_mean:
            return new_obs - self.obs_mean, reward, done, info
        else:
            return new_obs, reward, done, info


def get_action_shape(action_space):
    """
    returns a list or nested list of action shapes for a given environment
    """
    # if discrete
    if isinstance(action_space, gym.spaces.Discrete):
        return [action_space.n]
    # if continuous
    if isinstance(action_space, gym.spaces.Box):
        return list(action_space.shape)
    # if a tuple, return a nested list
    if isinstance(action_space, gym.spaces.Tuple):
        out = []
        for s in action_space.spaces:
            assert not isinstance(s, gym.spaces.Tuple), "We do not support nested tuple spaces"
            out.extend(get_action_shape(s))
        return out
    else:
        assert False, "Unsupported action space type : {}".format(action_space)

def mu_scale(mu, name, action_space):
    """
    returns a vector that scales the unnormalized outputs from the mu network into the range that
    the environment expects
    """
    assert isinstance(action_space, gym.spaces.Tuple), "action must be a tuple space"
    assert len(mu) == len(action_space.spaces), "must have as many outputs as spaces"
    scaled_mus = []
    for i, (mu_output, space) in enumerate(zip(mu, action_space.spaces)):
        assert not isinstance(space, gym.spaces.Tuple), "Do not support nested tuple spaces"

        if isinstance(space, gym.spaces.Discrete):
            scaled_mus.append(tf.nn.softmax(mu_output, name="{}_{}_scaled".format(i, name)))

        elif isinstance(space, gym.spaces.Box):
            high = space.high
            low = space.low
            value_range = high - low

            value_shift = (high + low) / 2.0
            value_scale = 2.0 / value_range

            scaled_mus.append(
                tf.identity(
                    (tf.tanh(mu_output) / value_scale) + value_shift,
                    name="{}_{}_scaled".format(i, name)
                )
            )
        else:
            assert False, "Unsupported action space type : {}".format(action_space)
    return scaled_mus

def sample_action(action_dist):
    """
    for discrete output spaces, will return an index sampled from the weighted distribution
    """
    return np.random.choice(range(len(action_dist)), p=action_dist)

def clip_action(action, space):
    """
    For continuous output spaces, will clip the output to be within the desired range
    for discrete output spaces, will return an index sampled from the weighted distribution
    """
    m = np.amin([action, space.high], axis=0)
    m = np.amax([m, space.low], axis=0)
    return m

def get_action(action_proto, space, noise_process):
    """
    Will generate a tuple of (tf action, env action)
    tf action is the action needed for tensorflow and the env action is used in stepping the
    env. they are different since we parameterize discrete actions as a probability distribution and the env parameterizes it as an int
    """
    assert isinstance(space, gym.spaces.Tuple), "action space must be in for of tuple"
    tf_actions = []
    env_actions = []
    for ap, space, nop in zip(action_proto, space.spaces, noise_process):
        if isinstance(space, gym.spaces.Discrete):
            if nop.sample():
                # if we are sampling random action
                logits = np.random.random(space.n)
                ap = softmax(logits)

            env_action = sample_action(ap)

            tf_actions.append(ap)
            env_actions.append(env_action)

        elif isinstance(space, gym.spaces.Box):
            action = clip_action(ap, space) if not nop.sample() else space.sample()
            # for continuous spaces, tf action and env action are the same
            tf_actions.append(action)
            env_actions.append(action)

        else:
            assert False, "Unsupported action space type: {}".format(space)

    return tf_actions, env_actions



class ReplayMemory:
    def __init__(self, batch_size, buffer_size,
                 num_threads=1, folder="{}/tmp".format(FLAGS.train_dir)):
        self.folder = folder
        self.buffer_size = buffer_size
        gfile.MakeDirs(self.folder)
        self.fnames = []
        self.batch_size = batch_size
        self.batch_fetcher = thread_utils.ThreadedRepeater(self._get_tup, num_threads, batch_size)

    def add_transition(self, tup):
        if len(self.fnames) < self.buffer_size:
            fname ="{}/{}.npz".format(self.folder, random.random())
            self.fnames.append(fname)
        else:
            fname = random.choice(self.fnames)
        with open(fname, 'w') as f:
            np.save(f, tup)

    def _get_tup(self):
        try:
            if len(self.fnames) < self.batch_size:
                return [False]
            fname = random.choice(self.fnames)
            with open(fname, 'r') as f:
                tup = np.load(f)
            return [tup]
        except:
            print("failed to load")
            return self._get_tup()

    def get_batch(self):
        return self.batch_fetcher.run()


class DDPG:

    def __init__(self, env, train_dir, noise_process,
                 Q_func, mu_func,
                 buffer_size=1e6, batch_size=64,
                 mu_lr=.0001, Q_lr=.001, discount_factor=.99, tau=.001,
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
        self.mu_func = mu_func

        self.mu_lr = mu_lr
        self.Q_lr = Q_lr
        self.discount_factor = discount_factor
        self.tau = tau
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayMemory(batch_size, self.buffer_size)

        self.env = env
        self.batch_size = batch_size
        self.last_length = 0

        self._set_action_space()
        self.set_noise_process(noise_process)
        # placeholder for feeding observations into the algorithm
        self.obs_placeholder = tf.placeholder(
            "float", [None] + list(env.observation_space.shape), "obs_placeholder"
        )

        if len(env.observation_space.shape) == 3:
            tf.image_summary("observation", self.obs_placeholder[:, :, :, :3])
        self.next_obs_placeholder = tf.placeholder(
            "float", [None] + list(env.observation_space.shape), "next_obs_placeholder"
        )

        self.action_placeholders = [
            tf.placeholder(
                "float",
                [None] + [action_shape], "action_placeholder_{}".format(i)
            ) for i, action_shape in enumerate(self.action_shapes())
        ]

        self.step = 0
        self.global_step = tf.Variable(0, trainable=False)
        self.reward_placeholder = tf.placeholder("float", [None], "reward_placeholder")
        self.done_placeholder = tf.placeholder("float", [None], "done_placeholder")

        self._initialize_mu_vars()
        self._initialize_Q_vars()
        self._initialize_Q_loss()
        self._initialize_mu_loss()

        # ops for updating target networks
        self.update_target_networks_op = self._update_target_networks_op()
        self.target_networks_init_op = self._target_networks_init_op()
        self.train_op = self._train_op()

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

    def action_shapes(self):
        return get_action_shape(self.action_space)

    def get_Q(self, obs, action):
        # add actions to placeholders
        fd = {self.obs_placeholder: [obs]}
        for i, a in action:
            fd[self.action_placeholders[i]] = [a]
        q = self.sess.run([self.Q], feed_dict=fd)
        return q

    def get_mu(self, obs):
        mu = self.sess.run(
            self.mu_live,
            feed_dict={
                self.obs_placeholder: [obs]
            }
        )
        return [m[0] for m in mu]

    def step_env(self, action):
        if isinstance(self.env.action_space, gym.spaces.Tuple):
            return self.env.step(action)
        else:
            assert len(self.action_space.spaces) == 1 and len(action) == 1
            return self.env.step(action[0])

    def training_step(self, max_steps, train=True, render=False):
        observation = self.env.reset()
        total_reward = 0.0
        for i in xrange(max_steps):
            if render:
                self.env.render()
            action_proto = self.get_mu(observation)
            tf_action, env_action = get_action(action_proto, self.action_space, self.noise_process)
            # step the env
            new_obs, reward, done, info = self.step_env(env_action)

            total_reward += reward
            transition = (observation, tf_action, reward, new_obs, float(done))
            # add tranition to the replay buffer
            self.replay_buffer.add_transition(transition)

            if train:
                t = time.time()
                batch = self.replay_buffer.get_batch()
                batch_time = time.time() - t
                obs, acts, r, next_obs, d = zip(*batch)
                fd = {
                    self.obs_placeholder: np.array(obs),
                    self.next_obs_placeholder: np.array(next_obs),
                    self.reward_placeholder: np.array(r),
                    self.done_placeholder: np.array(d)
                }
                # add actions to feed dict
                acts = zip(*acts)
                for ind, a in enumerate(acts):
                    fd[self.action_placeholders[ind]] = np.array(a)

                if self.step % 10 == 0:
                    _, q_loss, mu_loss, sum_str = self.sess.run(
                        [self.train_op, self.Q_loss, self.mu_loss, self.summary_op],
                        feed_dict=fd
                    )
                    self.summary_writer.add_summary(sum_str, self.step)

                else:
                    _, q_loss, mu_loss = self.sess.run(
                        [self.train_op, self.Q_loss, self.mu_loss],
                        feed_dict=fd
                    )
                if (self.step % 100 == 0):
                    print("Global Step {}: Q loss = {}, mu loss = {}, Last Steps = {}, reward = {} | batch load time = {}".format(self.step, q_loss, mu_loss, self.last_length, reward, batch_time))
                    print("    Proto Action {}, TF Action {}, Env Action {}".format(action_proto, tf_action, env_action))
                self.step += 1

            if done:
                self.last_length = i + 1
                return total_reward
            observation = new_obs

        self.last_length = i + 1
        return total_reward

    def set_noise_process(self, np):
        if isinstance(np, list):
            assert len(np) == len(self.action_space.spaces)
            self.noise_process = np
        else:
            self.noise_process = [np for space in self.action_space.spaces]

    def _set_action_space(self):
        """
        sets action space, always a tuple space
        """
        if isinstance(self.env.action_space, gym.spaces.Tuple):
            self.action_space = self.env.action_space
        else:
            self.action_space = spaces.Tuple([self.env.action_space])

    def _initialize_mu_vars(self):
        with tf.variable_scope("mu") as scope:
            self.mu = mu_scale(
                self.mu_func(self.obs_placeholder, True, self.action_shapes()),
                "mu",
                self.action_space
            )
        with tf.variable_scope("mu", reuse=True) as scope:
            self.mu_live = mu_scale(
                self.mu_func(self.obs_placeholder, False, self.action_shapes(), reuse=True),
                "mu_live",
                self.action_space
            )
        with tf.variable_scope("mu_prime") as scope:
            self.mu_prime = mu_scale(
                self.mu_func(self.next_obs_placeholder, True, self.action_shapes()),
                "mu_prime",
                self.action_space
            )

    def _initialize_Q_vars(self):
        with tf.variable_scope("Q") as scope:
            self.Q = self.Q_func(self.obs_placeholder, self.action_placeholders, True)[:, 0]
        with tf.variable_scope("Q", reuse=True) as scope:
            self.Q_pred = self.Q_func(self.obs_placeholder, self.mu, False, reuse=True)[:, 0]
        with tf.variable_scope("Q_prime") as scope:
            self.Q_prime = self.Q_func(self.next_obs_placeholder, self.mu_prime, True)[:, 0]

    def _initialize_Q_loss(self):
        with tf.variable_scope("Q_loss") as scope:
            # generate loss for learning q parameters

            # the term for the non terminal values
            self.Q_non_term = tf.mul(
                self.reward_placeholder + self.discount_factor * self.Q_prime,
                -1. * (self.done_placeholder - 1.), name="Q_targets_non_term"
            )
            # the term for the terminal values
            self.Q_term = tf.mul(
                self.reward_placeholder,
                self.done_placeholder, name="Q_targets_term"
            )
            self.Q_targets = tf.add(self.Q_non_term, self.Q_term, name="Q_targets")
            tf.histogram_summary("Q_targets", self.Q_targets)
            self.Q_loss = tf.reduce_mean(tf.square(self.Q - self.Q_targets), name="Q_loss")
            tf.add_to_collection("losses", self.Q_loss)

    def _initialize_mu_loss(self):
        with tf.variable_scope("mu_loss") as scope:
            # generate loss for learning mu parameters
            self.mu_loss = tf.mul(-1.0, tf.reduce_mean(self.Q_pred), name="mu_loss")
            tf.add_to_collection("losses", self.mu_loss)


    def _update_target_networks_op(self):
        """
        Generates op for updating the target networks after each iteration
        """
        with tf.variable_scope("update_target_networks") as scope:
            Qp_Q = get_corresponding_vars("Q_prime", "Q", non_trainable_tag="_av")
            mup_mu = get_corresponding_vars("mu_prime", "mu", non_trainable_tag="_av")

            Q_tups = Qp_Q.values()
            Q_updates = [
                self.tau * Q_var + (1 - self.tau) * Qp_var for Qp_var, Q_var in Q_tups
            ]
            mu_tups = mup_mu.values()
            mu_updates = [
                self.tau * mu_var + (1 - self.tau) * mup_var for mup_var, mu_var in mu_tups
            ]

            Q_assign_ops = [
                Qp_var.assign(Q_update) for Q_update, (Qp_var, Q_var) in zip(Q_updates, Q_tups)
            ]
            mu_assign_ops = [
                mup_var.assign(mu_update) for mu_update, (mup_var, mu_var) in zip(mu_updates, mu_tups)
            ]
            with tf.control_dependencies(Q_assign_ops + mu_assign_ops):
                update_target_networks_op = tf.no_op("update_target_networks")
        return update_target_networks_op

    def _target_networks_init_op(self):
        """
        Generates op for seting initial values of the main and target networks to be the same
        """
        with tf.variable_scope("target_networks_init") as scope:
            Q_op = assign_set("Q_prime", "Q")
            mu_op = assign_set("mu_prime", "mu")
            with tf.control_dependencies([Q_op, mu_op]):
                target_networks_init_op = tf.no_op("target_networks_init")
        return target_networks_init_op

    def _train_op(self):
        """
        Generates op for one iteration of training
        """
        weight_loss = get_weight_loss()

        Q_vars = get_trainable_var_subset("Q")
        for v in Q_vars:
            print("Q:", v.name)
        mu_vars = get_trainable_var_subset("mu")
        for v in mu_vars:
            print("mu:", v.name)

        ql = tf.add_n([weight_loss, self.Q_loss], "total_Q_loss")
        ml = tf.add_n([weight_loss, self.mu_loss], "total_mu_loss")

        loss_averages_op = _add_loss_summaries([ql, ml])

        with tf.control_dependencies([loss_averages_op]):
            Q_opt = tf.train.AdamOptimizer(self.Q_lr)
            mu_opt = tf.train.AdamOptimizer(self.mu_lr)

            Q_grads = Q_opt.compute_gradients(ql, Q_vars)
            mu_grads = mu_opt.compute_gradients(ml, mu_vars)
            for grad, var in Q_grads + mu_grads:
                if grad is not None:
                    tf.histogram_summary(var.op.name + '/gradients', grad)

            self.Q_train_op = Q_opt.minimize(ql, var_list=Q_vars)

            self.mu_train_op = mu_opt.minimize(ml, var_list=mu_vars)

        for var in Q_vars + mu_vars:
            tf.histogram_summary(var.op.name, var)

        # create a single op that
        #   1) updates Q network
        #   2) updates mu network
        #   3) updates target networks
        with tf.control_dependencies([self.Q_train_op, self.mu_train_op]):
            with tf.control_dependencies([self.update_target_networks_op]):
                train_op = tf.no_op("train_op")
        return train_op

class SiameseImageDDPG(DDPG):
    """
    Version of ddpg for image observations where the Q and mu networks share image processing features with a seperate image processing network
    """
    def __init__(self, env, train_dir, noise_process,
                 Q_func, mu_func, im_func,
                 buffer_size=1e6, batch_size=64,
                 mu_lr=.0001, Q_lr=.001, im_lr=.001, discount_factor=.99, tau=.001,
                 checkpoint_path=None):
        self.im_func = im_func
        self.im_lr = im_lr
        DDPG.__init__(self, env, train_dir, noise_process,
                 Q_func, mu_func,
                 buffer_size=buffer_size, batch_size=batch_size,
                 mu_lr=mu_lr, Q_lr=Q_lr, discount_factor=discount_factor, tau=tau,
                 checkpoint_path=checkpoint_path)

    def _initialize_mu_vars(self):
        # process the observations with the image network
        with tf.variable_scope("im") as scope:
            self.obs_features = self.im_func(self.obs_placeholder, True)
        with tf.variable_scope("im_prime") as scope:
            self.next_obs_features = self.im_func(self.next_obs_placeholder, False)

        with tf.variable_scope("mu") as scope:
            self.mu = mu_scale(
                self.mu_func(self.obs_features, True, self.action_shapes()),
                "mu",
                self.action_space
            )
        with tf.variable_scope("mu", reuse=True) as scope:
            self.mu_live = mu_scale(
                self.mu_func(self.obs_features, False, self.action_shapes(), reuse=True),
                "mu_live",
                self.action_space
            )
        with tf.variable_scope("mu_prime") as scope:
            self.mu_prime = mu_scale(
                self.mu_func(self.next_obs_features, False, self.action_shapes()),
                "mu_prime",
                self.action_space
            )

    def _initialize_Q_vars(self):
        with tf.variable_scope("Q") as scope:
            self.Q = self.Q_func(self.obs_features, self.action_placeholders, True)[:, 0]
        with tf.variable_scope("Q", reuse=True) as scope:
            self.Q_pred = self.Q_func(self.obs_features, self.mu, True, reuse=True)[:, 0]
        with tf.variable_scope("Q_prime") as scope:
            self.Q_prime = self.Q_func(self.next_obs_features, self.mu_prime, False)[:, 0]

    def _update_target_networks_op(self):
        """
        Generates op for updating the target networks after each iteration
        """
        with tf.variable_scope("update_target_networks") as scope:
            Qp_Q = get_corresponding_vars("Q_prime", "Q", non_trainable_tag="_av")
            mup_mu = get_corresponding_vars("mu_prime", "mu", non_trainable_tag="_av")
            imp_im = get_corresponding_vars("im_prime", "im", non_trainable_tag="_av")

            Q_tups = Qp_Q.values()
            Q_updates = [
                self.tau * Q_var + (1 - self.tau) * Qp_var for Qp_var, Q_var in Q_tups
            ]
            mu_tups = mup_mu.values()
            mu_updates = [
                self.tau * mu_var + (1 - self.tau) * mup_var for mup_var, mu_var in mu_tups
            ]

            im_tups = imp_im.values()
            im_updates = [
                self.tau * im_var + (1 - self.tau) * imp_var for imp_var, im_var in im_tups
            ]

            Q_assign_ops = [
                Qp_var.assign(Q_update) for Q_update, (Qp_var, Q_var) in zip(Q_updates, Q_tups)
            ]
            mu_assign_ops = [
                mup_var.assign(mu_update) for mu_update, (mup_var, mu_var) in zip(mu_updates, mu_tups)
            ]
            im_assign_ops = [
                imp_var.assign(im_update) for im_update, (imp_var, im_var) in zip(im_updates, im_tups)
            ]
            with tf.control_dependencies(Q_assign_ops + mu_assign_ops + im_assign_ops):
                update_target_networks_op = tf.no_op("update_target_networks")
        return update_target_networks_op

    def _target_networks_init_op(self):
        """
        Generates op for seting initial values of the main and target networks to be the same
        """
        with tf.variable_scope("target_networks_init") as scope:
            Q_op = assign_set("Q_prime", "Q")
            mu_op = assign_set("mu_prime", "mu")
            im_op = assign_set("im_prime", "im")
            with tf.control_dependencies([Q_op, mu_op, im_op]):
                target_networks_init_op = tf.no_op("target_networks_init")
        return target_networks_init_op

    def _train_op(self):
        """
        Generates op for one iteration of training
        """
        weight_loss = get_weight_loss()

        Q_vars = get_trainable_var_subset("Q")
        for v in Q_vars:
            print("Q:", v.name)
        mu_vars = get_trainable_var_subset("mu")
        for v in mu_vars:
            print("mu:", v.name)

        im_vars = get_trainable_var_subset("im")
        for v in im_vars:
            print("im:", v.name)

        ql = tf.add_n([weight_loss, self.Q_loss], "total_Q_loss")
        ml = tf.add_n([weight_loss, self.mu_loss], "total_mu_loss")
        il = tf.add_n([weight_loss, self.Q_loss], "total_im_loss")

        loss_averages_op = _add_loss_summaries([ql, ml])

        with tf.control_dependencies([loss_averages_op]):
            Q_opt = tf.train.AdamOptimizer(self.Q_lr)
            mu_opt = tf.train.AdamOptimizer(self.mu_lr)
            im_opt = tf.train.AdamOptimizer(self.im_lr)

            Q_grads = Q_opt.compute_gradients(ql, Q_vars)
            mu_grads = mu_opt.compute_gradients(ml, mu_vars)
            im_grads = im_opt.compute_gradients(il, im_vars)
            vs = {}
            for grad, var in Q_grads + mu_grads + im_grads:
                if grad is not None and var.op.name not in vs:
                    tf.histogram_summary(var.op.name + '/gradients', grad)
                    vs[var.op.name] = True

            self.Q_train_op = Q_opt.minimize(ql, var_list=Q_vars)
            self.mu_train_op = mu_opt.minimize(ml, var_list=mu_vars)
            self.im_train_op = im_opt.minimize(il, var_list=im_vars)

        for var in Q_vars + mu_vars + im_vars:
            tf.histogram_summary(var.op.name, var)

        # create a single op that
        #   1) updates Q network
        #   2) updates mu network
        #   3) updates target networks
        with tf.control_dependencies([self.Q_train_op, self.mu_train_op, self.im_train_op]):
            with tf.control_dependencies([self.update_target_networks_op]):
                train_op = tf.no_op("train_op")
        return train_op

class SiameseImageDDPGWithReconstruction(SiameseImageDDPG):
    def __init__(self, env, train_dir, noise_process,
                 Q_func, mu_func, im_func, dec_func,
                 buffer_size=1e6, batch_size=64,
                 mu_lr=.0001, Q_lr=.001, im_lr=.001, discount_factor=.99, tau=.001,
                 checkpoint_path=None):
        self.dec_func = dec_func
        self.im_lr = im_lr
        SiameseImageDDPG.__init__(
            self, env, train_dir, noise_process,
            Q_func, mu_func, im_func,
            buffer_size=buffer_size, batch_size=batch_size,
            mu_lr=mu_lr, Q_lr=Q_lr, discount_factor=discount_factor, tau=tau,
            checkpoint_path=checkpoint_path
        )

    def _initialize_mu_vars(self):
        # process the observations with the image network
        with tf.variable_scope("im") as scope:
            self.obs_features = self.im_func(self.obs_placeholder, True)
        with tf.variable_scope("dec") as scope:
            self.decoded_obs = self.dec_func(self.obs_features, self.action_placeholders, True)
            tf.image_summary("decoded_obs", self.decoded_obs[:, :, :, :3])
        with tf.variable_scope("im_prime") as scope:
            self.next_obs_features = self.im_func(self.next_obs_placeholder, False)

        with tf.variable_scope("mu") as scope:
            self.mu = mu_scale(
                self.mu_func(self.obs_features, True, self.action_shapes()),
                "mu",
                self.action_space
            )
        with tf.variable_scope("mu", reuse=True) as scope:
            self.mu_live = mu_scale(
                self.mu_func(self.obs_features, False, self.action_shapes(), reuse=True),
                "mu_live",
                self.action_space
            )
        with tf.variable_scope("mu_prime") as scope:
            self.mu_prime = mu_scale(
                self.mu_func(self.next_obs_features, False, self.action_shapes()),
                "mu_prime",
                self.action_space
            )

    def _initialize_mu_loss(self):
        with tf.variable_scope("mu_loss") as scope:
            # generate loss for learning mu parameters
            self.mu_loss = tf.mul(-1.0, tf.reduce_mean(self.Q_pred), name="mu_loss")
            tf.add_to_collection("losses", self.mu_loss)
        with tf.variable_scope("im_loss") as scope:
            self.im_loss = tf.nn.l2_loss(self.next_obs_placeholder - self.decoded_obs, name="im_loss")
            if FLAGS.r_loss_decay is not None:
                self.r_loss_scale = tf.Variable(1.0, trainable=False)
                dec = 1.0 / FLAGS.r_loss_decay
                decayed_r_loss = tf.maximum(self.r_loss_scale - dec, 0.0)
                self.decay_r_loss_scale_op = self.r_loss_scale.assign(decayed_r_loss)
                self.im_loss = self.r_loss_scale * self.im_loss
            tf.add_to_collection("losses", self.im_loss)

    def _train_op(self):
        """
        Generates op for one iteration of training
        """
        weight_loss = get_weight_loss()

        Q_vars = get_trainable_var_subset("Q")
        for v in Q_vars:
            print("Q:", v.name)
        mu_vars = get_trainable_var_subset("mu")
        for v in mu_vars:
            print("mu:", v.name)

        im_vars = get_trainable_var_subset("im")
        for v in im_vars:
            print("im:", v.name)
        dec_vars = get_trainable_var_subset("dec")
        for v in dec_vars:
            print("dec:", v.name)

        ql = tf.add_n([weight_loss, self.Q_loss], "total_Q_loss")
        ml = tf.add_n([weight_loss, self.mu_loss], "total_mu_loss")
        il = tf.add_n([weight_loss, self.im_loss, self.Q_loss], "total_im_loss")

        loss_averages_op = _add_loss_summaries([ql, ml])

        with tf.control_dependencies([loss_averages_op]):
            Q_opt = tf.train.AdamOptimizer(self.Q_lr)
            mu_opt = tf.train.AdamOptimizer(self.mu_lr)
            im_opt = tf.train.AdamOptimizer(self.im_lr)

            Q_grads = Q_opt.compute_gradients(ql, Q_vars)
            mu_grads = mu_opt.compute_gradients(ml, mu_vars)
            im_grads = im_opt.compute_gradients(il, im_vars + dec_vars)
            vs = {}
            for grad, var in Q_grads + mu_grads + im_grads:
                if grad is not None and var.op.name not in vs:
                    tf.histogram_summary(var.op.name + '/gradients', grad)
                    vs[var.op.name] = True

            self.Q_train_op = Q_opt.minimize(ql, var_list=Q_vars)

            self.mu_train_op = mu_opt.minimize(ml, var_list=mu_vars)

            self.im_train_op = im_opt.minimize(il, var_list=im_vars+dec_vars)

        for var in Q_vars + mu_vars + im_vars + dec_vars:
            tf.histogram_summary(var.op.name, var)

        # create a single op that
        #   1) updates Q network
        #   2) updates mu network
        #   3) updates target networks
        with tf.control_dependencies([self.Q_train_op, self.mu_train_op, self.im_train_op, self.decay_r_loss_scale_op]):
            with tf.control_dependencies([self.update_target_networks_op]):
                train_op = tf.no_op("train_op")
        return train_op


if __name__ == "__main__":
    env_n = FLAGS.env
    if env_n == "AcrobotContinuous":
            env = acrobot_continuous.AcrobotEnv()
    elif env_n == "CartPoleContinuous":
        env = pole_env.CartPoleEnv()
    else:
        env = gym.make(env_n)
    if FLAGS.image:
        env = VisualEnv(env, 3, 64)



    if gfile.Exists(FLAGS.train_dir):
        gfile.DeleteRecursively(FLAGS.train_dir)
    gfile.MakeDirs(FLAGS.train_dir)
    logfile = open("{}/log.txt".format(FLAGS.train_dir), 'w')

    n = 1.
    noise_process = RandomSampleProcess(n)
    if FLAGS.image:
        if not FLAGS.siamese:
            ddpg = DDPG(env, FLAGS.train_dir, noise_process, networks.Q_network_conv, networks.mu_network_conv, batch_size=16, buffer_size=FLAGS.buffer_size)
        else:
            if FLAGS.reproject:
                ddpg = SiameseImageDDPGWithReconstruction(env, FLAGS.train_dir, noise_process, networks.Q_network_im, networks.mu_network_im, networks.im_network, networks.dec_network, batch_size=16, buffer_size=FLAGS.buffer_size)
            else:
                ddpg = SiameseImageDDPG(env, FLAGS.train_dir, noise_process, networks.Q_network_im, networks.mu_network_im, networks.im_network, batch_size=16, buffer_size=FLAGS.buffer_size)
    else:
        if FLAGS.norm is None:
            ddpg = DDPG(env, FLAGS.train_dir, noise_process, networks.Q_network, networks.mu_network, buffer_size=FLAGS.buffer_size)
        elif FLAGS.norm == "weight":
            ddpg = DDPG(env, FLAGS.train_dir, noise_process, networks.Q_network_wn, networks.mu_network_wn, buffer_size=FLAGS.buffer_size)
        elif FLAGS.norm == "layer":
            ddpg = DDPG(env, FLAGS.train_dir, noise_process, networks.Q_network_ln, networks.mu_network_ln, buffer_size=FLAGS.buffer_size)
        elif FLAGS.norm == "batch":
            ddpg = DDPG(env, FLAGS.train_dir, noise_process, networks.Q_network_bn, networks.mu_network_bn, buffer_size=FLAGS.buffer_size)
        else:
            assert False

    while len(ddpg.replay_buffer.fnames) < ddpg.batch_size:
        ddpg.training_step(FLAGS.max_steps, False, False)
        print(len(ddpg.replay_buffer.fnames))
    _ = ddpg.replay_buffer.get_batch()
    rewards = []
    for i in range(5000):
        if i % 1000 == 0:
            n /= 10
            ddpg.set_noise_process(RandomSampleProcess(n))
            print("reducing noise to {}".format(n))
        if i % 10 == 0:
            print("Completed {} Episodes".format(i))
            if FLAGS.vis:
                ddpg.set_noise_process(RandomSampleProcess(0.0))
                r = ddpg.training_step(FLAGS.max_steps, False, FLAGS.vis)
                print("Reward: {}".format(r))
                ddpg.set_noise_process(RandomSampleProcess(n))
        r = ddpg.training_step(FLAGS.max_steps, True, False)
        logfile.write("{}, {}, {}\n".format(ddpg.step, i, r))
        logfile.flush()
        if len(rewards) < 100:
            rewards.append(r)
        else:
            rewards = rewards[1:]
            rewards.append(r)
            ave_r = sum(rewards) / 100.
            print(FLAGS.train_dir, "ave_r", ave_r)
            if ave_r > FLAGS.target:
                print("Solved in {} episodes".format(i))
                break
    gfile.DeleteRecursively("{}/tmp".format(FLAGS.train_dir))
