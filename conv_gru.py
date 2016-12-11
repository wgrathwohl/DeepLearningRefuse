"""
Implementation of Convolutional GRU and trains an autoencoder to use it
"""
from tensorflow.python.platform import gfile
import tensorflow as tf
import os
from layers import *
import data_handler

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("checkpoint_path", None, "if used will restore the model")
tf.app.flags.DEFINE_integer('batch_size', 16, "number of seperate models in a batch")
tf.app.flags.DEFINE_integer('num_frames', 10, "number of frames in a video")
tf.app.flags.DEFINE_integer('read_frames', 6, "number of frames to encode")
tf.app.flags.DEFINE_string('train_dir', None, "the training directory")
tf.app.flags.DEFINE_integer('max_steps', 1000000, "total number of steps to run")
tf.app.flags.DEFINE_boolean("random", True, "the random flag for batches")
tf.app.flags.DEFINE_integer("iterations_per_decay", 10000, "num iterations to decay")
tf.app.flags.DEFINE_integer("iterations_per_valid", 1000, "num iterations to validation")
tf.app.flags.DEFINE_float('initial_learning_rate', .00001, "initial learning rate")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1, "learning rate decay factor")

MIN_IMAGE_VALUE = 0.0
MAX_IMAGE_VALUE = 1.0
IMAGE_VALUE_RANGE = MAX_IMAGE_VALUE - MIN_IMAGE_VALUE
IMAGE_SHIFT = (MAX_IMAGE_VALUE + MIN_IMAGE_VALUE) / 2.0
IMAGE_SCALE = 2.0 / IMAGE_VALUE_RANGE

def shift_images(ims):
    return (ims - IMAGE_SHIFT) * IMAGE_SCALE
def unshift_images(ims):
    return (ims / IMAGE_SCALE) + IMAGE_SHIFT

class ConvGRUCell(tf.nn.rnn_cell.RNNCell):
    """
    Implements a standard conv gru cell
    """
    def __init__(self, n_outputs, filter_shape, stddev, wd, filter_stride=(1, 1), base_layer=conv_layer, nonlinearity=tf.nn.relu, output_pooling=(2, 2)):
        self.n_outputs = n_outputs
        self.filter_shape = filter_shape
        self.stddev = stddev
        self.wd = wd
        self.filter_stride = filter_stride
        self.base_layer = base_layer
        self.nonlinearity = nonlinearity
        self.output_pooling = output_pooling
        self._state_size = None
        self._output_size = None

    @property
    def state_size(self):
        assert self._state_size is not None
        return self._state_size

    @property
    def output_size(self):
        assert self._output_size is not None
        return self._output_size

    def __call__(self, inputs, state, scope_name, prev_layer_state=None, reuse=False):
        # prev_layer_state is the hidden state from the current timestep of the previous layer
        with tf.variable_scope(scope_name) as scope:
            if prev_layer_state is None:
                gate_input = tf.concat(3, [inputs, state], name="gate_input")
            else:
                gate_input = tf.concat(3, [inputs, state, prev_layer_state], name="gate_input")
            z = self.base_layer(
                gate_input, "z_conv", self.n_outputs, self.filter_shape,
                self.stddev, 0.0 if reuse else self.wd,
                filter_stride=self.filter_stride, nonlinearity=tf.sigmoid
            )
            r = self.base_layer(
                gate_input, "r_conv", self.n_outputs, self.filter_shape,
                self.stddev, 0.0 if reuse else self.wd,
                filter_stride=self.filter_stride, nonlinearity=tf.sigmoid
            )
            h_input = tf.concat(3, [inputs, r * state], name="h_input")
            h_tilde = self.base_layer(
                h_input, "h_tilde_conv", self.n_outputs, self.filter_shape,
                self.stddev, 0.0 if reuse else self.wd,
                filter_stride=self.filter_stride, nonlinearity=self.nonlinearity
            )
            h = tf.identity(((1 - z) * state) + (z * h_tilde), name="h")
            if self.output_pooling is not None:
                out = tf.nn.avg_pool(
                    h,
                    (1, self.output_pooling[0], self.output_pooling[1], 1),
                    (1, self.output_pooling[0], self.output_pooling[1], 1),
                    "SAME"
                )
            else:
                out = h

            self._state_size = h.get_shape().as_list()
            self._output_size = out.get_shape().as_list()
        return out, h


class ConvGruMultiCell(tf.nn.rnn_cell.RNNCell):
    """
    Implements a stacked conv gru cell
    cells is a list of base cells
    """
    def __init__(self, cells):
        self.cells = cells
        self._state_size = None
        self._output_size = None

    @property
    def state_size(self):
        assert self._state_size is not None
        return self._state_size

    @property
    def output_size(self):
        assert self._output_size is not None
        return self._output_size

    def zero_state(self, inputs, dtype):
        for i, cell in enumerate(self.cells):
            if i == 0:
                cell._state_size = inputs.get_shape().as_list()
            else:
                cell._state_size = [e for e in self.cells[i-1]._state_size]
                cell._state_size[1] = self.cells[i-1]._state_size[1] / self.cells[i-1].output_pooling[0]
                cell._state_size[2] = self.cells[i-1]._state_size[2] / self.cells[i-1].output_pooling[1]
            cell._state_size[-1] = cell.n_outputs
        return [tf.zeros(c._state_size) for c in self.cells]


    def __call__(self, inputs, states, scope_name, reuse=False):
        """
        Generates state and output ops
        inputs are lists of tensors, they must have the same length and the same length as cells
        """
        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            next_states = []
            outputs = []
            for i, cell in enumerate(self.cells):
                with tf.variable_scope("Cell%d" % i) as cell_scope:
                    cur_state = states[i]
                    if i == 0:
                        # this is the first cell so the input is "inputs"
                        cur_inp = inputs
                    else:
                        cur_inp = outputs[i-1]

                    cur_out, new_state = cell(
                        cur_inp, cur_state, "Cell%d_internal" % i,
                        prev_layer_state=None, reuse=reuse
                    )
                    outputs.append(cur_out)
                    next_states.append(new_state)
            out = outputs[-1]
            self._output_size = out.get_shape().as_list()
            return out, next_states

class DeconvGRUCell(ConvGRUCell):
    def __init__(self, n_outputs, filter_shape, stddev, wd, out_shape,
                 conv_filter_stride=(1, 1), deconv_filter_stride=(2, 2),
                 base_conv_layer=conv_layer, base_deconv_layer=deconv_layer,
                 conv_nonlinearity=tf.nn.relu, deconv_nonlinearity=tf.nn.relu):
        ConvGRUCell.__init__(
            self, n_outputs, filter_shape, stddev, wd, filter_stride=(1, 1),
            base_layer=base_conv_layer, nonlinearity=conv_nonlinearity, output_pooling=None
        )
        self.base_deconv_layer = base_deconv_layer
        self.deconv_filter_stride = deconv_filter_stride
        self.deconv_nonlinearity = deconv_nonlinearity
        self.out_shape = out_shape

    def __call__(self, inputs, state, scope_name, prev_layer_state=None, reuse=False):
        with tf.variable_scope(scope_name):
            out, h = ConvGRUCell.__call__(
                self, inputs, state, "{}_conv_gru".format(scope_name),
                prev_layer_state=None, reuse=False
            )
            print self.deconv_nonlinearity
            deconv_out = self.base_deconv_layer(
                h, "deconv_out", self.out_shape,
                self.filter_shape, self.deconv_filter_stride,
                self.stddev, 0.0 if reuse else self.wd,
                nonlinearity=self.deconv_nonlinearity
            )
            self._output_size = deconv_out.get_shape().as_list()
        return deconv_out, h

class DeconvGruMultiCell(tf.nn.rnn_cell.RNNCell):
    """
    Implements a stacked conv gru cell
    cells is a list of base cells
    """
    def __init__(self, cells):
        self.cells = cells

    @property
    def state_size(self):
        assert self._state_size is not None
        return self._state_size

    @property
    def output_size(self):
        assert self._output_size is not None
        return self._output_size

    def zero_state(self, inputs, dtype):
        cell_shapes = []
        for i, cell in enumerate(self.cells):
            if i == 0:
                cell._state_size = inputs.get_shape().as_list()
            else:
                cell._state_size = self.cells[i-1].out_shape
            cell._state_size[-1] = cell.n_outputs

            cell_shapes.append(cell._state_size)
        return [tf.zeros(s) for s in cell_shapes]


    def __call__(self, inputs, states, scope_name, reuse=False):
        """
        Generates state and output ops
        inputs are lists of tensors, they must have the same length and the same length as cells
        """
        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            next_states = []
            outputs = []
            for i, cell in enumerate(self.cells):
                with tf.variable_scope("Cell%d" % i) as cell_scope:
                    cur_state = states[i]
                    if i == 0:
                        # this is the first cell so the input is "inputs"
                        cur_inp = inputs
                    else:
                        cur_inp = outputs[i-1]
                    #print(cur_inp.get_shape().as_list(), cur_state.get_shape().as_list())
                    cur_out, new_state = cell(
                        cur_inp, cur_state, "Cell%d_internal" % i,
                        prev_layer_state=None, reuse=reuse
                    )
                    outputs.append(cur_out)
                    next_states.append(new_state)
            # output is the last cell's output
            return outputs[-1], next_states

def train():
    # testing
    IMAGE_SIZE = (64, 64, 1)
    NUM_CHANNELS = IMAGE_SIZE[-1]
    # read in dataset
    # Dataset generator
    NUM_DIGITS=1
    dataset = data_handler.BouncingMNISTDataHandler(
        num_frames=FLAGS.num_frames, batch_size=FLAGS.batch_size,
        image_size=IMAGE_SIZE[0], num_digits=NUM_DIGITS
    )

    vid_placeholder = tf.placeholder(
        "float", [FLAGS.batch_size, FLAGS.num_frames, IMAGE_SIZE[0], IMAGE_SIZE[1], 1],
        name="vid_placeholder"
    )
    shifted_vids = shift_images(vid_placeholder)
    ims = [tf.squeeze(im, [1]) for im in tf.split(1, FLAGS.num_frames, shifted_vids)]
    for i, im in enumerate(ims):
        tf.image_summary("image_{}".format(i), im, max_images=1)


    # vid_pyr = zip(ims, ims_ds, ims_ds2, ims_ds3, ims_ds4)
    rnn_inputs = ims[:FLAGS.read_frames]
    decoder_targets = ims[FLAGS.read_frames:]

    cells = [ConvGRUCell(16, (3, 3), .001, .001, (1, 1), base_layer=layer_normalized_conv_layer, nonlinearity=tf.nn.elu) for i in range(4)]
    multi_cell = ConvGruMultiCell(cells)
    init_state = multi_cell.zero_state(rnn_inputs[0], float)

    outs = []
    states = []
    for i, inp in enumerate(rnn_inputs):
        if i == 0:
            out, state = multi_cell(inp, init_state, "encoder")
        else:
            out, state = multi_cell(inp, states[i-1], "encoder", reuse=True)
        outs.append(out)
        states.append(state)


    # grab encoding from the top layer of the encoder from the last frame
    encoding = outs[-1]
    tf.histogram_summary("encoding", encoding)

    # set up decoder
    s = encoding.get_shape().as_list()
    d_shapes = [[s[0], s[1]*2**(i+1), s[2]*2**(i+1), s[3]] for i in range(4)]
    d_shapes[-1][-1] = NUM_CHANNELS

    d_cells = [DeconvGRUCell(16, (3, 3), .001, .001, d_shape, base_conv_layer=layer_normalized_conv_layer, conv_nonlinearity=tf.nn.elu, deconv_nonlinearity=tf.tanh if i == 3 else tf.nn.elu) for i, d_shape in enumerate(d_shapes)]
    d_multi_cell = DeconvGruMultiCell(d_cells)
    d_init_state = d_multi_cell.zero_state(encoding, float)

    d_outs = []
    d_states = []
    for i in range(len(decoder_targets)):
        if i == 0:
            out, state = d_multi_cell(encoding, d_init_state, "decoder")
        else:
            out, state = d_multi_cell(encoding, d_states[i-1], "decoder", reuse=True)
        d_outs.append(out)
        d_states.append(state)

    pred_frames = d_outs
    for i, im in enumerate(pred_frames):
        tf.image_summary("pred_image_{}".format(i), unshift_images(im), max_images=1)

    assert len(pred_frames) == len(decoder_targets)
    pred_losses = [tf.nn.l2_loss(t-p) for t, p in zip(decoder_targets, pred_frames)]
    pred_loss = tf.add_n(pred_losses, name="pred_loss")
    tf.scalar_summary("pred_loss", pred_loss)
    tf.add_to_collection("losses", pred_loss)
    losses = tf.get_collection("losses")
    total_loss = tf.add_n(losses, name="total_loss")
    opt = tf.train.AdamOptimizer(.001)
    grads = opt.compute_gradients(total_loss)
    for g, v in grads:
        tf.histogram_summary("{}_gradients".format(v.name), g)
    train_op = opt.apply_gradients(grads)

    best_saver = tf.train.Saver(tf.all_variables())

    # Create a saver.
    saver = tf.train.Saver(tf.trainable_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    if FLAGS.checkpoint_path is not None:
        print("Loading Checkpoint {}".format(FLAGS.checkpoint_path))
        restorer = tf.train.Saver(tf.all_variables())
        restorer.restore(sess, FLAGS.checkpoint_path)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)

    best_valid_loss = np.inf
    for step in xrange(FLAGS.max_steps):
        batch = dataset.GetBatch(random=FLAGS.random)[0]
        fd = {vid_placeholder: batch}
        if step % 100 == 0:
            _, pl, summary_str = sess.run([train_op, pred_loss, summary_op], feed_dict=fd)
            summary_writer.add_summary(summary_str, step)
        else:
            _, pl = sess.run([train_op, pred_loss], feed_dict=fd)

        if step % 10 == 0:
            print("Step {}, Loss = {}".format(step, pl))

        if step % FLAGS.iterations_per_valid == 0:
            v_loss = 0.
            v_steps = 100
            for v_step in range(v_steps):
                batch = dataset.GetTestBatch(random=FLAGS.random)[0]
                fd = {vid_placeholder: batch}
                pl, = sess.run([pred_loss], feed_dict=fd)
                v_loss += pl
            v_loss /= v_steps
            print("Valid loss = {}".format(v_loss))
            if v_loss < best_valid_loss:
                print("Best valid loss")
                best_valid_loss = v_loss
                checkpoint_path = os.path.join(FLAGS.train_dir, "best_model.ckpt")
                best_saver.save(sess, checkpoint_path, global_step=step)





def main(argv=None):
    if gfile.Exists(FLAGS.train_dir):
        gfile.DeleteRecursively(FLAGS.train_dir)
    gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
