"""
Will house networks
"""
import tensorflow as tf
from layers import *

def mu_network(features, is_training, n_outputs, reuse=False):
    test = not is_training
    n_features_fc1 = 400
    fc1 = linear_layer(
        features, "fc1", n_features_fc1,
        .003, .00 if not reuse else 0.0
    )
    print(fc1.get_shape().as_list(), "fc1 shape")
    n_features_fc2 = 300
    fc2 = linear_layer(
        fc1, "fc2", n_features_fc2,
        .003, .00 if not reuse else 0.0
    )
    print(fc2.get_shape().as_list(), "fc2 shape")
    outs = [
        linear_layer(
            fc2, "out_{}".format(i), n_out,
            .003, .00 if not reuse else 0.0,
            nonlinearity=None
        ) for i, n_out in n_outputs
    ]
    print([out.get_shape().as_list() for out in outs], "out shape")
    return outs

def Q_network(features, actions, is_training, reuse=False):
    test = not is_training
    n_features_fc1 = 400
    fc1 = linear_layer(
        features, "fc1", n_features_fc1,
        .003, .001 if not reuse else 0.0
    )
    print(fc1.get_shape().as_list(), "fc1 shape")
    fc1_with_actions = tf.concat(1, [fc1] + actions, "fc1_with_actions")
    print(fc1_with_actions.get_shape().as_list(), "fc1 with actions shape")
    n_features_fc2 = 300
    fc2 = linear_layer(
        fc1_with_actions, "fc2", n_features_fc2,
        .003, .001 if not reuse else 0.0
    )
    print(fc2.get_shape().as_list(), "fc2 shape")
    out = linear_layer(
        fc2, "out", 1,
        .003, .001 if not reuse else 0.0,
        nonlinearity=None
    )
    print(out.get_shape().as_list(), "out shape")
    return out

def DQL_net(features, n_actions, is_training, reuse=False, return_features=False):
    test = not is_training
    n_features_fc1 = 400
    fc1, fp1 = linear_layer(
        features, "fc1", n_features_fc1,
        .003, 0.0, return_params=True
    )
    print(fc1.get_shape().as_list(), "fc1 shape")
    n_features_fc2 = 300
    fc2, fp2 = linear_layer(
        fc1, "fc2", n_features_fc2,
        .003, 0.0, return_params=True
    )
    print(fc2.get_shape().as_list(), "fc2 shape")
    out, op = linear_layer(
        fc2, "out", n_actions,
        .003, 0.0, return_params=True,
        nonlinearity=None
    )
    print(out.get_shape().as_list(), "out shape")
    if return_features:
        return out, fc2
    else:
        return out, [fp1, fp2, op]

def DQL_net_feats(features, is_training, reuse=False):
    test = not is_training
    n_features_fc1 = 400
    fc1, p1 = linear_layer(
        features, "fc1", n_features_fc1,
        .003, 0,#.0005 if not reuse else 0.0,
        return_params=True
    )
    print(fc1.get_shape().as_list(), "fc1 shape")
    n_features_fc2 = 300
    fc2, p2 = linear_layer(
        fc1, "fc2", n_features_fc2,
        .003, 0,#.0005 if not reuse else 0.0,
        return_params=True
    )
    print(fc2.get_shape().as_list(), "fc2 shape")

    return fc2, [p1, p2]

def DQL_net_Q(features, n_actions, is_training, reuse=False):
    out, p = linear_layer(
        features, "out", n_actions,
        .003, 0, return_params=True,
        nonlinearity=None
    )
    return out, [p]

def DQL_net_comp(action_sequence, sequence_lengths, cell_size=32):
    print(action_sequence.get_shape().as_list(), "one hot action sequence size")
    max_length = action_sequence.get_shape().as_list()[1]
    cell = tf.nn.rnn_cell.GRUCell(cell_size)
    split_action_seq = [tf.squeeze(t, [1]) for t in tf.split(1, max_length, action_sequence)]

    rnn_outputs, rnn_states = tf.nn.rnn(
        cell,
        split_action_seq,
        dtype=tf.float32,
        sequence_length=sequence_lengths
    )
    log(rnn_states)
    return rnn_states

def DQL_net_state_prop(action_sequence, sequence_lengths, state):
    print(action_sequence.get_shape().as_list(), "one hot action sequence size")
    cell_size = state.get_shape().as_list()[1]
    cell = tf.nn.rnn_cell.GRUCell(cell_size)

    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
        cell,
        action_sequence,
        dtype=tf.float32,
        sequence_length=sequence_lengths,
        initial_state=state
    )
    log(rnn_outputs)
    # return all but last timestep
    return rnn_outputs

def DQL_net_dec(start_obs_feats, compressed_action_feats, out_space):
    feats_conc = tf.concat(1, [start_obs_feats, compressed_action_feats])
    print("conc feats shape", feats_conc.get_shape().as_list())
    n_features_fc1 = 400
    fc1 = linear_layer(
        feats_conc, "dec_fc1", n_features_fc1,
        .003, 0,#.0005,
        return_params=False
    )
    print(fc1.get_shape().as_list(), "fc1 shape")
    print("out shape", out_space.shape)
    n_features_fc2 = out_space.shape[0]
    fc2 = linear_layer(
        fc1, "dec_fc2", n_features_fc2,
        .003, 0,#.0005,
        return_params=False, nonlinearity=None
    )
    # print(fc2.get_shape().as_list(), "fc2 shape")
    # # get into [0, 1] range
    # z1out = (fc2 + 1.) / 2.
    # # scale to [out_space.high, out_space.low] range
    # s_out = (z1out * (out_space.high - out_space.low)) + out_space.low
    # print(s_out.get_shape().as_list(), "scaled dec out shape")

    #return s_out
    return fc2


def DQL_net_conv_preprocess(features, max_frames, reuse=False):
    # shape should be [batch_size, max_frames, h, w, c]
    log(features)
    assert len(features.get_shape().as_list()) == 5
    im_shape = features.get_shape().as_list()[2:]
    unwrapped_features = tf.reshape(features, [-1] + im_shape)
    # shape should be [batch_size * max_frames, h, w, c]
    # apply conv stuff
    log(unwrapped_features)
    conv1, cp1 = conv_layer(
        unwrapped_features, "conv1", 8,
        [5, 5], "MSFT", 0.0 if reuse else .0005,
        filter_stride=[2, 2], return_params=True
    )
    log(conv1)
    conv2, cp2 = conv_layer(
        conv1, "conv2", 8,
        [5, 5], "MSFT", 0.0 if reuse else .0005,
        filter_stride=[2, 2], return_params=True
    )
    log(conv2)
    conv3, cp3 = conv_layer(
        conv2, "conv3", 8,
        [5, 5], "MSFT", 0.0 if reuse else .0005,
        filter_stride=[2, 2], return_params=True
    )
    log(conv3)
    conv3_flat, dim = reshape_conv_layer(conv3)
    log(conv3_flat)
    # shape should be [batch_size * max_frames, dim]
    # wrap back into [batch_size, max_frames, dim]
    wrapped_conv3_flat = tf.reshape(conv3_flat, [-1, max_frames, dim])
    log(wrapped_conv3_flat)
    return wrapped_conv3_flat

def DQL_net_conv_recurrent(features, n_actions, max_steps, initial_state=None, cell_size=64):
    batch_size, max_steps, h, w, c = features.get_shape().as_list()
    rnn_inputs = DQL_net_conv_preprocess(features, max_steps)

    cell = tf.nn.rnn_cell.GRUCell(cell_size)
    sl = sequence_length(features)
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
        cell,
        rnn_inputs,
        dtype=tf.float32,
        sequence_length=sl,
        initial_state=initial_state
    )
    log(rnn_outputs)
    # shape should be [batch_size, max_frames, cell_size]
    # reshape to [batch_size * max_frames, cell_size]
    q_value_inputs = tf.reshape(rnn_outputs, [-1, cell_size])
    log(q_value_inputs)
    unwrapped_q = linear_layer(
        q_value_inputs, "unwrapped_q", n_actions,
        .005, 0.0 if reuse else .0005,
        nonlinearity=None, return_params=False
    )
    # shape should be [batch_size * max_frames, n_actions]
    # reshape to [batch_size, max_frames, n_actions]
    q_values = tf.reshape(unwrapped_q, [-1, max_steps, n_actions])
    log(q_values)
    return q_values, rnn_states


# def DQL_net_conv_preprocess(features, im_shape, reuse=False):
#     log(features)
#     conv_features = tf.reshape(features, [-1] + im_shape)
#     log(conv_features)
#     conv1, cp1 = conv_layer(
#         conv_features, "conv1", 8,
#         [5, 5], "MSFT", 0,
#         filter_stride=[2, 2], return_params=True
#     )
#     log(conv1)
#     conv2, cp2 = conv_layer(
#         conv1, "conv2", 8,
#         [5, 5], "MSFT", 0,
#         filter_stride=[2, 2], return_params=True
#     )
#     log(conv2)
#     conv3, cp3 = conv_layer(
#         conv2, "conv3", 8,
#         [5, 5], "MSFT", 0,
#         filter_stride=[2, 2], return_params=True
#     )
#     log(conv3)
#     conv3_flat, dim = reshape_conv_layer(conv3)
#     log(conv3_flat)
#     return conv3_flat


def sequence_length(data):
    """
    Takes in a tensor of data in form [batch_index, time_index, observation_size] or
    [batch_index, time_index, obsx, obsy]
    and returns a tensor indicating the length of each sequence ([batch_size])
    which is indicated by the last non-zero element
    """
    used = used_frames(data)
    # used is now [batch_size, max_length, 1] where the last dim is [0, 1] if used
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def used_frames(data):
    if len(data.get_shape().as_list()) == 3:
        r_inds = 2
    elif len(data.get_shape().as_list()) == 4:
        r_inds = [2, 3]
    elif len(data.get_shape().as_list()) == 5:
        r_inds = [2, 3, 4]
    else:
        assert False
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=r_inds))
    return used

def env_feature_predictor(features, actions, max_length=100):
    """
    Takes in features and a set of actions, sets features as inital state of RNN
    and returns a new tensor in feature space to be used for decoding of future state after
    actions are taken
    """
    # the rnn states should be the same size as the input features
    num_hidden = features.get_shape().as_list()[1]

    output, state = tf.nn.dynamic_rnn(
        tf.nn.rnn_cell.GRUCell(num_hidden),
        actions,
        dtype=tf.float32,
        sequence_length=length(actions),
        initial_state=features
    )
    return state[:, -1, :]

def action_compressor(actions, compressed_action_size=25, max_length=100):
    """
    Takes in a set of actions, runs them through a recurrent model and returns the model's output
    can be thought of as "compressing" a sequence of states into a single vector
    this will be concatenated with a states features and fed to the nth state decoder
    """
    # the rnn states should be the same size as the input features
    output, state = tf.nn.dynamic_rnn(
        tf.nn.rnn_cell.GRUCell(compressed_action_size),
        actions,
        dtype=tf.float32,
        sequence_length=length(actions)
    )
    return output[:, -1, :]



def DQL_net_conv(features, n_actions, is_training, reuse=False, return_features=False):
    test = not is_training
    # pooled_feats = tf.nn.avg_pool(
    #     features, ksize=[1, 4, 4, 1],
    #     strides=[1, 4, 4, 1], padding='SAME'
    # )
    conv1, cp1 = conv_layer(
        features, "conv1", 8,
        [5, 5], "MSFT", .0005 if not reuse else 0.0,
        filter_stride=[2, 2], return_params=True
    )
    print(conv1.get_shape().as_list(), "conv1 shape")
    conv2, cp2 = conv_layer(
        conv1, "conv2", 8,
        [5, 5], "MSFT", .0005 if not reuse else 0.0,
        filter_stride=[2, 2], return_params=True
    )
    print(conv2.get_shape().as_list(), "conv2 shape")
    conv3, cp3 = conv_layer(
        conv2, "conv3", 8,
        [5, 5], "MSFT", .0005 if not reuse else 0.0,
        filter_stride=[2, 2], return_params=True
    )
    print(conv3.get_shape().as_list(), "conv3 shape")
    conv3_flat, dim = reshape_conv_layer(conv3)
    fc4, fp4 = linear_layer(
        conv3_flat, "fc4", 64,
        .003, .0005 if not reuse else 0.0, return_params=True
    )
    print(fc4.get_shape().as_list(), "fc4 shape")
    out, op = linear_layer(
        fc4, "out", n_actions,
        .003, .0005 if not reuse else 0.0, return_params=True,
        nonlinearity=None
    )
    print(out.get_shape().as_list(), "out shape")
    if return_features:
        return out, fc4
    else:
        return out, [cp1, cp2, cp3, fp4, op]

def mu_network_conv(features, is_training, n_outputs, reuse=False):
    test = not is_training
    print(features.get_shape().as_list())
    conv1 = layer_normalized_conv_layer(
            features, "conv1", 32,
            [3, 3], "MSFT", 0.00 if not reuse else 0.0,
            filter_stride=[1, 1]
        )
    print(conv1.get_shape().as_list(), "CONV 1 SHAPE")
    conv2 = layer_normalized_conv_layer(
            conv1, "conv2", 32,
            [3, 3], "MSFT", 0.00 if not reuse else 0.0,
            filter_stride=[1, 1]
        )
    print(conv2.get_shape().as_list(), "CONV 2 SHAPE")
    conv3 = layer_normalized_conv_layer(
            conv2, "conv3", 32,
            [3, 3], "MSFT", 0.00 if not reuse else 0.0,
            filter_stride=[1, 1]
        )
    print(conv3.get_shape().as_list(), "CONV 3 SHAPE")
    conv3_flat, dim = reshape_conv_layer(conv3)

    fc4 = layer_normalized_linear_layer(
        conv3_flat, "fc4", 200,
        .003, .00 if not reuse else 0.0
    )
    print(fc4.get_shape().as_list(), "fc4 shape")
    fc5 = layer_normalized_linear_layer(
        fc4, "fc5", 200,
        .0003, .00 if not reuse else 0.0
    )
    print(fc5.get_shape().as_list(), "fc5 shape")
    outs = [
        linear_layer(
            fc5, "out_{}".format(i), n_out,
            .0003, .00 if not reuse else 0.0,
            nonlinearity=None
        ) for i, n_out in enumerate(n_outputs)
    ]
    print([out.get_shape().as_list() for out in outs], "out shape")
    return outs

def Q_network_conv(features, actions, is_training, reuse=False):
    test = not is_training
    conv1 = layer_normalized_conv_layer(
            features, "conv1", 32,
            [3, 3], "MSFT", 0.001 if not reuse else 0.0,
            filter_stride=[1, 1]
        )
    print(conv1.get_shape().as_list(), "CONV 1 SHAPE")
    conv2 = layer_normalized_conv_layer(
            conv1, "conv2", 32,
            [3, 3], "MSFT", 0.001 if not reuse else 0.0,
            filter_stride=[1, 1]
        )
    print(conv2.get_shape().as_list(), "CONV 2 SHAPE")
    conv3 = layer_normalized_conv_layer(
            conv2, "conv3", 32,
            [3, 3], "MSFT", 0.001 if not reuse else 0.0,
            filter_stride=[1, 1]
        )
    print(conv3.get_shape().as_list(), "CONV 3 SHAPE")
    conv3_flat, dim = reshape_conv_layer(conv3)

    fc4 = layer_normalized_linear_layer(
        conv3_flat, "fc4", 200,
        .003, .001 if not reuse else 0.0
    )
    print(fc4.get_shape().as_list(), "fc4 shape")
    fc4_with_actions = tf.concat(1, [fc4] + actions, "fc4_with_actions")
    print(fc4_with_actions.get_shape().as_list(), "fc4 with actions shape")
    fc5 = layer_normalized_linear_layer(
        fc4_with_actions, "fc5", 200,
        .003, .001 if not reuse else 0.0
    )
    print(fc5.get_shape().as_list(), "fc5 shape")
    out = linear_layer(
        fc5, "out", 1,
        .003, .001 if not reuse else 0.0,
        nonlinearity=None
    )
    print(out.get_shape().as_list(), "out shape")
    return out


def im_network(features, is_training, reuse=False):
    test = not is_training
    conv1 = layer_normalized_conv_layer(
            features, "conv1", 32,
            [3, 3], "MSFT", 0.001 if not reuse else 0.0,
            filter_stride=[1, 1]
        )
    print(conv1.get_shape().as_list(), "CONV 1 SHAPE")
    conv2 = layer_normalized_conv_layer(
            conv1, "conv2", 32,
            [3, 3], "MSFT", 0.001 if not reuse else 0.0,
            filter_stride=[1, 1]
        )
    print(conv2.get_shape().as_list(), "CONV 2 SHAPE")
    conv3 = layer_normalized_conv_layer(
            conv2, "conv3", 32,
            [3, 3], "MSFT", 0.001 if not reuse else 0.0,
            filter_stride=[1, 1]
        )
    print(conv3.get_shape().as_list(), "CONV 3 SHAPE")
    conv3_flat, dim = reshape_conv_layer(conv3)

    im_feats = layer_normalized_linear_layer(
        conv3_flat, "im_feats", 200,
        .003, .001 if not reuse else 0.0
    )
    print(im_feats.get_shape().as_list(), "im feats shape")
    return im_feats

def dec_network(features, actions, is_training, reuse=False):
    features_with_actions = tf.concat(1, [features] + actions, "features_with_actions")
    fc1 = layer_normalized_linear_layer(
        features_with_actions, "fc1", 64*64*32,
        .003, .001 if not reuse else 0.0
    )
    batch_size = tf.shape(fc1)[0]
    tf_out_shape = tf.pack([batch_size, 64, 64, 32])
    conv1 = tf.reshape(fc1, tf_out_shape, name="conv1")
    conv2 = deconv_layer(
        conv1, "conv2", (64, 64, 9), 32,
        [3, 3], [1, 1], .01, 0.001 if not reuse else 0.0,
        nonlinearity=tf.tanh
    )
    # print(conv2.get_shape().as_list(), "CONV 2 SHAPE")
    # conv3 = layer_normalized_deconv_layer(
    #         conv2, "conv3", (32, 32, 32), 32,
    #         [3, 3], [1, 1], .01, 0.001 if not reuse else 0.0
    #     )
    # print(conv3.get_shape().as_list(), "CONV 3 SHAPE")
    # conv4 = deconv_layer(
    #         conv3, "conv4", (32, 32, 3*3), 32,
    #         [3, 3], [1, 1], .01, 0.001 if not reuse else 0.0,
    #         nonlinearity=tf.sigmoid
    #     )
    # print(conv4.get_shape().as_list(), "CONV 4 SHAPE")
    return conv2

def canvas_im_network(features, is_training, reuse=False):
    im, canvas = tf.split(3, 2, features)
    print(im.get_shape().as_list(), "im size")
    print(canvas.get_shape().as_list(), "canvas size")
    with tf.variable_scope("imf", reuse=reuse):
        im_feats = im_network(im, is_training, reuse=reuse)
    with tf.variable_scope("imf", reuse=True):
        canvas_feats = im_network(canvas, is_training, reuse=True)
    out_feats = tf.concat(1, [im_feats, canvas_feats], name="concat_im_feats")
    print(out_feats.get_shape().as_list(), "out canvas im networks feats")
    return out_feats

def mu_network_im(features, is_training, n_outputs, reuse=False):
    test = not is_training
    n_features_fc2 = 200
    fc2 = layer_normalized_linear_layer(
        features, "fc2", n_features_fc2,
        .003, .001 if not reuse else 0.0
    )
    print(fc2.get_shape().as_list(), "fc2 shape")
    outs = [
        linear_layer(
            fc2, "out_{}".format(i), n_out,
            .003, .001 if not reuse else 0.0,
            nonlinearity=None
        ) for i, n_out in enumerate(n_outputs)
    ]
    print([out.get_shape().as_list() for out in outs], "out shape")
    return outs

def Q_network_im(features, actions, is_training, reuse=False):
    test = not is_training
    fc1_with_actions = tf.concat(1, [features] + actions, "fc1_with_actions")
    print(fc1_with_actions.get_shape().as_list(), "fc1 with actions shape")
    n_features_fc2 = 200
    fc2 = layer_normalized_linear_layer(
        fc1_with_actions, "fc2", n_features_fc2,
        .003, .001 if not reuse else 0.0
    )
    print(fc2.get_shape().as_list(), "fc2 shape")
    out = linear_layer(
        fc2, "out", 1,
        .003, .001 if not reuse else 0.0,
        nonlinearity=None
    )
    print(out.get_shape().as_list(), "out shape")
    return out

def mu_network_wn(features, is_training, n_outputs, reuse=False):
    test = not is_training
    n_features_fc1 = 400
    fc1 = weight_normalized_linear_layer(
        features, "fc1", n_features_fc1,
        .05, .00 if not reuse else 0.0
    )
    print(fc1.get_shape().as_list(), "fc1 shape")
    n_features_fc2 = 300
    fc2 = weight_normalized_linear_layer(
        fc1, "fc2", n_features_fc2,
        .05, .00 if not reuse else 0.0
    )
    print(fc2.get_shape().as_list(), "fc2 shape")
    outs = [
        linear_layer(
            fc2, "out_{}".format(i), n_out,
            .003, .001 if not reuse else 0.0,
            nonlinearity=None
        ) for i, n_out in enumerate(n_outputs)
    ]
    print([out.get_shape().as_list() for out in outs], "out shape")
    return outs

def Q_network_wn(features, actions, is_training, reuse=False):
    test = not is_training
    n_features_fc1 = 400
    fc1 = weight_normalized_linear_layer(
        features, "fc1", n_features_fc1,
        .05, .001 if not reuse else 0.0
    )
    print(fc1.get_shape().as_list(), "fc1 shape")
    fc1_with_actions = tf.concat(1, [fc1] + actions, "fc1_with_actions")
    print(fc1_with_actions.get_shape().as_list(), "fc1 with actions shape")
    n_features_fc2 = 300
    fc2 = weight_normalized_linear_layer(
        fc1_with_actions, "fc2", n_features_fc2,
        .05, .001 if not reuse else 0.0
    )
    print(fc2.get_shape().as_list(), "fc2 shape")
    out = linear_layer(
        fc2, "out", 1,
        .05, .001 if not reuse else 0.0,
        nonlinearity=None
    )
    print(out.get_shape().as_list(), "out shape")
    return out


def mu_network_ln(features, is_training, n_outputs, reuse=False):
    test = not is_training
    n_features_fc1 = 400
    fc1 = layer_normalized_linear_layer(
        features, "fc1", n_features_fc1,
        .003, .001 if not reuse else 0.0
    )
    print(fc1.get_shape().as_list(), "fc1 shape")
    n_features_fc2 = 300
    fc2 = layer_normalized_linear_layer(
        fc1, "fc2", n_features_fc2,
        .003, .001 if not reuse else 0.0
    )
    print(fc2.get_shape().as_list(), "fc2 shape")
    outs = [
        linear_layer(
            fc2, "out_{}".format(i), n_out,
            .003, .001 if not reuse else 0.0,
            nonlinearity=None
        ) for i, n_out in enumerate(n_outputs)
    ]
    print([out.get_shape().as_list() for out in outs], "out shape")
    return outs

def Q_network_ln(features, actions, is_training, reuse=False):
    test = not is_training
    n_features_fc1 = 400
    fc1 = layer_normalized_linear_layer(
        features, "fc1", n_features_fc1,
        .003, .001 if not reuse else 0.0
    )
    print(fc1.get_shape().as_list(), "fc1 shape")
    fc1_with_actions = tf.concat(1, [fc1] + actions, "fc1_with_actions")
    print(fc1_with_actions.get_shape().as_list(), "fc1 with actions shape")
    n_features_fc2 = 300
    fc2 = layer_normalized_linear_layer(
        fc1_with_actions, "fc2", n_features_fc2,
        .003, .001 if not reuse else 0.0
    )
    print(fc2.get_shape().as_list(), "fc2 shape")
    out = linear_layer(
        fc2, "out", 1,
        .003, .001 if not reuse else 0.0,
        nonlinearity=None
    )
    print(out.get_shape().as_list(), "out shape")
    return out

def mu_network_bn(features, is_training, n_outputs, reuse=False):
    test = not is_training
    n_features_fc1 = 400
    fc1 = batch_normalized_linear_layer(
        features, "fc1", n_features_fc1,
        .003, .001 if not reuse else 0.0, test=test
    )
    print(fc1.get_shape().as_list(), "fc1 shape")
    n_features_fc2 = 300
    fc2 = batch_normalized_linear_layer(
        fc1, "fc2", n_features_fc2,
        .003, .001 if not reuse else 0.0, test=test
    )
    print(fc2.get_shape().as_list(), "fc2 shape")
    outs = [
        linear_layer(
            fc2, "out_{}".format(i), n_out,
            .003, .001 if not reuse else 0.0,
            nonlinearity=None
        ) for i, n_out in enumerate(n_outputs)
    ]
    print([out.get_shape().as_list() for out in outs], "out shape")
    return outs

def Q_network_bn(features, actions, is_training, reuse=False):
    test = not is_training
    n_features_fc1 = 400
    fc1 = batch_normalized_linear_layer(
        features, "fc1", n_features_fc1,
        .003, .001 if not reuse else 0.0, test=test
    )
    print(fc1.get_shape().as_list(), "fc1 shape")
    fc1_with_actions = tf.concat(1, [fc1] + actions, "fc1_with_actions")
    print(fc1_with_actions.get_shape().as_list(), "fc1 with actions shape")
    n_features_fc2 = 300
    fc2 = batch_normalized_linear_layer(
        fc1_with_actions, "fc2", n_features_fc2,
        .003, .001 if not reuse else 0.0, test=test
    )
    print(fc2.get_shape().as_list(), "fc2 shape")
    out = linear_layer(
        fc2, "out", 1,
        .003, .001 if not reuse else 0.0,
        nonlinearity=None
    )
    print(out.get_shape().as_list(), "out shape")
    return out

def cifar_image_encoder_network(images, is_training, scope_name="encoder", reuse=False):
    """
    Builds the network that encodes the images

    Args:
        images: Images returned from distorted_inputs() or inputs().
        is_training: True if training, false if eval
        reuse: if true, will use previously allocated variables
    Returns:
        image features
    """
    NUM_CHANNELS = 3
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 32
        conv1 = batch_normalized_conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = 64
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv2,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")

        # conv3
        n_filters_conv3 = 128
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv3,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        # conv4
        n_filters_conv4 = 256
        conv4 = batch_normalized_conv_layer(
            conv3, "conv4", n_filters_conv4,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")

        conv4_flat, dim = reshape_conv_layer(conv4)
        print(conv4_flat.get_shape().as_list(), "CONV 4 FLAT SHAPE")

        # 1 fully connected layer to maintain spatial info
        n_outputs = 128
        out = out = batch_normalized_linear_layer(
            conv4_flat, "output", n_outputs,
            .01, .004 if not reuse else 0.0, test=test, nonlinearity=None
        )
        print(out.get_shape().as_list(), "OUT SHAPE")

    return out


def cifar_discriminator_network(images, is_training, scope_name="discriminator", reuse=False, return_feats=False):
    """
    Builds the network that is trainged to discriminate between generated images
    and images from the dataset

    Args:
        images: Images returned from distorted_inputs() or inputs().
        is_training: True if training, false if eval
        reuse: if true, will use previously allocated variables
    Returns:
        image features
    """
    NUM_CHANNELS = 3
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 32
        conv1 = batch_normalized_conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = 64
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv2,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")

        # conv3
        n_filters_conv3 = 128
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv3,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        # conv4
        n_filters_conv4 = 256
        conv4 = batch_normalized_conv_layer(
            conv3, "conv4", n_filters_conv4,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")

        # conv5
        n_filters_conv5 = 512
        conv5 = batch_normalized_conv_layer(
            conv4, "conv5", n_filters_conv5,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        print(conv5.get_shape().as_list(), "CONV 5 SHAPE")

        # do global mean pooling
        top_feats = global_pooling_layer(conv5, scope.name+"_features")
        print(top_feats.get_shape().as_list(), "feats shape")

        out = batch_normalized_linear_layer(
            top_feats, "output", 1,
            .01, .004 if not reuse else 0.0, test=test, nonlinearity=None
        )
        print(out.get_shape().as_list(), "disc out shape")
    if return_feats:
        return out, top_feats
    return out

def cifar_discriminator_network_easy(images, is_training, scope_name="discriminator", reuse=False):
    """
    simplest discriminator one layer then global pooling and softmax
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 32
        conv1 = batch_normalized_conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")
        # do global mean pooling
        top_feats = global_pooling_layer(conv1, scope.name + "_features")
        print(top_feats.get_shape().as_list(), "feats shape")

        out = batch_normalized_linear_layer(
            top_feats, "output", 1,
            .01, .004 if not reuse else 0.0, test=test, nonlinearity=None
        )
        print(out.get_shape().as_list(), "disc out shape")
    return out

def cifar_discriminator_network_medium(images, is_training, scope_name="discriminator", reuse=False):
    """
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 32
        conv1 = batch_normalized_conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = 64
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv2,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")
        # do global mean pooling
        top_feats = global_pooling_layer(conv2, scope.name+"_features")
        print(top_feats.get_shape().as_list(), "feats shape")

        out = batch_normalized_linear_layer(
            top_feats, "output", 1,
            .01, .004 if not reuse else 0.0, test=test, nonlinearity=None
        )
        print(out.get_shape().as_list(), "disc out shape")
    return out

def cifar_discriminator_network_hard(images, is_training, scope_name="discriminator", reuse=False):
    """
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 32
        conv1 = batch_normalized_conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = 64
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv2,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")
        # conv3
        n_filters_conv3 = 128
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv3,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")
        # do global mean pooling
        top_feats = global_pooling_layer(conv3, scope.name+"_features")
        print(top_feats.get_shape().as_list(), "feats shape")

        out = batch_normalized_linear_layer(
            top_feats, "output", 1,
            .01, .004 if not reuse else 0.0, test=test, nonlinearity=None
        )
        print(out.get_shape().as_list(), "disc out shape")
    return out

def cifar_discriminator_network_harder(images, is_training, scope_name="discriminator", reuse=False):
    """
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 32
        conv1 = batch_normalized_conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = 64
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv2,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")
        # conv3
        n_filters_conv3 = 128
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv3,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")
        # conv4
        n_filters_conv4 = 256
        conv4 = batch_normalized_conv_layer(
            conv3, "conv4", n_filters_conv4,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        # do global mean pooling
        top_feats = global_pooling_layer(conv4, scope.name+"_features")
        print(top_feats.get_shape().as_list(), "feats shape")

        out = batch_normalized_linear_layer(
            top_feats, "output", 1,
            .01, .004 if not reuse else 0.0, test=test, nonlinearity=None
        )
        print(out.get_shape().as_list(), "disc out shape")
    return out

def cifar_discriminator_network_hardest(images, is_training, scope_name="discriminator", reuse=False):
    """
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 32
        conv1 = batch_normalized_conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = 64
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv2,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")
        # conv3
        n_filters_conv3 = 128
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv3,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")
        # conv4
        n_filters_conv4 = 256
        conv4 = batch_normalized_conv_layer(
            conv3, "conv4", n_filters_conv4,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")
        # conv5
        n_filters_conv5 = 512
        conv5 = batch_normalized_conv_layer(
            conv4, "conv5", n_filters_conv5,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        print(conv5.get_shape().as_list(), "CONV 5 SHAPE")
        # do global mean pooling
        top_feats = global_pooling_layer(conv5, scope.name+"_features")
        print(top_feats.get_shape().as_list(), "feats shape")

        out = batch_normalized_linear_layer(
            top_feats, "output", 1,
            .01, .004 if not reuse else 0.0, test=test, nonlinearity=None
        )
        print(out.get_shape().as_list(), "disc out shape")
    return out

def cifar_discriminator_network_multi_output(images, is_training, scope_name="discriminator", reuse=False):
    """
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 32
        conv1 = batch_normalized_conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        feats1 = global_pooling_layer(conv1, "feats1")
        out1 = batch_normalized_linear_layer(
            feats1, "out1", 1,
            .01, .004 if not reuse else 0.0,
            test=test, nonlinearity=None
        )

        # conv2
        n_filters_conv2 = 64
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv2,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")

        feats2 = global_pooling_layer(conv2, "feats2")
        out2 = batch_normalized_linear_layer(
            feats2, "out2", 1,
            .01, .004 if not reuse else 0.0,
            test=test, nonlinearity=None
        )

        # conv3
        n_filters_conv3 = 128
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv3,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        feats3 = global_pooling_layer(conv3, "feats3")
        out3 = batch_normalized_linear_layer(
            feats3, "out3", 1,
            .01, .004 if not reuse else 0.0,
            test=test, nonlinearity=None
        )

        # conv4
        n_filters_conv4 = 256
        conv4 = batch_normalized_conv_layer(
            conv3, "conv4", n_filters_conv4,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")

        feats4 = global_pooling_layer(conv4, "feats4")
        out4 = batch_normalized_linear_layer(
            feats4, "out4", 1,
            .01, .004 if not reuse else 0.0,
            test=test, nonlinearity=None
        )
        # conv5
        n_filters_conv5 = 512
        conv5 = batch_normalized_conv_layer(
            conv4, "conv5", n_filters_conv5,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        print(conv5.get_shape().as_list(), "CONV 5 SHAPE")
        # do global mean pooling
        top_feats = global_pooling_layer(conv5, scope.name+"_features")
        print(top_feats.get_shape().as_list(), "feats shape")

        out = batch_normalized_linear_layer(
            top_feats, "output", 1,
            .01, .004 if not reuse else 0.0, test=test, nonlinearity=None
        )
        print(out.get_shape().as_list(), "disc out shape")
    return out1, out2, out3, out4, out

def cifar_image_decoder_network(features, is_training, scope_name="decorder", reuse=False):
    """
    Simple convolutional decorder
    uses transpose convolutions (incorrectly called deconvolutions)
    """
    NUM_CHANNELS = 3
    batch_size, num_features = features.get_shape().as_list()
    print(features.get_shape().as_list())
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        # project input into enough dims to reshape into conv layer of
        # the below shape
        conv1_shape = (batch_size, 4, 4, 256)
        num_outputs_fc1 = conv1_shape[1] * conv1_shape[2] * conv1_shape[3]
        fc1 = batch_normalized_linear_layer(
            features, "fc1", num_outputs_fc1,
            .01, .004 if not reuse else 0.0, test=test
        )
        conv1 = tf.reshape(fc1, conv1_shape)
        print(conv1.get_shape().as_list(), "Conv1 shape")

        conv2_shape = (batch_size, 4, 4, 128)
        conv2 = batch_normalized_deconv_layer(
            conv1, "deconv2", conv2_shape[-1], conv2_shape,
            [3, 3], [1, 1], .01, .004 if not reuse else 0.0, test=test
        )
        print(conv2.get_shape().as_list(), "Conv2 shape")

        conv3_shape = (batch_size, 8, 8, 64)
        conv3 = batch_normalized_deconv_layer(
            conv2, "deconv3", conv3_shape[-1], conv3_shape,
            [5, 5], [2, 2], .01, .004 if not reuse else 0.0, test=test
        )
        print(conv3.get_shape().as_list(), "Conv3 shape")

        conv4_shape = (batch_size, 16, 16, 32)
        conv4 = batch_normalized_deconv_layer(
            conv3, "deconv4", conv4_shape[-1], conv4_shape,
            [5, 5], [2, 2], .01, .004 if not reuse else 0.0, test=test
        )
        print(conv4.get_shape().as_list(), "Conv4 shape")

        conv5_shape = (batch_size, 32, 32, NUM_CHANNELS)
        conv5 = deconv_layer(
            conv4, "deconv5", conv5_shape[-1], conv5_shape,
            [5, 5], [2, 2], .01, .004 if not reuse else 0.0,
            nonlinearity=tf.tanh
        )
        print(conv5.get_shape().as_list(), "Conv5 shape")

    return conv5d

def mnist_image_encoder_network(images, is_training, scope_name="encoder", reuse=False):
    """
    Builds the network that encodes the images

    Args:
        images: Images returned from distorted_inputs() or inputs().
        is_training: True if training, false if eval
        reuse: if true, will use previously allocated variables
    Returns:
        image features
    """
    NUM_CHANNELS = 1
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 2
        conv1 = batch_normalized_conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = 4
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv2,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")

        # conv3
        n_filters_conv3 = 8
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv3,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        # conv4
        n_filters_conv4 = 16
        conv4 = batch_normalized_conv_layer(
            conv3, "conv4", n_filters_conv4,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")


        conv4_flat, dim = reshape_conv_layer(conv4)
        print(conv4_flat.get_shape().as_list(), "CONV 4 FLAT SHAPE")

        # 1 fully connected layer to maintain spatial info
        n_outputs = 16
        out = out = batch_normalized_linear_layer(
            conv4_flat, "output", n_outputs,
            .01, .004 if not reuse else 0.0, test=test, nonlinearity=None
        )
        print(out.get_shape().as_list(), "OUT SHAPE")

    return out


def mnist_discriminator_network(images, is_training, scope_name="discriminator", reuse=False, return_feats=False):
    """
    Builds the network that is trainged to discriminate between generated images
    and images from the dataset

    Args:
        images: Images returned from distorted_inputs() or inputs().
        is_training: True if training, false if eval
        reuse: if true, will use previously allocated variables
    Returns:
        image features
    """
    NUM_CHANNELS = 1
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 2
        conv1 = batch_normalized_conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = 4
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv2,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")

        # conv3
        n_filters_conv3 = 8
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv3,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        # conv4
        n_filters_conv4 = 16
        conv4 = batch_normalized_conv_layer(
            conv3, "conv4", n_filters_conv4,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")

        # do global mean pooling
        top_feats = global_pooling_layer(conv4, scope.name+"_features")
        print(top_feats.get_shape().as_list(), "feats shape")

        out = batch_normalized_linear_layer(
            top_feats, "output", 1,
            .01, .004 if not reuse else 0.0, test=test, nonlinearity=None
        )
        print(out.get_shape().as_list(), "disc out shape")

    if return_feats:
        return out, top_feats
    return out

def mnist_image_decoder_network(features, is_training, scope_name="decorder", reuse=False):
    """
    Simple convolutional decorder
    uses transpose convolutions (incorrectly called deconvolutions)
    """
    NUM_CHANNELS = 1
    batch_size, num_features = features.get_shape().as_list()
    print(features.get_shape().as_list())
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        # project input into enough dims to reshape into conv layer of
        # the below shape
        conv1_shape = (batch_size, 4, 4, 16)
        num_outputs_fc1 = conv1_shape[1] * conv1_shape[2] * conv1_shape[3]
        fc1 = batch_normalized_linear_layer(
            features, "fc1", num_outputs_fc1,
            .01, .004 if not reuse else 0.0, test=test
        )
        print(fc1.get_shape().as_list(), "FC1 shape")
        conv1 = tf.reshape(fc1, conv1_shape)
        print(conv1.get_shape().as_list(), "Conv1 shape")

        conv2_shape = (batch_size, 4, 4, 8)
        conv2 = batch_normalized_deconv_layer(
            conv1, "deconv2", conv2_shape[-1], conv2_shape,
            [3, 3], [1, 1], .01, .004 if not reuse else 0.0, test=test
        )
        print(conv2.get_shape().as_list(), "Conv2 shape")

        conv3_shape = (batch_size, 7, 7, 4)
        conv3 = batch_normalized_deconv_layer(
            conv2, "deconv3", conv3_shape[-1], conv3_shape,
            [5, 5], [2, 2], .01, .004 if not reuse else 0.0, test=test
        )
        print(conv3.get_shape().as_list(), "Conv3 shape")

        conv4_shape = (batch_size, 14, 14, 2)
        conv4 = batch_normalized_deconv_layer(
            conv3, "deconv4", conv4_shape[-1], conv4_shape,
            [5, 5], [2, 2], .01, .004 if not reuse else 0.0, test=test
        )
        print(conv4.get_shape().as_list(), "Conv4 shape")

        conv5_shape = (batch_size, 28, 28, NUM_CHANNELS)
        conv5 = deconv_layer(
            conv4, "deconv5", conv5_shape[-1], conv5_shape,
            [5, 5], [2, 2], .01, .004 if not reuse else 0.0,
            nonlinearity=tf.tanh
        )
        print(conv5.get_shape().as_list(), "Conv5 shape")

    return conv5

def moving_mnist_image_encoder_network(images, is_training, scope_name="encoder", reuse=False, base_filters=4, n_outputs=64):
    """
    Builds the network that encodes the images

    Args:
        images: Images returned from distorted_inputs() or inputs().
        is_training: True if training, false if eval
        reuse: if true, will use previously allocated variables
    Returns:
        image features
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = base_filters
        conv1_1 = batch_normalized_conv_layer(
            images, "conv1_1", n_filters_conv1,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        conv1_2 = batch_normalized_conv_layer(
            conv1_1, "conv1_2", n_filters_conv1,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv1_2.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = base_filters * 2
        conv2_1 = batch_normalized_conv_layer(
            conv1_2, "conv2_1", n_filters_conv2,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        conv2_2 = batch_normalized_conv_layer(
            conv2_1, "conv2_2", n_filters_conv2,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv2_2.get_shape().as_list(), "CONV 2 SHAPE")

        # conv3
        n_filters_conv3 = base_filters * 4
        conv3 = batch_normalized_conv_layer(
            conv2_2, "conv3_1", n_filters_conv3,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        # conv4
        n_filters_conv4 = base_filters * 8
        conv4 = batch_normalized_conv_layer(
            conv3, "conv4", n_filters_conv4,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")

        conv4_flat, dim = reshape_conv_layer(conv4)
        print(conv4_flat.get_shape().as_list(), "CONV 4 FLAT SHAPE")

        # 1 fully connected layer to maintain spatial info
        out = linear_layer(
            conv4_flat, "output", n_outputs,
            .01, .004 if not reuse else 0.0, nonlinearity=None
        )
        print(out.get_shape().as_list(), "OUT SHAPE")

    return out

def moving_mnist_image_decoder_network(features, is_training, scope_name="decorder", reuse=False, base_filters=4):
    """
    Simple convolutional decorder
    uses transpose convolutions (incorrectly called deconvolutions)
    """
    batch_size, num_features = features.get_shape().as_list()
    print(features.get_shape().as_list())
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        # project input into enough dims to reshape into conv layer of
        # the below shape
        conv1_shape = (batch_size, 4, 4, base_filters * 8)
        num_outputs_fc1 = conv1_shape[1] * conv1_shape[2] * conv1_shape[3]
        fc1 = batch_normalized_linear_layer(
            features, "fc1", num_outputs_fc1,
            .01, .004 if not reuse else 0.0, test=test
        )
        conv1 = tf.reshape(fc1, conv1_shape)
        print(conv1.get_shape().as_list(), "Conv1 shape")

        conv2_shape = (batch_size, 8, 8, base_filters * 4)
        conv2 = batch_normalized_deconv_layer(
            conv1, "deconv2", conv2_shape,
            [3, 3], [2, 2], .01, .004 if not reuse else 0.0, test=test
        )
        print(conv2.get_shape().as_list(), "Conv2 shape")

        conv3_shape = (batch_size, 16, 16, base_filters * 2)
        conv3 = batch_normalized_deconv_layer(
            conv2, "deconv3", conv3_shape,
            [3, 3], [2, 2], .01, .004 if not reuse else 0.0, test=test
        )
        print(conv3.get_shape().as_list(), "Conv3 shape")

        conv4_1_shape = (batch_size, 32, 32, base_filters * 2)
        conv4_2_shape = (batch_size, 32, 32, base_filters)
        conv4_1 = batch_normalized_deconv_layer(
            conv3, "deconv4_1", conv4_1_shape,
            [3, 3], [2, 2], .01, .004 if not reuse else 0.0, test=test
        )
        conv4_2 = batch_normalized_deconv_layer(
            conv4_1, "deconv4_2", conv4_2_shape,
            [3, 3], [1, 1], .01, .004 if not reuse else 0.0, test=test
        )
        print(conv4_2.get_shape().as_list(), "Conv4 shape")

        conv5_1_shape = (batch_size, 64, 64, base_filters)
        conv5_2_shape = (batch_size, 64, 64, 1)
        conv5_1 = batch_normalized_deconv_layer(
            conv4_2, "deconv5_1", conv5_1_shape,
            [3, 3], [2, 2], .01, .004 if not reuse else 0.0, test=test
        )
        conv5_2 = deconv_layer(
            conv5_1, "deconv5_2", conv5_2_shape,
            [3, 3], [1, 1], .01, .004 if not reuse else 0.0,
            nonlinearity=tf.tanh
        )
        print(conv5_2.get_shape().as_list(), "Conv5 shape")

    return conv5_2

def moving_mnist_image_encoder_network_no_bn(images, is_training, scope_name="encoder", reuse=False, base_filters=16, n_outputs=64):
    """
    Builds the network that encodes the images

    Args:
        images: Images returned from distorted_inputs() or inputs().
        is_training: True if training, false if eval
        reuse: if true, will use previously allocated variables
    Returns:
        image features
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = base_filters
        conv1 = conv_layer(
            images, "conv1", n_filters_conv1,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = base_filters * 2
        conv2 = conv_layer(
            conv1, "conv2", n_filters_conv2,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")

        # conv3
        n_filters_conv3 = base_filters * 4
        conv3 = conv_layer(
            conv2, "conv3", n_filters_conv3,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        # conv4
        n_filters_conv4 = base_filters * 8
        conv4 = conv_layer(
            conv3, "conv4", n_filters_conv4,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")

        conv4_flat, dim = reshape_conv_layer(conv4)
        print(conv4_flat.get_shape().as_list(), "CONV 4 FLAT SHAPE")

        # 1 fully connected layer to maintain spatial info
        out = linear_layer(
            conv4_flat, "output", n_outputs,
            .01, .004 if not reuse else 0.0, nonlinearity=None
        )
        print(out.get_shape().as_list(), "OUT SHAPE")

    return out

def moving_mnist_image_encoder_network_no_bn_variational(images, is_training, scope_name="encoder", reuse=False, base_filters=4, n_outputs=64):
    """
    Builds the network that encodes the images

    Args:
        images: Images returned from distorted_inputs() or inputs().
        is_training: True if training, false if eval
        reuse: if true, will use previously allocated variables
    Returns:
        image features
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = base_filters
        conv1 = conv_layer(
            images, "conv1", n_filters_conv1,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = base_filters * 2
        conv2 = conv_layer(
            conv1, "conv2", n_filters_conv2,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")

        # conv3
        n_filters_conv3 = base_filters * 4
        conv3 = conv_layer(
            conv2, "conv3", n_filters_conv3,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        # conv4
        n_filters_conv4 = base_filters * 8
        conv4 = conv_layer(
            conv3, "conv4", n_filters_conv4,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")

        conv4_flat, dim = reshape_conv_layer(conv4)
        print(conv4_flat.get_shape().as_list(), "CONV 4 FLAT SHAPE")

        # 1 fully connected layer to maintain spatial info
        out_mu = linear_layer(
            conv4_flat, "output_mu", n_outputs,
            .01, .004 if not reuse else 0.0, nonlinearity=None
        )
        out_log_sigma_sq = linear_layer(
            conv4_flat, "output_log_sigma", n_outputs,
            .01, .004 if not reuse else 0.0, nonlinearity=None
        )
        print(out_mu.get_shape().as_list(), "OUT MU SHAPE")
        print(out_log_sigma_sq.get_shape().as_list(), "OUT LOG SIGMA SQ SHAPE")

    return out_mu, out_log_sigma_sq

def video_variational_encoder_fc(images, train=True, scope_name="encoder", reuse=False, n_outputs=64):
    with tf.variable_scope(scope_name) as scope:
        wd = 0.0 if reuse else .0005
        feats = tf.reshape(images, [-1] + [np.prod(images.get_shape().as_list()[1:])])
        fc1 = batch_normalized_linear_layer(feats, "fc1", 500, .01, wd, test=not train)
        fc2 = batch_normalized_linear_layer(fc1, "fc2", 500, .01, wd, test=not train)
        out_mu = linear_layer(
            fc2, "output_mu", n_outputs,
            .01, wd, nonlinearity=None
        )
        out_log_sigma_sq = linear_layer(
            fc2, "output_log_sigma", n_outputs,
            .01, wd, nonlinearity=None
        )
    return out_mu, out_log_sigma_sq#, [p1, pm, ps]

def video_variational_decoder_fc(features, train=True, scope_name="decoder", reuse=False):
    with tf.variable_scope(scope_name) as scope:
        wd = 0.0 if reuse else .0005
        fc1 = batch_normalized_linear_layer(features, "fc1", 500, .01, wd, test=not train)
        fc2 = batch_normalized_linear_layer(fc1, "fc2", 500, .01, wd, test=not train)
        out_v = linear_layer(fc2, "out_mu", 64*64, .01, wd, nonlinearity=None)
        out_mu_im = tf.reshape(out_v, [-1, 64, 64, 1])
    return out_mu_im#, [p1, po]


def video_variational_encoder(images, train=True, scope_name="encoder", reuse=False, n_outputs=64):
    """
    Builds the network that encodes the images

    Args:
        images: Images returned from distorted_inputs() or inputs().
        reuse: if true, will use previously allocated variables
    Returns:
        image features
    """
    test = not train
    wd = 0.0 if reuse else .0005
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 16
        conv1 = batch_normalized_conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], "MSFT", wd,
            filter_stride=[2, 2], test=test
        )

        # conv2
        n_filters_conv2 = 32
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv2,
            [5, 5], "MSFT", wd,
            filter_stride=[2, 2], test=test
        )

        # conv3
        n_filters_conv3 = 64
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv3,
            [3, 3], "MSFT", wd,
            filter_stride=[2, 2], test=test
        )

        # conv4
        n_filters_conv4 = 128
        conv4 = batch_normalized_conv_layer(
            conv3, "conv4", n_filters_conv4,
            [3, 3], "MSFT", wd,
            filter_stride=[2, 2], test=test
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")

        conv4_flat, dim = reshape_conv_layer(conv4)

        n_outputs_fc5 = 256
        fc5 = batch_normalized_linear_layer(conv4_flat, "fc5", n_outputs_fc5, .01, wd, test=test)

        # 1 fully connected layer to maintain spatial info
        out_mu = linear_layer(
            fc5, "output_mu", n_outputs,
            .01, wd, nonlinearity=None
        )
        out_log_sigma_sq = linear_layer(
            fc5, "output_log_sigma", n_outputs,
            .01, wd, nonlinearity=None
        )

    return out_mu, out_log_sigma_sq

def video_variational_decoder(features, train=True, scope_name="decoder", reuse=False):
    """
    Simple convolutional decorder
    uses transpose convolutions (incorrectly called deconvolutions)
    """
    test = not train
    wd = 0.0 if reuse else .0005
    batch_size, num_features = features.get_shape().as_list()
    print(features.get_shape().as_list())
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        fc1 = batch_normalized_linear_layer(features, "fc1", 256, .01, wd, test=test)
        print(fc1.get_shape().as_list(), "fc1 shape")

        conv2_shape = (batch_size, 4, 4, 128)
        num_outputs_fc2 = conv2_shape[1] * conv2_shape[2] * conv2_shape[3]
        fc2 = batch_normalized_linear_layer(fc1, "fc2", num_outputs_fc2, .01, wd, test=test)
        conv2 = tf.reshape(fc2, conv2_shape)

        conv3_shape = (batch_size, 8, 8, 64)
        conv3 = batch_normalized_deconv_layer(conv2, "deconv3", conv3_shape, [3, 3], [2, 2], .01, wd, test=test)

        conv4_shape = (batch_size, 16, 16, 32)
        conv4 = batch_normalized_deconv_layer(conv3, "deconv4", conv4_shape, [3, 3], [2, 2], .01, wd, test=test)

        conv5_shape = (batch_size, 32, 32, 16)
        conv5 = batch_normalized_deconv_layer(conv4, "deconv5", conv5_shape, [5, 5], [2, 2], .01, wd, test=test)

        conv6_shape = (batch_size, 64, 64, 1)
        conv6_mu = deconv_layer(
            conv5, "deconv6_mu", conv6_shape,
            [5, 5], [2, 2], .01, wd,
            nonlinearity=None
        )

    return conv6_mu

def video_variational_encoder_large(images, train=True, scope_name="encoder", reuse=False, n_outputs=64):
    """
    Builds the network that encodes the images

    Args:
        images: Images returned from distorted_inputs() or inputs().
        reuse: if true, will use previously allocated variables
    Returns:
        image features
    """
    test = not train
    wd = 0.0 if reuse else .0005
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 16
        conv1 = batch_normalized_conv_layer(
            images, "conv1", n_filters_conv1,
            [3, 3], "MSFT", wd,
            filter_stride=[1, 1], test=test
        )
        conv1_2 = batch_normalized_conv_layer(
            conv1, "conv1_2", n_filters_conv1,
            [3, 3], "MSFT", wd,
            filter_stride=[2, 2], test=test
        )

        # conv2
        n_filters_conv2 = 32
        conv2 = batch_normalized_conv_layer(
            conv1_2, "conv2", n_filters_conv2,
            [3, 3], "MSFT", wd,
            filter_stride=[1, 1], test=test
        )
        conv2_2 = batch_normalized_conv_layer(
            conv2, "conv2_2", n_filters_conv2,
            [3, 3], "MSFT", wd,
            filter_stride=[2, 2], test=test
        )

        # conv3
        n_filters_conv3 = 64
        conv3 = batch_normalized_conv_layer(
            conv2_2, "conv3", n_filters_conv3,
            [3, 3], "MSFT", wd,
            filter_stride=[1, 1], test=test
        )
        conv3_2 = batch_normalized_conv_layer(
            conv3, "conv3_2", n_filters_conv3,
            [3, 3], "MSFT", wd,
            filter_stride=[2, 2], test=test
        )

        # conv4
        n_filters_conv4 = 128
        conv4 = batch_normalized_conv_layer(
            conv3_2, "conv4", n_filters_conv4,
            [3, 3], "MSFT", wd,
            filter_stride=[1, 1], test=test
        )
        conv4_2 = batch_normalized_conv_layer(
            conv4, "conv4_2", n_filters_conv4,
            [3, 3], "MSFT", wd,
            filter_stride=[2, 2], test=test
        )

        conv4_flat, dim = reshape_conv_layer(conv4_2)
        log(conv4_flat)

        n_outputs_fc5 = 256
        fc5 = batch_normalized_linear_layer(conv4_flat, "fc5", n_outputs_fc5, .01, wd, test=test)

        # 1 fully connected layer to maintain spatial info
        out_mu = linear_layer(
            fc5, "output_mu", n_outputs,
            .01, wd, nonlinearity=None
        )
        out_log_sigma_sq = linear_layer(
            fc5, "output_log_sigma", n_outputs,
            .01, wd, nonlinearity=None
        )

    return out_mu, out_log_sigma_sq

def video_variational_decoder_large(features, train=True, scope_name="decoder", reuse=False):
    """
    Simple convolutional decorder
    uses transpose convolutions (incorrectly called deconvolutions)
    """
    test = not train
    wd = 0.0 if reuse else .0005
    batch_size, num_features = features.get_shape().as_list()
    print(features.get_shape().as_list())
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        fc1 = batch_normalized_linear_layer(features, "fc1", 256, .01, wd, test=test)
        print(fc1.get_shape().as_list(), "fc1 shape")

        conv2_shape = (batch_size, 4, 4, 128)
        num_outputs_fc2 = conv2_shape[1] * conv2_shape[2] * conv2_shape[3]
        fc2 = batch_normalized_linear_layer(fc1, "fc2", num_outputs_fc2, .01, wd, test=test)
        conv2 = tf.reshape(fc2, conv2_shape)

        conv3_shape = (batch_size, 8, 8, 128)
        conv3_2_shape = (batch_size, 8, 8, 64)
        conv3 = batch_normalized_deconv_layer(
            conv2, "deconv3", conv3_shape,
            [3, 3], [2, 2], .01, wd, test=test
        )
        conv3_2 = batch_normalized_deconv_layer(
            conv3, "deconv3_2", conv3_2_shape,
            [3, 3], [1, 1], .01, wd, test=test
        )

        conv4_shape = (batch_size, 16, 16, 64)
        conv4_2_shape = (batch_size, 16, 16, 32)
        conv4 = batch_normalized_deconv_layer(
            conv3_2, "deconv4", conv4_shape,
            [3, 3], [2, 2], .01, wd, test=test
        )
        conv4_2 = batch_normalized_deconv_layer(
            conv4, "deconv4_2", conv4_2_shape,
            [3, 3], [1, 1], .01, wd, test=test
        )

        conv5_shape = (batch_size, 32, 32, 32)
        conv5_2_shape = (batch_size, 32, 32, 16)
        conv5 = batch_normalized_deconv_layer(
            conv4_2, "deconv5", conv5_shape,
            [3, 3], [2, 2], .01, wd, test=test
        )
        conv5_2 = batch_normalized_deconv_layer(
            conv5, "deconv5_2", conv5_2_shape,
            [3, 3], [1, 1], .01, wd, test=test
        )

        conv6_shape = (batch_size, 64, 64, 16)
        conv6_mu_shape = (batch_size, 64, 64, 1)
        conv6 = batch_normalized_deconv_layer(
            conv5_2, "deconv6", conv6_shape,
            [3, 3], [2, 2], .01, wd, test=test
        )
        conv6_mu = deconv_layer(
            conv6, "deconv6_mu", conv6_mu_shape,
            [3, 3], [1, 1], .01, wd,
            nonlinearity=None
        )

    return conv6_mu

def moving_mnist_image_decoder_network_no_bn(features, is_training, scope_name="decorder", reuse=False, base_filters=16):
    """
    Simple convolutional decorder
    uses transpose convolutions (incorrectly called deconvolutions)
    """
    batch_size, num_features = features.get_shape().as_list()
    print(features.get_shape().as_list())
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        # project input into enough dims to reshape into conv layer of
        # the below shape
        conv1_shape = (batch_size, 4, 4, base_filters * 8)
        num_outputs_fc1 = conv1_shape[1] * conv1_shape[2] * conv1_shape[3]
        fc1 = linear_layer(
            features, "fc1", num_outputs_fc1,
            .01, .004 if not reuse else 0.0, nonlinearity=lrelu
        )
        conv1 = tf.reshape(fc1, conv1_shape)
        print(conv1.get_shape().as_list(), "Conv1 shape")

        conv2_shape = (batch_size, 8, 8, base_filters * 4)
        conv2 = deconv_layer(
            conv1, "deconv2", conv2_shape,
            [3, 3], [2, 2], .01, .004 if not reuse else 0.0, nonlinearity=lrelu
        )
        print(conv2.get_shape().as_list(), "Conv2 shape")

        conv3_shape = (batch_size, 16, 16, base_filters * 2)
        conv3 = deconv_layer(
            conv2, "deconv3", conv3_shape,
            [3, 3], [2, 2], .01, .004 if not reuse else 0.0, nonlinearity=lrelu
        )
        print(conv3.get_shape().as_list(), "Conv3 shape")

        conv4_shape = (batch_size, 32, 32, base_filters)
        conv4 = deconv_layer(
            conv3, "deconv4", conv4_shape,
            [3, 3], [2, 2], .01, .004 if not reuse else 0.0, nonlinearity=lrelu
        )
        print(conv4.get_shape().as_list(), "Conv4 shape")

        conv5_shape = (batch_size, 64, 64, 1)
        conv5 = deconv_layer(
            conv4, "deconv5", conv5_shape,
            [3, 3], [2, 2], .01, .004 if not reuse else 0.0,
            nonlinearity=tf.tanh
        )
        print(conv5.get_shape().as_list(), "Conv5 shape")

    return conv5

def discriminator_network(features, is_training, scope_name="discriminator", reuse=False):
    """
    Simple linear discriminator network
    2 layers of same size as features
    """
    batch_size, feature_size = features.get_shape().as_list()
    test = not is_training
    with tf.variable_scope(scope_name) as scope:
        layer_1 = batch_normalized_linear_layer(
            features, "layer_1", 1000,
            01, .004 if not reuse else 0.0, test=test
        )
        layer_2 = batch_normalized_linear_layer(
            layer_1, "layer_2", 1000,
            01, .004 if not reuse else 0.0, test=test
        )
        out = batch_normalized_linear_layer(
            layer_2, "output", 1,
            .01, .004 if not reuse else 0.0, test=test,
            nonlinearity=None
        )
    return out

def discriminator_network_no_input_norm(features, is_training, scope_name="discriminator", reuse=False):
    """
    Simple linear discriminator network
    2 layers of same size as features
    """
    batch_size, feature_size = features.get_shape().as_list()
    test = not is_training
    with tf.variable_scope(scope_name) as scope:
        layer_1 = linear_layer(
            features, "layer_1", feature_size,
            01, .004 if not reuse else 0.0
        )
        layer_2 = batch_normalized_linear_layer(
            layer_1, "layer_2", feature_size,
            01, .004 if not reuse else 0.0, test=test
        )
        out = batch_normalized_linear_layer(
            layer_2, "output", 1,
            .01, .004 if not reuse else 0.0, test=test,
            nonlinearity=None
        )
    return out

def moving_mnist_image_encoder_network_extra_fc(images, is_training, scope_name="encoder", reuse=False, base_filters=4, n_outputs=64):
    """
    Builds the network that encodes the images

    Args:
        images: Images returned from distorted_inputs() or inputs().
        is_training: True if training, false if eval
        reuse: if true, will use previously allocated variables
    Returns:
        image features
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = base_filters
        conv1_1 = batch_normalized_conv_layer(
            images, "conv1_1", n_filters_conv1,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        conv1_2 = batch_normalized_conv_layer(
            conv1_1, "conv1_2", n_filters_conv1,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv1_2.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = base_filters * 2
        conv2_1 = batch_normalized_conv_layer(
            conv1_2, "conv2_1", n_filters_conv2,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1]
        )
        conv2_2 = batch_normalized_conv_layer(
            conv2_1, "conv2_2", n_filters_conv2,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv2_2.get_shape().as_list(), "CONV 2 SHAPE")

        # conv3
        n_filters_conv3 = base_filters * 4
        conv3 = batch_normalized_conv_layer(
            conv2_2, "conv3_1", n_filters_conv3,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        # conv4
        n_filters_conv4 = base_filters * 8
        conv4 = batch_normalized_conv_layer(
            conv3, "conv4", n_filters_conv4,
            [3, 3], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2]
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")

        conv4_flat, dim = reshape_conv_layer(conv4)
        print(conv4_flat.get_shape().as_list(), "CONV 4 FLAT SHAPE")

        # 1 fully connected layer to maintain spatial info
        fc = out = batch_normalized_linear_layer(
            conv4_flat, "fc", 2 * n_outputs,
            .01, .004 if not reuse else 0.0, test=test, nonlinearity=None
        )
        out = out = batch_normalized_linear_layer(
            fc, "output", n_outputs,
            .01, .004 if not reuse else 0.0, test=test, nonlinearity=None
        )
        print(out.get_shape().as_list(), "OUT SHAPE")

    return out

def moving_mnist_image_decoder_network_extra_fc(features, is_training, scope_name="decorder", reuse=False, base_filters=4):
    """
    Simple convolutional decorder
    uses transpose convolutions (incorrectly called deconvolutions)
    """
    batch_size, num_features = features.get_shape().as_list()
    print(features.get_shape().as_list())
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        # project input into enough dims to reshape into conv layer of
        # the below shape
        conv1_shape = (batch_size, 4, 4, base_filters * 8)
        num_outputs_fc1 = conv1_shape[1] * conv1_shape[2] * conv1_shape[3]
        fc0 = batch_normalized_linear_layer(
            features, "fc0", 2 * num_features,
            .01, .004 if not reuse else 0.0, test=test
        )
        fc1 = batch_normalized_linear_layer(
            fc0, "fc1", num_outputs_fc1,
            .01, .004 if not reuse else 0.0, test=test
        )
        conv1 = tf.reshape(fc1, conv1_shape)
        print(conv1.get_shape().as_list(), "Conv1 shape")

        conv2_shape = (batch_size, 8, 8, base_filters * 4)
        conv2 = batch_normalized_deconv_layer(
            conv1, "deconv2", conv2_shape[-1], conv2_shape,
            [3, 3], [2, 2], .01, .004 if not reuse else 0.0, test=test
        )
        print(conv2.get_shape().as_list(), "Conv2 shape")

        conv3_shape = (batch_size, 16, 16, base_filters * 2)
        conv3 = batch_normalized_deconv_layer(
            conv2, "deconv3", conv3_shape[-1], conv3_shape,
            [3, 3], [2, 2], .01, .004 if not reuse else 0.0, test=test
        )
        print(conv3.get_shape().as_list(), "Conv3 shape")

        conv4_1_shape = (batch_size, 32, 32, base_filters * 2)
        conv4_2_shape = (batch_size, 32, 32, base_filters)
        conv4_1 = batch_normalized_deconv_layer(
            conv3, "deconv4_1", conv4_1_shape[-1], conv4_1_shape,
            [3, 3], [2, 2], .01, .004 if not reuse else 0.0, test=test
        )
        conv4_2 = batch_normalized_deconv_layer(
            conv4_1, "deconv4_2", conv4_2_shape[-1], conv4_2_shape,
            [3, 3], [1, 1], .01, .004 if not reuse else 0.0, test=test
        )
        print(conv4_2.get_shape().as_list(), "Conv4 shape")

        conv5_1_shape = (batch_size, 64, 64, base_filters)
        conv5_2_shape = (batch_size, 64, 64, 1)
        conv5_1 = batch_normalized_deconv_layer(
            conv4_2, "deconv5_1", conv5_1_shape[-1], conv5_1_shape,
            [3, 3], [2, 2], .01, .004 if not reuse else 0.0, test=test
        )
        conv5_2 = deconv_layer(
            conv5_1, "deconv5_2", conv5_2_shape[-1], conv5_2_shape,
            [3, 3], [1, 1], .01, .004 if not reuse else 0.0,
            nonlinearity=tf.tanh
        )
        print(conv5_2.get_shape().as_list(), "Conv5 shape")

    return conv5_2

def lsun_generator_network(features, is_training, scope_name="generator", reuse=False):
    """
    Simple convolutional decorder
    uses transpose convolutions (incorrectly called deconvolutions)
    """
    batch_size, num_features = features.get_shape().as_list()
    print(batch_size, num_features)
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        # project input into enough dims to reshape into conv layer of
        # the below shape
        conv1_shape = (batch_size, 4, 4, 1024)
        num_outputs_fc1 = conv1_shape[1] * conv1_shape[2] * conv1_shape[3]
        fc1 = batch_normalized_linear_layer(
            features, "fc1", num_outputs_fc1,
            .02, .004 if not reuse else 0.0, test=test
        )
        conv1 = tf.reshape(fc1, conv1_shape)
        print(conv1.get_shape().as_list(), "Conv1 shape")

        conv2_shape = (batch_size, 8, 8, 512)
        conv2 = batch_normalized_deconv_layer(
            conv1, "deconv2", conv2_shape,
            [5, 5], [2, 2], .02, .004 if not reuse else 0.0, test=test
        )
        print(conv2.get_shape().as_list(), "Conv2 shape")

        conv3_shape = (batch_size, 16, 16, 256)
        conv3 = batch_normalized_deconv_layer(
            conv2, "deconv3", conv3_shape,
            [5, 5], [2, 2], .02, .004 if not reuse else 0.0, test=test
        )
        print(conv3.get_shape().as_list(), "Conv3 shape")

        conv4_shape = (batch_size, 32, 32, 128)
        conv4 = batch_normalized_deconv_layer(
            conv3, "deconv4", conv4_shape,
            [5, 5], [2, 2], .02, .004 if not reuse else 0.0, test=test
        )
        print(conv4.get_shape().as_list(), "Conv4 shape")

        conv5_shape = (batch_size, 64, 64, 3)
        conv5 = deconv_layer(
            conv4, "deconv5", conv5_shape,
            [5, 5], [2, 2], .02, .004 if not reuse else 0.0,
            nonlinearity=tf.nn.tanh, tied_bias=True
        )
        print(conv5.get_shape().as_list(), "Conv5 shape")

    return conv5

def lsun_generator_network_128(features, is_training, scope_name="generator", reuse=False):
    """
    Simple convolutional decorder
    uses transpose convolutions (incorrectly called deconvolutions)
    """
    batch_size, num_features = features.get_shape().as_list()
    print(batch_size, num_features)
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        # project input into enough dims to reshape into conv layer of
        # the below shape
        conv1_shape = (batch_size, 4, 4, 1024)
        num_outputs_fc1 = conv1_shape[1] * conv1_shape[2] * conv1_shape[3]
        fc1 = batch_normalized_linear_layer(
            features, "fc1", num_outputs_fc1,
            .02, .004 if not reuse else 0.0, test=test
        )
        conv1 = tf.reshape(fc1, conv1_shape)
        print(conv1.get_shape().as_list(), "Conv1 shape")

        conv2_shape = (batch_size, 8, 8, 512)
        conv2 = batch_normalized_deconv_layer(
            conv1, "deconv2", conv2_shape,
            [5, 5], [2, 2], .02, .004 if not reuse else 0.0, test=test
        )
        print(conv2.get_shape().as_list(), "Conv2 shape")

        conv3_shape = (batch_size, 16, 16, 256)
        conv3 = batch_normalized_deconv_layer(
            conv2, "deconv3", conv3_shape,
            [5, 5], [2, 2], .02, .004 if not reuse else 0.0, test=test
        )
        print(conv3.get_shape().as_list(), "Conv3 shape")

        conv4_shape = (batch_size, 32, 32, 128)
        conv4 = batch_normalized_deconv_layer(
            conv3, "deconv4", conv4_shape,
            [5, 5], [2, 2], .02, .004 if not reuse else 0.0, test=test
        )
        print(conv4.get_shape().as_list(), "Conv4 shape")

        conv5_shape = (batch_size, 64, 64, 64)
        conv5 = batch_normalized_deconv_layer(
            conv4, "deconv5", conv5_shape,
            [5, 5], [2, 2], .02, .004 if not reuse else 0.0, test=test
        )
        print(conv5.get_shape().as_list(), "Conv5 shape")

        conv6_shape = (batch_size, 128, 128, 3)
        conv6 = deconv_layer(
            conv5, "deconv6", conv6_shape,
            [5, 5], [2, 2], .02, .004 if not reuse else 0.0,
            nonlinearity=tf.nn.tanh
        )
        print(conv6.get_shape().as_list(), "Conv6 shape")

    return conv6

def lsun_discriminator_network(images, is_training, scope_name="discriminator", reuse=False):
    """
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 128
        conv1 = conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], .02, 0.004 if not reuse else 0.0,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = 256
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv2,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")

        # conv3
        n_filters_conv3 = 512
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv3,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        # conv4
        n_filters_conv4 = 1024
        conv4 = batch_normalized_conv_layer(
            conv3, "conv4", n_filters_conv4,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")

        conv4_flat, dim = reshape_conv_layer(conv4)
        print(conv4_flat.get_shape().as_list(), "CONV 4 FLAT SHAPE")

        # 1 fully connected layer to maintain spatial info
        n_outputs = 1
        out = linear_layer(
            conv4_flat, "output", n_outputs,
            .02, .004 if not reuse else 0.0, nonlinearity=None
        )
        print(out.get_shape().as_list(), "disc out shape")
    return out

def lsun_discriminator_network_128(images, is_training, scope_name="discriminator", reuse=False):
    """
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 128
        conv1 = conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], .02, 0.004 if not reuse else 0.0,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = 256
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv2,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")

        # conv3
        n_filters_conv3 = 512
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv3,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        # conv4
        n_filters_conv4 = 1024
        conv4 = batch_normalized_conv_layer(
            conv3, "conv4", n_filters_conv4,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")

        # conv4
        n_filters_conv5 = 1024
        conv5 = batch_normalized_conv_layer(
            conv4, "conv5", n_filters_conv5,
            [3, 3], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv5.get_shape().as_list(), "CONV 5 SHAPE")

        conv5_flat, dim = reshape_conv_layer(conv5)
        print(conv5_flat.get_shape().as_list(), "CONV 5 FLAT SHAPE")

        # 1 fully connected layer to maintain spatial info
        n_outputs = 1
        out = linear_layer(
            conv5_flat, "output", n_outputs,
            .02, .004 if not reuse else 0.0, nonlinearity=None
        )
        print(out.get_shape().as_list(), "disc out shape")
    return out

def lsun_discriminator_network_multiout(images, is_training, scope_name="discriminator", reuse=False):
    """
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 128
        conv1 = conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], .02, 0.004 if not reuse else 0.0,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        feats1 = global_pooling_layer(conv1, "feats1")
        out1 = linear_layer(
            feats1, "out1", 1,
            .02, .004 if not reuse else 0.0,
            nonlinearity=None
        )

        # conv2
        n_filters_conv2 = 256
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv2,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")

        feats2 = global_pooling_layer(conv2, "feats2")
        out2 = linear_layer(
            feats2, "out2", 1,
            .02, .004 if not reuse else 0.0,
            nonlinearity=None
        )

        # conv3
        n_filters_conv3 = 512
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv3,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        feats3 = global_pooling_layer(conv3, "feats3")
        out3 = linear_layer(
            feats3, "out3", 1,
            .02, .004 if not reuse else 0.0,
            nonlinearity=None
        )

        # conv4
        n_filters_conv4 = 1024
        conv4 = batch_normalized_conv_layer(
            conv3, "conv4", n_filters_conv4,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")

        feats4 = global_pooling_layer(conv4, "feats4")
        out4 = linear_layer(
            feats4, "out4", 1,
            .02, .004 if not reuse else 0.0,
            nonlinearity=None
        )

        conv4_flat, dim = reshape_conv_layer(conv4)
        print(conv4_flat.get_shape().as_list(), "CONV 4 FLAT SHAPE")

        # 1 fully connected layer to maintain spatial info
        n_outputs = 1
        out = linear_layer(
            conv4_flat, "output", n_outputs,
            .02, .004 if not reuse else 0.0, nonlinearity=None
        )

        print(out.get_shape().as_list(), "disc out shape")
    return out1, out2, out3, out4, out

def lsun_discriminator_network_no_batchnorm(images, is_training, scope_name="discriminator", reuse=False):
    """
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 128
        conv1 = conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], .02, 0.004 if not reuse else 0.0,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        feats1 = global_pooling_layer(conv1, "feats1")
        out1 = linear_layer(
            feats1, "out1", 1,
            .02, .004 if not reuse else 0.0,
            nonlinearity=None
        )

        # conv2
        n_filters_conv2 = 256
        conv2 = conv_layer(
            conv1, "conv2", n_filters_conv2,
            [5, 5], .02, 0.004 if not reuse else 0.0,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")

        feats2 = global_pooling_layer(conv2, "feats2")
        out2 = linear_layer(
            feats2, "out2", 1,
            .02, .004 if not reuse else 0.0,
            nonlinearity=None
        )

        # conv3
        n_filters_conv3 = 512
        conv3 = conv_layer(
            conv2, "conv3", n_filters_conv3,
            [5, 5], .02, 0.004 if not reuse else 0.0,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        feats3 = global_pooling_layer(conv3, "feats3")
        out3 = linear_layer(
            feats3, "out3", 1,
            .02, .004 if not reuse else 0.0,
            nonlinearity=None
        )

        # conv4
        n_filters_conv4 = 1024
        conv4 = conv_layer(
            conv3, "conv4", n_filters_conv4,
            [5, 5], .02, 0.004 if not reuse else 0.0,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")

        feats4 = global_pooling_layer(conv4, "feats4")
        out4 = linear_layer(
            feats4, "out4", 1,
            .02, .004 if not reuse else 0.0,
            nonlinearity=None
        )

        conv4_flat, dim = reshape_conv_layer(conv4)
        print(conv4_flat.get_shape().as_list(), "CONV 4 FLAT SHAPE")

        # 1 fully connected layer to maintain spatial info
        n_outputs = 1
        out = linear_layer(
            conv4_flat, "output", n_outputs,
            .02, .004 if not reuse else 0.0, nonlinearity=None
        )

        print(out.get_shape().as_list(), "disc out shape")
    return out1, out2, out3, out4, out

def lsun_discriminator_network_dropout(images, is_training, scope_name="discriminator", reuse=False):
    """
    each output is actually K outputs that are averaged togther to produce the actual output. this is trained with droupout -- the idea is that with this there will be no single parameter that can be pushed by the training signal and instead it will be distributed across all of the biases
    """
    DROPOUT_OUTPUTS = 64
    KEEP_PROB = .25
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 128
        conv1 = batch_normalized_conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        feats1 = global_pooling_layer(conv1, "feats1")
        out1_m = batch_normalized_linear_layer(
            feats1, "out1", DROPOUT_OUTPUTS,
            .01, .004 if not reuse else 0.0,
            test=test, nonlinearity=None
        )
        out1_d = tf.nn.dropout(out1_m, KEEP_PROB) if is_training else out1_m
        out1 = tf.reduce_mean(out1_d, 1, keep_dims=True)

        # conv2
        n_filters_conv2 = 256
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv2,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")

        feats2 = global_pooling_layer(conv2, "feats2")
        out2_m = batch_normalized_linear_layer(
            feats2, "out2", DROPOUT_OUTPUTS,
            .01, .004 if not reuse else 0.0,
            test=test, nonlinearity=None
        )
        out2_d = tf.nn.dropout(out2_m, KEEP_PROB) if is_training else out2_m
        out2 = tf.reduce_mean(out2_d, 1, keep_dims=True)

        # conv3
        n_filters_conv3 = 512
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv3,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        feats3 = global_pooling_layer(conv3, "feats3")
        out3_m = batch_normalized_linear_layer(
            feats3, "out3", DROPOUT_OUTPUTS,
            .01, .004 if not reuse else 0.0,
            test=test, nonlinearity=None
        )
        out3_d = tf.nn.dropout(out3_m, KEEP_PROB) if is_training else out3_m
        out3 = tf.reduce_mean(out3_d, 1, keep_dims=True)

        # conv4
        n_filters_conv4 = 1024
        conv4 = batch_normalized_conv_layer(
            conv3, "conv4", n_filters_conv4,
            [5, 5], "MSFT", 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv4.get_shape().as_list(), "CONV 4 SHAPE")

        feats4 = global_pooling_layer(conv4, "feats4")
        out4_m = batch_normalized_linear_layer(
            feats4, "out4", DROPOUT_OUTPUTS,
            .01, .004 if not reuse else 0.0,
            test=test, nonlinearity=None
        )
        out4_d = tf.nn.dropout(out4_m, KEEP_PROB) if is_training else out4_m
        out4 = tf.reduce_mean(out4_d, 1, keep_dims=True)

        conv4_flat, dim = reshape_conv_layer(conv4)
        print(conv4_flat.get_shape().as_list(), "CONV 4 FLAT SHAPE")

        # 1 fully connected layer to maintain spatial info
        out_m = batch_normalized_linear_layer(
            conv4_flat, "output", DROPOUT_OUTPUTS,
            .01, .004 if not reuse else 0.0, test=test, nonlinearity=None
        )
        out_d = tf.nn.dropout(out_m, KEEP_PROB) if is_training else out_m
        out = tf.reduce_mean(out_d, 1, keep_dims=True)

        print(out.get_shape().as_list(), "disc out shape")
    return out1, out2, out3, out4, out

def imagenet_generator_network(features, is_training, scope_name="generator", reuse=False):
    """
    Simple convolutional decorder
    uses transpose convolutions (incorrectly called deconvolutions)
    """
    batch_size, num_features = features.get_shape().as_list()
    print(batch_size, num_features)
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        # project input into enough dims to reshape into conv layer of
        # the below shape
        conv1_shape = (batch_size, 4, 4, 1024)
        num_outputs_fc1 = conv1_shape[1] * conv1_shape[2] * conv1_shape[3]
        fc1 = batch_normalized_linear_layer(
            features, "fc1", num_outputs_fc1,
            .02, .004 if not reuse else 0.0, test=test
        )
        conv1 = tf.reshape(fc1, conv1_shape)
        print(conv1.get_shape().as_list(), "Conv1 shape")

        conv2_shape = (batch_size, 8, 8, 512)
        conv2 = batch_normalized_deconv_layer(
            conv1, "deconv2", conv2_shape,
            [5, 5], [2, 2], .02, .004 if not reuse else 0.0, test=test
        )
        print(conv2.get_shape().as_list(), "Conv2 shape")

        conv3_shape = (batch_size, 16, 16, 256)
        conv3 = batch_normalized_deconv_layer(
            conv2, "deconv3", conv3_shape,
            [5, 5], [2, 2], .02, .004 if not reuse else 0.0, test=test
        )
        print(conv3.get_shape().as_list(), "Conv3 shape")

        conv4_shape = (batch_size, 32, 32, 3)
        conv4 = deconv_layer(
            conv3, "deconv4", conv4_shape,
            [5, 5], [2, 2], .02, .004 if not reuse else 0.0, nonlinearity=tf.nn.tanh
        )
        print(conv4.get_shape().as_list(), "Conv4 shape")
    return conv4

def imagenet_discriminator_network(images, is_training, scope_name="discriminator", reuse=False, return_activations=False):
    """
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 128
        conv1 = conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], .02, 0.004 if not reuse else 0.0,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = 256
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv2,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")

        # conv3
        n_filters_conv3 = 512
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv3,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        conv3_flat, dim = reshape_conv_layer(conv3)
        print(conv3_flat.get_shape().as_list(), "CONV 3 FLAT SHAPE")

        # 1 fully connected layer to maintain spatial info
        n_outputs = 1
        out = linear_layer(
            conv3_flat, "output", n_outputs,
            .02, .004 if not reuse else 0.0, nonlinearity=None
        )
        print(out.get_shape().as_list(), "disc out shape")
    if not return_activations:
        return out
    else:
        return conv1, conv2, conv3, out

def imagenet_discriminator_network_multiout(images, is_training, scope_name="discriminator", reuse=False):
    """
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 128
        conv1 = conv_layer(
            images, "conv1", n_filters_conv1,
            [5, 5], .02, 0.004 if not reuse else 0.0,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv1.get_shape().as_list(), "CONV 1 SHAPE")

        feats1 = global_pooling_layer(conv1, "feats1")
        out1 = linear_layer(
            feats1, "out1", 1,
            .02, .004 if not reuse else 0.0,
            nonlinearity=None
        )

        # conv2
        n_filters_conv2 = 256
        conv2 = batch_normalized_conv_layer(
            conv1, "conv2", n_filters_conv2,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv2.get_shape().as_list(), "CONV 2 SHAPE")

        feats2 = global_pooling_layer(conv2, "feats2")
        out2 = linear_layer(
            feats2, "out2", 1,
            .02, .004 if not reuse else 0.0,
            nonlinearity=None
        )

        # conv3
        n_filters_conv3 = 512
        conv3 = batch_normalized_conv_layer(
            conv2, "conv3", n_filters_conv3,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv3.get_shape().as_list(), "CONV 3 SHAPE")

        feats3 = global_pooling_layer(conv3, "feats3")
        out3 = linear_layer(
            feats3, "out3", 1,
            .02, .004 if not reuse else 0.0,
            nonlinearity=None
        )

        conv3_flat, dim = reshape_conv_layer(conv3)
        print(conv3_flat.get_shape().as_list(), "CONV 3 FLAT SHAPE")

        # 1 fully connected layer to maintain spatial info
        n_outputs = 1
        out = linear_layer(
            conv3_flat, "output", n_outputs,
            .02, .004 if not reuse else 0.0, nonlinearity=None
        )

        print(out.get_shape().as_list(), "disc out shape")
    return out1, out2, out3, out

def imagenet_generator_network_large(features, is_training, scope_name="generator", reuse=False):
    """
    Simple convolutional decorder
    uses transpose convolutions (incorrectly called deconvolutions)
    same as used in dcgan paper
    """
    batch_size, num_features = features.get_shape().as_list()
    print(batch_size, num_features)
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        # project input into enough dims to reshape into conv layer of
        # the below shape
        conv1_shape = (batch_size, 4, 4, 1024)
        num_outputs_fc1 = conv1_shape[1] * conv1_shape[2] * conv1_shape[3]
        fc1 = batch_normalized_linear_layer(
            features, "fc1", num_outputs_fc1,
            .02, .004 if not reuse else 0.0, test=test
        )
        conv1 = tf.reshape(fc1, conv1_shape)
        print(conv1.get_shape().as_list(), "Conv1 shape")

        conv2_shape = (batch_size, 8, 8, 512)
        conv2a = batch_normalized_deconv_layer(
            conv1, "deconv2a", conv2_shape,
            [5, 5], [2, 2], .02, .004 if not reuse else 0.0, test=test
        )
        print(conv2a.get_shape().as_list(), "Conv2 shape")
        conv2b = batch_normalized_deconv_layer(
            conv2a, "deconv2b", conv2_shape,
            [5, 5], [1, 1], .02, .004 if not reuse else 0.0, test=test
        )
        print(conv2b.get_shape().as_list(), "Conv2 shape")

        conv3_shape = (batch_size, 16, 16, 256)
        conv3a = batch_normalized_deconv_layer(
            conv2b, "deconv3a", conv3_shape,
            [5, 5], [2, 2], .02, .004 if not reuse else 0.0, test=test
        )
        print(conv3a.get_shape().as_list(), "Conv3 shape")
        conv3b = batch_normalized_deconv_layer(
            conv3a, "deconv3b", conv3_shape,
            [5, 5], [1, 1], .02, .004 if not reuse else 0.0, test=test
        )
        print(conv3b.get_shape().as_list(), "Conv3 shape")

        conv4_shape = (batch_size, 32, 32, 3)
        conv4a = batch_normalized_deconv_layer(
            conv3b, "deconv4a", conv4_shape,
            [5, 5], [2, 2], .02, .004 if not reuse else 0.0, test=test
        )
        print(conv4a.get_shape().as_list(), "Conv4 shape")
        conv4b = deconv_layer(
            conv4a, "deconv4b", conv4_shape,
            [5, 5], [1, 1], .02, .004 if not reuse else 0.0, nonlinearity=tf.nn.tanh
        )
        print(conv4b.get_shape().as_list(), "Conv4 shape")
    return conv4b

def imagenet_discriminator_network_large(images, is_training, scope_name="discriminator", reuse=False, return_activations=False):
    """
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 128
        conv1a = conv_layer(
            images, "conv1a", n_filters_conv1,
            [5, 5], .02, 0.004 if not reuse else 0.0,
            filter_stride=[1, 1], nonlinearity=lrelu
        )
        print(conv1a.get_shape().as_list(), "CONV 1 SHAPE")
        # BATCH NORMALIZE THIS!!!!!
        conv1b = batch_normalized_conv_layer(
            conv1a, "conv1b", n_filters_conv1,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv1b.get_shape().as_list(), "CONV 1 SHAPE")

        # conv2
        n_filters_conv2 = 256
        conv2a = batch_normalized_conv_layer(
            conv1b, "conv2a", n_filters_conv2,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1], nonlinearity=lrelu
        )
        print(conv2a.get_shape().as_list(), "CONV 2 SHAPE")
        conv2b = batch_normalized_conv_layer(
            conv2a, "conv2b", n_filters_conv2,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv2b.get_shape().as_list(), "CONV 2 SHAPE")

        # conv3
        n_filters_conv3 = 512
        conv3a = batch_normalized_conv_layer(
            conv2b, "conv3a", n_filters_conv3,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1], nonlinearity=lrelu
        )
        print(conv3a.get_shape().as_list(), "CONV 3 SHAPE")
        conv3b = batch_normalized_conv_layer(
            conv3a, "conv3b", n_filters_conv3,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv3b.get_shape().as_list(), "CONV 3 SHAPE")

        conv3_flat, dim = reshape_conv_layer(conv3b)
        print(conv3_flat.get_shape().as_list(), "CONV 3 FLAT SHAPE")

        # 1 fully connected layer to maintain spatial info
        n_outputs = 1
        out = linear_layer(
            conv3_flat, "output", n_outputs,
            .02, .004 if not reuse else 0.0, nonlinearity=None
        )
        print(out.get_shape().as_list(), "disc out shape")
    if not return_activations:
        return out
    else:
        return conv1a, conv1b, conv2a, conv2b, conv3a, conv3b, out

def imagenet_discriminator_network_large_multiout(images, is_training, scope_name="discriminator", reuse=False, return_activations=False):
    """
    """
    test = not is_training
    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        print(images.get_shape().as_list(), "images SHAPE")
        # conv1
        n_filters_conv1 = 128
        conv1a = conv_layer(
            images, "conv1a", n_filters_conv1,
            [5, 5], .02, 0.004 if not reuse else 0.0,
            filter_stride=[1, 1], nonlinearity=lrelu
        )
        print(conv1a.get_shape().as_list(), "CONV 1 SHAPE")
        # BATCH NORMALIZE THIS!!!!!
        conv1b = batch_normalized_conv_layer(
            conv1a, "conv1b", n_filters_conv1,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv1b.get_shape().as_list(), "CONV 1 SHAPE")

        feats1 = global_pooling_layer(conv1b, "feats1")
        out1 = linear_layer(
            feats1, "out1", 1,
            .02, .004 if not reuse else 0.0,
            nonlinearity=None
        )

        # conv2
        n_filters_conv2 = 256
        conv2a = batch_normalized_conv_layer(
            conv1b, "conv2a", n_filters_conv2,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1], nonlinearity=lrelu
        )
        print(conv2a.get_shape().as_list(), "CONV 2 SHAPE")
        conv2b = batch_normalized_conv_layer(
            conv2a, "conv2b", n_filters_conv2,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv2b.get_shape().as_list(), "CONV 2 SHAPE")

        feats2 = global_pooling_layer(conv2b, "feats2")
        out2 = linear_layer(
            feats2, "out2", 1,
            .02, .004 if not reuse else 0.0,
            nonlinearity=None
        )

        # conv3
        n_filters_conv3 = 512
        conv3a = batch_normalized_conv_layer(
            conv2b, "conv3a", n_filters_conv3,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[1, 1], nonlinearity=lrelu
        )
        print(conv3a.get_shape().as_list(), "CONV 3 SHAPE")
        conv3b = batch_normalized_conv_layer(
            conv3a, "conv3b", n_filters_conv3,
            [5, 5], .02, 0.004 if not reuse else 0.0, test=test,
            filter_stride=[2, 2], nonlinearity=lrelu
        )
        print(conv3b.get_shape().as_list(), "CONV 3 SHAPE")

        feats3 = global_pooling_layer(conv3b, "feats3")
        out3 = linear_layer(
            feats3, "out3", 1,
            .02, .004 if not reuse else 0.0,
            nonlinearity=None
        )

        conv3_flat, dim = reshape_conv_layer(conv3b)
        print(conv3_flat.get_shape().as_list(), "CONV 3 FLAT SHAPE")

        # 1 fully connected layer to maintain spatial info
        n_outputs = 1
        out = linear_layer(
            conv3_flat, "output", n_outputs,
            .02, .004 if not reuse else 0.0, nonlinearity=None
        )
        print(out.get_shape().as_list(), "disc out shape")
    if not return_activations:
        return (out1, out2, out3, out)
    else:
        return (conv1a, conv1b, conv2a, conv2b, conv3a, conv3b), (out1, out2, out3, out)
