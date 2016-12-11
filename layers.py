"""
houses neural network layers
"""
import tensorflow as tf
from utils import _variable_with_weight_decay, _create_variable, _activation_summary
from tensorflow.python.ops import control_flow_ops

def log(var):
    print(var.name, var.get_shape().as_list())
def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky rectifier.
    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_nn_ops
import numpy as np
def conv3d_transpose(value,
                     filter,
                     output_shape,
                     strides,
                     padding="SAME",
                     name=None):
  """The transpose of `conv3d`.
  Args:
    value: A 5-D `Tensor` of type `float` and shape
      `[batch, depth, height, width, in_channels]`.
    filter: A 5-D `Tensor` with the same type as `value` and shape
      `[depth, height, width, output_channels, in_channels]`.  `filter`'s
      `in_channels` dimension must match that of `value`.
    output_shape: A 1-D `Tensor` representing the output shape of the
      deconvolution op.
    strides: A list of ints. The stride of the sliding window for each
      dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the [comment here](https://www.tensorflow.org/api_docs/python/nn.html#convolution)
    name: Optional name for the returned tensor.

  Returns:
    A `Tensor` with the same type as `value`.

  Raises:
    ValueError: If input/output depth does not match `filter`'s shape, or if
      padding is other than `'VALID'` or `'SAME'`.
  """
  with ops.op_scope([value, filter, output_shape], name,
                    "conv3d_transpose") as name:
    value = ops.convert_to_tensor(value, name="value")
    filter = ops.convert_to_tensor(filter, name="filter")
    if not value.get_shape()[4].is_compatible_with(filter.get_shape()[4]):
      raise ValueError("input channels does not match filter's input channels, "
                       "{} != {}".format(value.get_shape()[4], filter.get_shape(
                       )[4]))

    output_shape_ = ops.convert_to_tensor(output_shape, name="output_shape")
    if not output_shape_.get_shape().is_compatible_with(tensor_shape.vector(5)):
      raise ValueError("output_shape must have shape (4,), got {}"
                       .format(output_shape_.get_shape()))

    if isinstance(output_shape, (list, np.ndarray)):
      # output_shape's shape should be == [5] if reached this point.
      if not filter.get_shape()[3].is_compatible_with(output_shape[4]):
        raise ValueError(
            "output_shape does not match filter's output channels, "
            "{} != {}".format(output_shape[4], filter.get_shape()[3]))

    if padding != "VALID" and padding != "SAME":
      raise ValueError("padding must be either VALID or SAME:"
                       " {}".format(padding))

    return gen_nn_ops.conv3d_backprop_input(input=tf.zeros(output_shape_),
                                            filter=filter,
                                            out_backprop=value,
                                            strides=strides,
                                            padding=padding,
                                            name=name)

from tensorflow.python.ops import nn_ops
@ops.RegisterGradient("Conv3DBackpropInput")
def _Conv3DBackpropGrad(op, grad):
  """The derivatives for 3d deconvolution.

  Args:
    op: the Deconvolution op.
    grad: the tensor representing the gradient w.r.t. the output

  Returns:
    the gradients w.r.t. the input and the filter
  """
  return [None,
          nn_ops.conv3d_backprop_filter(grad, op.inputs[1],
                                        op.inputs[2], op.get_attr("strides"),
                                        op.get_attr("padding")),
          nn_ops.conv3d(grad, op.inputs[1], op.get_attr("strides"),
                        op.get_attr("padding"))]

def conv_layer(state_below, scope_name, n_outputs, filter_shape, stddev, wd, filter_stride=(1, 1), nonlinearity=tf.nn.relu, return_params=False):
    """
    A Standard convolutional layer
    assumes that state_below is 4d tensor with shape (batch_size, height, width, channels)
    """
    if nonlinearity is None:
        nonlinearity = tf.identity
    n_inputs = state_below.get_shape().as_list()[3]
    with tf.variable_scope(scope_name) as scope:
        kernel = _variable_with_weight_decay(
            "weights", shape=[filter_shape[0], filter_shape[1], n_inputs, n_outputs],
            wd=wd, stddev=stddev
        )
        scale = _create_variable("scale", [1], tf.constant_initializer(1.0), trainable=False)
        conv = tf.nn.conv2d(state_below, kernel, [1, filter_stride[0], filter_stride[1], 1], padding="SAME")
        biases = _create_variable("biases", [n_outputs], tf.constant_initializer(0.0))
        pre_act = scale * (conv + biases)
        output = nonlinearity(pre_act, name=scope.name)
        _activation_summary(output)
        log(output)
    if not return_params:
        return output
    else:
        if state_below.op.node_def.op == u'Placeholder':
            inp = tf.identity(state_below)
        else:
            inp = state_below
        return output, (kernel, biases, pre_act, output, scale, inp)

def layer_normalized_conv_layer(state_below, scope_name, n_outputs, filter_shape, stddev, wd, eps=.0000001, norm_dims=[1, 2, 3], filter_stride=(1, 1), nonlinearity=tf.nn.relu):
    """
    A Standard convolutional layer
    assumes that state_below is 4d tensor with shape (batch_size, height, width, channels)
    """
    if nonlinearity is None:
        nonlinearity = tf.identity
    n_inputs = state_below.get_shape().as_list()[3]
    with tf.variable_scope(scope_name) as scope:
        kernel = _variable_with_weight_decay(
            "weights", shape=[filter_shape[0], filter_shape[1], n_inputs, n_outputs],
            wd=wd, stddev=stddev
        )
        conv = tf.nn.conv2d(state_below, kernel, [1, filter_stride[0], filter_stride[1], 1], padding='SAME')

        a_mean, a_variance = tf.nn.moments(conv, norm_dims, keep_dims=True)
        a_inv_std = tf.rsqrt(a_variance + eps)

        beta = _create_variable("beta", [n_outputs], tf.constant_initializer(0.0))
        gamma = _create_variable("gamma", [n_outputs], tf.constant_initializer(1.0))

        pre_act = tf.mul(tf.mul(gamma, a_inv_std), (conv - a_mean)) + beta

        output = nonlinearity(pre_act, name=scope.name)
        _activation_summary(output)
    return output

def conv3d_layer(state_below, scope_name, n_outputs, filter_shape, stddev, wd, filter_stride=(1, 1, 1), nonlinearity=tf.nn.relu):
    """
    assumes that state_below is 5d tensor with shape (batch_size, depth, height, width, channels)
    """
    if nonlinearity is None:
        nonlinearity = tf.identity
    n_inputs = state_below.get_shape().as_list()[4]
    with tf.variable_scope(scope_name) as scope:
        # initialize parameters
        kernel = _variable_with_weight_decay(
            "weights", shape=[filter_shape[0], filter_shape[1], filter_shape[2], n_inputs, n_outputs],
            stddev=stddev, wd=wd
        )
        biases = _create_variable("beta", [n_outputs], tf.constant_initializer(0.0))
        # apply convolution
        conv = tf.nn.conv3d(state_below, kernel, [1, filter_stride[0], filter_stride[1], filter_stride[2], 1], padding='SAME')

        bias = tf.nn.bias_add(conv, biases)
        output = nonlinearity(bias, name=scope.name)
        _activation_summary(output)

    return output

def deconv_layer(state_below, scope_name, out_shape, filter_shape, filter_stride, stddev, wd, nonlinearity=tf.nn.relu, return_params=False):
    """
    tied_bias: if true, bias is a vector (global over all outputs), if false is a tensor (local to each pixel location)
    """
    n_outputs = out_shape[-1]
    if nonlinearity is None:
        nonlinearity = tf.identity
    state_below_shape = tf.shape(state_below)
    batch_size = state_below.get_shape().as_list()[0]
    n_inputs = state_below.get_shape().as_list()[-1]
    with tf.variable_scope(scope_name) as scope:
        kernel = _variable_with_weight_decay(
            "weights", shape=[filter_shape[0], filter_shape[1], n_outputs, n_inputs],
            stddev=stddev, wd=wd
        )
        deconv = tf.nn.conv2d_transpose(
            state_below, kernel, out_shape,
            strides=[1, filter_stride[0], filter_stride[1], 1]
        )
        scale = _create_variable("scale", [1], tf.constant_initializer(1.0), trainable=False)
        biases = _create_variable("biases", [n_outputs], tf.constant_initializer(0.0))
        pre_act = scale * tf.nn.bias_add(deconv, biases)

        output = nonlinearity(pre_act, name=scope.name)
        _activation_summary(output)
        log(output)
    if return_params:
        if state_below.op.node_def.op == u'Placeholder':
            inp = tf.identity(state_below)
        else:
            inp = state_below
        return output, (kernel, biases, pre_act, output, scale, inp)
    else:
        return output

def deconv3d_layer(state_below, scope_name, out_shape, filter_shape, filter_stride, stddev, wd, nonlinearity=tf.nn.relu):
    """
    """
    if nonlinearity is None:
        nonlinearity = tf.identity
    n_outputs = out_shape[-1]
    n_inputs = state_below.get_shape().as_list()[4]
    state_below_shape = tf.shape(state_below)
    batch_size = state_below_shape[0]
    tf_out_shape = tf.pack([batch_size, out_shape[0], out_shape[1], out_shape[2], out_shape[3]])
    with tf.variable_scope(scope_name) as scope:
        kernel = _variable_with_weight_decay(
            "weights", shape=[filter_shape[0], filter_shape[1], filter_shape[2], n_outputs, n_inputs],
            stddev=stddev, wd=wd
        )
        deconv = conv3d_transpose(
            state_below, kernel, tf_out_shape,
            strides=[1, filter_stride[0], filter_stride[1], filter_stride[2], 1]
        )

        biases = _create_variable("biases", [n_outputs], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(deconv, biases)

        output = nonlinearity(bias, name=scope.name)
        _activation_summary(output)
    return output


def batch_normalized_conv_layer(state_below, scope_name, n_outputs, filter_shape, stddev, wd, filter_stride=(1, 1), eps=.00001, moving_average_decay=0.99, test=False, nonlinearity=tf.nn.relu):
    """
    Convolutional layer with batch normalization
    assumes that state_below is 4d tensor with shape (batch_size, height, width, channels)
    """
    if nonlinearity is None:
        nonlinearity = tf.identity

    n_inputs = state_below.get_shape().as_list()[3]
    with tf.variable_scope(scope_name) as scope:
        # initialize parameters
        kernel = _variable_with_weight_decay(
            "weights", shape=[filter_shape[0], filter_shape[1], n_inputs, n_outputs],
            stddev=stddev, wd=wd
        )
        beta = _create_variable("beta", [n_outputs], tf.constant_initializer(0.0))
        gamma = _create_variable("gamma", [n_outputs], tf.constant_initializer(1.0))
        # apply conv
        conv = tf.nn.conv2d(state_below, kernel, [1, filter_stride[0], filter_stride[1], 1], padding='SAME')
        # get moments
        conv_mean, conv_variance = tf.nn.moments(conv, [0, 1, 2])
        # get mean and variance variables
        mean_av = _create_variable("bn_mean_av", [n_outputs], tf.constant_initializer(0.0), False)
        variance_av = _create_variable("bn_variance_av", [n_outputs], tf.constant_initializer(1.0), False)

        if not test:
            # if training, set up ema
            ema = tf.train.ExponentialMovingAverage(moving_average_decay)
            ema_apply_op = ema.apply([conv_mean, conv_variance])
            with tf.control_dependencies([ema_apply_op]):
                assign_mean_av = mean_av.assign(ema.average(conv_mean))
                assign_variance_av = variance_av.assign(ema.average(conv_variance))
                with tf.control_dependencies([assign_mean_av, assign_variance_av]):
                    bn = tf.nn.batch_normalization(conv, conv_mean, conv_variance, beta, gamma, eps, name=scope.name+"_bn")
        else:
            bn = tf.nn.batch_normalization(conv, mean_av, variance_av, beta, gamma, eps, name=scope.name+"_bn")

        output = nonlinearity(bn, name=scope.name)
        log(output)
        _activation_summary(output)

    return output

def batch_normalized_3d_conv_layer(state_below, scope_name, n_outputs, filter_shape, stddev, wd, filter_stride=(1, 1, 1), eps=.00001, test=False, nonlinearity=tf.nn.relu):
    """
    Convolutional layer with batch normalization
    assumes that state_below is 5d tensor with shape (batch_size, depth, height, width, channels)
    """
    if nonlinearity is None:
        nonlinearity = tf.identity
    n_inputs = state_below.get_shape().as_list()[4]
    with tf.variable_scope(scope_name) as scope:
        # initialize parameters
        kernel = _variable_with_weight_decay(
            "weights", shape=[filter_shape[0], filter_shape[1], filter_shape[2], n_inputs, n_outputs],
            stddev=stddev, wd=wd
        )
        beta = _create_variable("beta", [n_outputs], tf.constant_initializer(0.0))
        gamma = _create_variable("gamma", [n_outputs], tf.constant_initializer(1.0))
        # apply convolution
        conv = tf.nn.conv3d(state_below, kernel, [1, filter_stride[0], filter_stride[1], filter_stride[2], 1], padding='SAME')
        # get moments
        conv_mean, conv_variance = tf.nn.moments(conv, [0, 1, 2, 3])
        # get mean and variance variables
        mean_av = _create_variable("bn_mean_av", [n_outputs], tf.constant_initializer(0.0), False)
        variance_av = _create_variable("bn_variance_av", [n_outputs], tf.constant_initializer(1.0), False)
        # assign the moments
        if not test:
            # if training, set up ema
            ema = tf.train.ExponentialMovingAverage(moving_average_decay)
            ema_apply_op = ema.apply([conv_mean, conv_variance])
            with tf.control_dependencies([ema_apply_op]):
                assign_mean_av = mean_av.assign(ema.average(conv_mean))
                assign_variance_av = variance_av.assign(ema.average(conv_variance))
                with tf.control_dependencies([assign_mean_av, assign_variance_av]):
                    bn = tf.nn.batch_normalization(
                        conv, conv_mean, conv_variance, beta, gamma, eps, name=scope.name+"_bn"
                    )
        else:
            bn = tf.nn.batch_normalization(conv, mean_av, variance_av, beta, gamma, eps, name=scope.name+"_bn")

        output = nonlinearity(bn, name=scope.name)
        _activation_summary(output)

    return output

def layer_normalized_deconv_layer(state_below, scope_name, out_shape, n_inputs, filter_shape, filter_stride, stddev, wd, nonlinearity=tf.nn.relu, eps=.0000001, norm_dims=[3]):
    """
    Deconvolutional layer with batch normalization

    tied_bias: if true, bias is a vector (global over all outputs), if false is a tensor (local to each pixel location)
    """
    n_outputs = out_shape[-1]
    if nonlinearity is None:
        nonlinearity = tf.identity
    state_below_shape = tf.shape(state_below)
    batch_size = state_below_shape[0]
    tf_out_shape = tf.pack([batch_size, out_shape[0], out_shape[1], out_shape[2]])
    with tf.variable_scope(scope_name) as scope:
        # initialize variables
        kernel = _variable_with_weight_decay(
            "weights", shape=[filter_shape[0], filter_shape[1], n_outputs, n_inputs],
            stddev=stddev, wd=wd
        )

        deconv = tf.nn.conv2d_transpose(
            state_below, kernel, tf_out_shape,
            strides=[1, filter_stride[0], filter_stride[1], 1],
            name=scope.name+"_deconv_applied"
        )
        # get moments
        deconv_mean, deconv_variance = tf.nn.moments(deconv, norm_dims, keep_dims=True)
        deconv_inv_std = tf.rsqrt(deconv_variance + eps)

        # initialize bn parameters
        beta = _create_variable("beta", [n_outputs], tf.constant_initializer(0.0))
        gamma = _create_variable("gamma", [n_outputs], tf.constant_initializer(1.0))

        pre_act = tf.mul(tf.mul(gamma, deconv_inv_std), (deconv - deconv_mean)) + beta
        output = nonlinearity(pre_act, name=scope.name)
        _activation_summary(output)

    return output

def batch_normalized_deconv_layer(state_below, scope_name, out_shape, filter_shape, filter_stride, stddev, wd, nonlinearity=tf.nn.relu, eps=.00001, test=False, moving_average_decay=0.99):
    """
    Deconvolutional layer with batch normalization

    tied_bias: if true, bias is a vector (global over all outputs), if false is a tensor (local to each pixel location)
    """
    n_outputs = out_shape[-1]
    batch_size = state_below.get_shape().as_list()[0]

    if nonlinearity is None:
        nonlinearity = tf.identity

    n_inputs = state_below.get_shape().as_list()[3]
    with tf.variable_scope(scope_name) as scope:
        # initialize variables
        kernel = _variable_with_weight_decay(
            "weights", shape=[filter_shape[0], filter_shape[1], n_outputs, n_inputs],
            stddev=stddev, wd=wd
        )

        deconv = tf.nn.conv2d_transpose(
            state_below, kernel, out_shape,
            strides=[1, filter_stride[0], filter_stride[1], 1],
            name=scope.name+"_deconv_applied"
        )
        # get moments
        deconv_mean, deconv_variance = tf.nn.moments(deconv, [0, 1, 2])
        bias_shape = [n_outputs]

        # initialize bn parameters
        beta = _create_variable("beta", bias_shape, tf.constant_initializer(0.0))
        gamma = _create_variable("gamma", bias_shape, tf.constant_initializer(1.0))

        mean_av = _create_variable("bn_mean", bias_shape, tf.constant_initializer(0.0), False)
        variance_av = _create_variable("bn_variance", bias_shape, tf.constant_initializer(eps), False)

        # assign the moments
        if not test:
            # if training, set up ema
            ema = tf.train.ExponentialMovingAverage(moving_average_decay)
            ema_apply_op = ema.apply([deconv_mean, deconv_variance])
            with tf.control_dependencies([ema_apply_op]):
                assign_mean_av = mean_av.assign(ema.average(deconv_mean))
                assign_variance_av = variance_av.assign(ema.average(deconv_variance))
                with tf.control_dependencies([assign_mean_av, assign_variance_av]):
                    bn = tf.nn.batch_normalization(
                        deconv, deconv_mean, deconv_variance, beta, gamma,
                        eps, name=scope.name+"_bn"
                    )
        else:
            bn = tf.nn.batch_normalization(deconv, mean_av, variance_av, beta, gamma, eps, name=scope.name+"_bn")
        output = nonlinearity(bn, name=scope.name)
        log(output)
        _activation_summary(output)

    return output

def batch_normalized_deconv3d_layer(state_below, scope_name, out_shape, filter_shape, filter_stride, stddev, wd, nonlinearity=tf.nn.relu, eps=.00001, test=False):
    """
    Deconvolutional 3d layer with batch normalization
    """
    n_outputs = out_shape[-1]
    batch_size = tf.shape(state_below)[0]
    tf_out_shape = tf.pack([batch_size, out_shape[0], out_shape[1], out_shape[2], out_shape[3]])
    if nonlinearity is None:
        nonlinearity = tf.identity

    n_inputs = state_below.get_shape().as_list()[4]
    with tf.variable_scope(scope_name) as scope:
        # initialize variables
        kernel = _variable_with_weight_decay(
            "weights", shape=[filter_shape[0], filter_shape[1], filter_shape[2], n_outputs, n_inputs],
            stddev=stddev, wd=wd
        )

        deconv = conv3d_transpose(
            state_below, kernel, tf_out_shape,
            strides=[1, filter_stride[0], filter_stride[1], filter_stride[2], 1],
            name=scope.name+"_deconv_applied"
        )
        # get moments
        deconv_mean, deconv_variance = tf.nn.moments(deconv, [0, 1, 2, 3])
        bias_shape = [n_outputs]


        # initialize bn parameters
        beta = _create_variable("beta", bias_shape, tf.constant_initializer(0.0))
        gamma = _create_variable("gamma", bias_shape, tf.constant_initializer(1.0))

        mean = _create_variable("bn_mean", bias_shape, tf.constant_initializer(0.0), False)
        variance = _create_variable("bn_variance", bias_shape, tf.constant_initializer(1.0), False)

        # assign the moments
        if not test:
            assign_mean = mean.assign(deconv_mean)
            assign_variance = variance.assign(deconv_variance)
            bn = tf.nn.batch_normalization(deconv, deconv_mean, deconv_variance, beta, gamma, eps, name=scope.name+"_bn")
        else:
            bn = tf.nn.batch_normalization(deconv, mean, variance, beta, gamma, eps, name=scope.name+"_bn")
        output = nonlinearity(bn, name=scope.name)
        if not test:
            output = control_flow_ops.with_dependencies(dependencies=[assign_mean, assign_variance], output_tensor=output)
        _activation_summary(output)

    return output

def batch_normalized_linear_layer(state_below, scope_name, n_outputs, stddev, wd, eps=.0000001,  moving_average_decay=0.99, test=False, nonlinearity=tf.nn.relu):
    """
    A linear layer with batch normalization
    assumes input is tensor of shape (batch_size, features)
    """
    if nonlinearity is None:
        nonlinearity = tf.identity
    n_inputs = state_below.get_shape()[1]
    with tf.variable_scope(scope_name) as scope:
        # initialize params
        weight = _variable_with_weight_decay(
            "weights", shape=[n_inputs, n_outputs],
            stddev=stddev, wd=wd
        )
        beta = _create_variable("beta", [n_outputs], tf.constant_initializer(0.0))
        gamma = _create_variable("gamma", [n_outputs], tf.constant_initializer(1.0))

        act = tf.matmul(state_below, weight)
        # get moments
        if act.get_shape().as_list()[1] == 1:
            act_mean_p, act_variance_p = tf.nn.moments(act[:, 0], [0])
            act_mean = tf.expand_dims(act_mean_p, 0)
            act_variance = tf.expand_dims(act_variance_p, 0)
        else:
            act_mean, act_variance = tf.nn.moments(act, [0])

        # get mean and variance variables
        mean_av = _create_variable('bn_mean_av', [n_outputs], tf.constant_initializer(0.0), False)
        variance_av = _create_variable('bn_variance_av', [n_outputs], tf.constant_initializer(eps), False)

        # assign the moments
        if not test:
            ema = tf.train.ExponentialMovingAverage(moving_average_decay)
            ema_apply_op = ema.apply([act_mean, act_variance])
            tf.histogram_summary(act_mean.name, act_mean)
            tf.histogram_summary(act_variance.name, act_variance)
            with tf.control_dependencies([ema_apply_op]):
                assign_mean_av = mean_av.assign(ema.average(act_mean))
                assign_variance_av = variance_av.assign(ema.average(act_variance))
                with tf.control_dependencies([assign_mean_av, assign_variance_av]):
                    bn = tf.nn.batch_normalization(act, act_mean, act_variance, beta, gamma, eps, name=scope.name+"_bn")
        else:
            bn = tf.nn.batch_normalization(act, mean_av, variance_av, beta, gamma, eps, name=scope.name+"_bn")

        output = nonlinearity(bn, name=scope.name)

        _activation_summary(output)
    return output

def layer_normalized_linear_layer(state_below, scope_name, n_outputs, stddev, wd, eps=.0000001, nonlinearity=tf.nn.relu):
    """
    A linear layer with batch normalization
    assumes input is tensor of shape (batch_size, features)
    """
    if n_outputs == 1:
        return linear_layer(state_below, scope_name, n_outputs, stddev, wd, nonlinearity=nonlinearity)

    if nonlinearity is None:
        nonlinearity = tf.identity
    n_inputs = state_below.get_shape()[1]

    with tf.variable_scope(scope_name) as scope:
        # initialize params
        weight = _variable_with_weight_decay(
            "weights", shape=[n_inputs, n_outputs],
            stddev=stddev, wd=wd
        )
        beta = _create_variable("beta", [n_outputs], tf.constant_initializer(0.0))
        gamma = _create_variable("gamma", [n_outputs], tf.constant_initializer(1.0))

        act = tf.matmul(state_below, weight)
        # get moments
        a_mean, a_variance = tf.nn.moments(act, [1], keep_dims=True)
        a_inv_std = tf.rsqrt(a_variance + eps)

        pre_act = tf.mul(tf.mul(gamma, a_inv_std), (act - a_mean)) + beta

        output = nonlinearity(pre_act, name=scope.name)
        _activation_summary(output)
    return output

def weight_normalized_linear_layer(state_below, scope_name, n_outputs, stddev, wd, nonlinearity=tf.nn.relu):
    """
    A linear layer with batch normalization
    assumes input is tensor of shape (batch_size, features)
    """
    if nonlinearity is None:
        nonlinearity = tf.identity
    n_inputs = state_below.get_shape()[1]
    with tf.variable_scope(scope_name) as scope:
        # initialize params
        v = _variable_with_weight_decay(
            "v", shape=[n_inputs, n_outputs],
            stddev=stddev, wd=wd
        )
        v_norm = tf.nn.l2_normalize(v, 0, name="v_norm")
        print(v.get_shape().as_list(), v_norm.get_shape().as_list())
        tf.histogram_summary(v_norm.name, v_norm)
        gamma = _create_variable("gamma", [n_outputs], tf.constant_initializer(1.0))

        weights = tf.mul(v_norm, gamma, name="weights")
        biases = _create_variable('biases', [n_outputs], tf.constant_initializer(0.0))
        activation = tf.nn.xw_plus_b(state_below, weights, biases, name="activation")

        output = nonlinearity(activation, name=scope.name)
        _activation_summary(output)
    return output

def reshape_conv_layer(state_below):
    """
    Reshapes a conv layer activations to be linear. Assumes that batch dimension is 0
    """
    dims = state_below.get_shape().as_list()
    batch_size = -1 if dims[0] is None else dims[0]
    conv_dims = dims[1:]
    dim = 1
    for d in conv_dims:
        dim *= d
    reshape = tf.reshape(state_below, [batch_size, dim])
    return reshape, dim


def reshape_to_conv(state_below, shape):
    """
    Reshapes linear activation (batch_size, num_feats)
    to 2D (batch_size, height, width, chanels)

    shape is (height, width, channels)
    """
    reshape = tf.reshape(state_below, shape)
    return reshape


def linear_layer(state_below, scope_name, n_outputs, stddev, wd, nonlinearity=tf.nn.relu, return_params=False):
    """
    Standard linear neural network layer
    """
    if nonlinearity is None:
        nonlinearity = tf.identity

    n_inputs = state_below.get_shape().as_list()[1]
    with tf.variable_scope(scope_name) as scope:
        weights = _variable_with_weight_decay(
            'weights', [n_inputs, n_outputs],
            stddev=stddev, wd=wd
        )
        scale = _create_variable("scale", [1], tf.constant_initializer(1.0), trainable=False)
        biases = _create_variable(
            'biases', [n_outputs], tf.constant_initializer(0.0)
        )
        pre_act = scale * tf.nn.xw_plus_b(state_below, weights, biases)

        output = nonlinearity(pre_act, name=scope.name)
        _activation_summary(output)
        log(output)
    if not return_params:
        return output
    else:
        if state_below.op.node_def.op == u'Placeholder':
            inp = tf.identity(state_below)
        else:
            inp = state_below
        return output, (weights, biases, pre_act, output, scale, inp)


def global_pooling_layer(state_below, scope_name, pool_type="mean"):
    """
    Performs global pooling over a 2-d convolutional layer's output
    So BxHxWxD -> BxD
    """

    if pool_type == "mean":
        f = tf.nn.avg_pool
    elif pool_type == "max":
        f = tf.nn.max_pool
    dims = state_below.get_shape().as_list()
    im_shape = dims[1:3]
    with tf.variable_scope(scope_name) as scope:
        pooled = f(
            state_below, ksize=[1, im_shape[0], im_shape[1], 1],
            strides=[1, im_shape[0], im_shape[1], 1], padding='SAME', name=scope.name
        )
        out_shape = pooled.get_shape().as_list()
        assert out_shape[1] == 1 and out_shape[2] == 1, out_shape
        num_channels = out_shape[-1]

        reshaped, dim = reshape_conv_layer(pooled)

        reshaped_shape = reshaped.get_shape().as_list()
        assert len(reshaped_shape) == 2, reshaped_shape
        assert reshaped_shape[-1] == num_channels, reshaped_shape
        return reshaped

    return pooled


def pool_to_shape(state_below, shape, scope_name, pool_type="mean"):
    """
    Takes in a tensor [batch, height, width, channels] and pools it to shape [batch, shape[0], shape[1], channels]
    """
    if pool_type == "mean":
        f = tf.nn.avg_pool
    elif pool_type == "max":
        f = tf.nn.max_pool
    else:
        assert False, "Bad pool type"
    im_shape = state_below.get_shape().as_list()
    with tf.variable_scope(scope_name) as scope:
        ksize = [1, im_shape[1] / shape[0], im_shape[2] / shape[1], 1]
        pooled = f(state_below, ksize=ksize, strides=ksize, padding="SAME", name=scope.name)
        out_shape = pooled.get_shape().as_list()
        assert out_shape[1] == shape[0] and out_shape[2] == shape[1]
        return pooled






def global_pooling_output_layer(state_below, scope_name, num_inputs, num_outputs, filter_shape, stddev, wd, pool_type, test):
    """
    Output layer for fully convolutional network. Applies num_outputs filters and pools them to num_outputs logits
    """
    with tf.variable_scope(scope_name) as scope:
        conv_outputs = batch_normalized_conv_layer(
            state_below, "{}_conv_outputs".format(scope.name),
            num_inputs, num_outputs, filter_shape, stddev, wd, test=test
        )
        pooled = global_pooling_layer(conv_outputs, "{}_pooled".format(scope.name), pool_type)
    return pooled


def randomized_relu(state_below, irange, name=None, is_training=False):
    """
    Randomized rectified linear unit
    """
    if not is_training:
        # if testing, use standard relu
        return tf.nn.relu(state_below, name=name)
    else:
        # sample in irange around 1 for pos side
        pos_rand = tf.random_uniform(tf.shape(state_below), 1 - (irange / 2.0), 1 + (irange / 2.0))
        # sampel in irange around 0 for neg side
        neg_rand = tf.random_uniform(tf.shape(state_below), -irange / 2.0, irange / 2.0)

        pos = tf.mul(state_below,  pos_rand)
        neg = tf.mul(state_below, neg_rand)

        where_pos = tf.greater(state_below, 0.0)

        out = tf.select(where_pos, pos, neg, name=name)
        return out


def spatial_softmax(state_below, scope_name, alpha):
    """
    Runs a spatial softmax operation (e^(state_below/alpha) / sum(e^(state_below/alpha))
    assumes state below is of shape (batch_size, height, width, channels)
    """
    assert len(state_below.get_shape().as_list()) == 4
    with tf.variable_scope(scope_name) as scope:
        scaled = state_below / alpha
        exped = tf.exp(scaled)
        sums = tf.reduce_sum(exped, reduction_indices=[1, 2], keep_dims=True)
        softmax = tf.div(exped, sums, name=scope.name)
    return softmax









