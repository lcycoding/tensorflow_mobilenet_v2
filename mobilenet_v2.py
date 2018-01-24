# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""MobileNet v2.

MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and different
head (for example: embeddings, localization and classification).

This is a revised version based on (https://arxiv.org/pdf/1801.04381.pdf)

"""

# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim


# arg scope for mnetv2
def mobilenet_v2_arg_scope(weight_decay=0.00004, is_training = True, stddev = 0.09, regularize_depthwise=False, dropout_prob=0.999):



    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None

    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weights_init,
                        activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'center': True, 'scale': True, 'decay': 0.997, 'epsilon': 0.001}):
        with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
            with slim.arg_scope([slim.separable_conv2d],
                                weights_regularizer=depthwise_regularizer, depth_multiplier=1.0):
                with slim.arg_scope([slim.dropout], is_training=is_training, keep_prob=dropout_prob) as sc:
                    return sc


# inverted bottleneck block
def inverted_block(neural_net, input_filter, output_filter, expand, stride):
    # fundamental network struture of inverted_residual_block
    res_block = neural_net
    # pointwise conv2d, expand feature up to 6 times ( recorded in mobilenetv2 paper )
    res_block = slim.conv2d(inputs=res_block, num_outputs=input_filter * expand, kernel_size=[1, 1])
    # depthwise conv2d
    res_block = slim.separable_conv2d(inputs=res_block, num_outputs=None, kernel_size=[3, 3], stride=stride)
    res_block = slim.conv2d(inputs=res_block, num_outputs=output_filter, kernel_size=[1, 1], activation_fn=None)
    # stride 2 blocks
    if stride == 2:
        return res_block
    # stride 1 block
    else:
        if input_filter != output_filter:
            neural_net = slim.conv2d(inputs=neural_net, num_outputs=output_filter, kernel_size=[1, 1], activation_fn=None)
        return tf.add(res_block, neural_net)


# repeated inverted bottleneck block
def pile_of_blocks(neural_net, expand, output_filter, blocks, stride):
    input_filter = neural_net.shape[3].value

    # first layer needs to consider stride
    neural_net = inverted_block(neural_net, input_filter, output_filter, expand, stride)

    for _ in range(1, blocks):
        neural_net = inverted_block(neural_net, input_filter, output_filter, expand, 1)

    return neural_net


def mobilenet_v2_base(inputs,
                      final_endpoint='',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      scope=None):
    endpoints = dict()
    expand = 6

    with tf.variable_scope(scope):
        with slim.arg_scope(mobilenet_v2_arg_scope()):
            neural_net = tf.identity(inputs)
            neural_net = slim.conv2d(neural_net, num_outputs=32, kernel_size=[3, 3], scope='conv1', stride=2)
            endpoints['conv1_1'] = neural_net
            neural_net = pile_of_blocks(neural_net=neural_net, expand=1, output_filter=16, blocks=1, stride=1)
            endpoints['inverted_residual_block1'] = neural_net
            neural_net = pile_of_blocks(neural_net=neural_net, expand=expand, output_filter=24, blocks=2, stride=2)
            endpoints['inverted_residual_block2_3'] = neural_net
            neural_net = pile_of_blocks(neural_net=neural_net, expand=expand, output_filter=32, blocks=3, stride=2)
            endpoints['inverted_residual_block4_6'] = neural_net
            neural_net = pile_of_blocks(neural_net=neural_net, expand=expand, output_filter=64, blocks=4, stride=1)
            endpoints['inverted_residual_block7_10'] = neural_net
            neural_net = pile_of_blocks(neural_net=neural_net, expand=expand, output_filter=96, blocks=3, stride=2)
            endpoints['inverted_residual_block11_13'] = neural_net
            neural_net = pile_of_blocks(neural_net=neural_net, expand=expand, output_filter=160,blocks=3, stride=2)
            endpoints['inverted_residual_block14_16'] = neural_net
            neural_net = pile_of_blocks(neural_net=neural_net, expand=expand, output_filter=320,blocks=1, stride=1)
            endpoints['inverted_residual_block17'] = neural_net
            neural_net = slim.conv2d(neural_net, 1280, [1, 1], scope='bottleneck')
            endpoints['bottleneck'] = neural_net

    return neural_net, endpoints


def mobilenet_v2(inputs,
                 num_classes=1000,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=None,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='MobilenetV2',
                 global_pool=False):
  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
      raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                     len(input_shape))

  with tf.variable_scope(scope, 'MobilenetV2', [inputs], reuse=reuse) as scope:
      neural_net, endpoints = mobilenet_v2_base(inputs, scope=scope,
                                          min_depth=min_depth,
                                          depth_multiplier=depth_multiplier,
                                          conv_defs=conv_defs)
      with tf.variable_scope('Logits'):
          neural_net = slim.avg_pool2d(neural_net, [7, 7])
          # 1 x 1 x k
          neural_net = slim.conv2d(neural_net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='features')
          neural_net = slim.flatten(neural_net)
          endpoints['features'] = neural_net
          if not num_classes:
              return neural_net, endpoints

          logits = tf.layers.dense(neural_net, num_classes, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.00004))

      endpoints['Logits'] = logits

      if prediction_fn:
          endpoints['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, endpoints

mobilenet_v2.default_image_size = 224


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


mobilenet_v2_075 = wrapped_partial(mobilenet_v2, depth_multiplier=0.75)
mobilenet_v2_050 = wrapped_partial(mobilenet_v2, depth_multiplier=0.50)
mobilenet_v2_025 = wrapped_partial(mobilenet_v2, depth_multiplier=0.25)


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out
