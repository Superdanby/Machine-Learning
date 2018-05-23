import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
slim = tf.contrib.slim

def VGG16(inputs, is_training, n_classes):
    inputs = tf.cast(inputs, tf.float32)
    inputs = ((inputs / 255.0)-0.5)*2
    #Use Pretrained Base Model
    with tf.variable_scope("vgg_16"):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
    #Append fully connected layer
    net = slim.flatten(net)
    net = slim.fully_connected(net, 4096,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=slim.l2_regularizer(0.0005),
            scope='finetune/fc1')
    net = slim.fully_connected(net, 4096,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=slim.l2_regularizer(0.0005),
            scope='finetune/fc2')
    net = slim.fully_connected(net, n_classes,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=slim.l2_regularizer(0.0005),
            scope='finetune/classification')
    return net
