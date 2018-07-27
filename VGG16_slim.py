import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import tensorflow as tf

slim = tf.contrib.slim

sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

import vgg

inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])

with slim.arg_scope(vgg.vgg_arg_scope()):
    with slim.arg_scope(vgg.vgg_arg_scope()):
        outputs, end_points = vgg.vgg_16(
            inputs, num_classes=None, is_training=False)

my_images = Image.open('input_image/에펠탑2.jpg')
my_images = my_images.resize((224, 224))
my_images = np.asarray(my_images) / 255.
my_images -= 0.5
my_images *= 2.0

my_images = np.expand_dims(my_images, axis=0)
saver = tf.train.Saver()

with tf.Session(config=sess_config) as sess:
    # use saver object to load variables from the saved model
    saver.restore(sess, "vgg_16_ckpt/vgg_16.ckpt")

    # print conv1_1 weight itself
    conv1_1_w = sess.run(tf.trainable_variables()[0])
    print(tf.trainable_variables()[0])
    print(tf.trainable_variables()[1])
    print(tf.trainable_variables()[2])
    print(conv1_1_w.shape)
    # print feature maps
    conv1_1, conv2_1, \
    conv3_2, conv4_3, \
    conv5_3 = sess.run([end_points['vgg_16/conv1/conv1_1'],
                        end_points['vgg_16/conv2/conv2_1'],
                        end_points['vgg_16/conv3/conv3_2'],
                        end_points['vgg_16/conv4/conv4_3'],
                        end_points['vgg_16/conv5/conv5_3']],
                       feed_dict={inputs: my_images})
print(conv1_1.shape)
print(conv2_1.shape)
print(conv3_2.shape)

channel_index = 30
plt.subplot(231)
plt.imshow(conv1_1[0,:,:,channel_index], cmap='gray')
plt.subplot(232)
plt.imshow(conv2_1[0,:,:,channel_index], cmap='gray')
plt.subplot(233)
plt.imshow(conv3_2[0,:,:,channel_index], cmap='gray')
plt.subplot(234)
plt.imshow(conv4_3[0,:,:,channel_index], cmap='gray')
plt.subplot(235)
plt.imshow(conv5_3[0,:,:,channel_index], cmap='gray')
plt.show()