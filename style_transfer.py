#import sys
#sys.path.append("$HOME/models/research/slim/")

import os
import time

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

from PIL import Image

import tensorflow as tf

slim = tf.contrib.slim

sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
os.environ["CUDA_VISIBLE_DEVICES"]="1"

input_data_dir = 'input_image/'
content_image_name = '에펠탑.jpg'
style_image_name = 'Gogh_The_Starry_Night.jpg'
noise_ratio = 0.6
max_L = 1024 # upper bound of image size
style_loss_weight = np.array([0.5, 1.0, 1.5, 3.0, 4.0])
content_weight = 0.0001
style_weight = 1.0
learning_rate = 0.01
max_steps = 100
print_steps = 10


def vgg_16(inputs,
           reuse=False,
           scope='vgg_16'):
    """Oxford Net VGG 16-Layers version D Example

    My Note: This code is modified version of vgg_16 which is loacted on `models/research/slim/nets/vgg.py`
    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      reuse: whether or not the model is being reused.
      scope: Optional scope for the variables.

    Returns:
      net: the output of the logits layer (if num_classes is a non-zero integer),
        or the input to the logits layer (if num_classes is 0 or None).
      end_points: a dict of tensors with intermediate activations.
    """
    with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],       # 일단 max 플링으로.
                            outputs_collections=end_points_collection):
            # 여기를 직접 채워 넣으시면 됩니다.
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

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points


content_image_ = Image.open(os.path.join(input_data_dir, content_image_name))
style_image_ = Image.open(os.path.join(input_data_dir, style_image_name))


def image_resize_with_upper_bound(image, max_L=max_L):
    """Resize images

    Args:
      image: PIL image format
      max_L: upper bound of the image size

    Returns:
      image: resized image with PIL format
      h: resized height
      w: resized width
      """
    w, h = image.size
    if np.max(np.array([h, w])) > max_L:
        if h < w:
            h = int(max_L * h / w)
            w = max_L
        else:
            w = int(max_L * w / h)
            h = max_L
    image = image.resize((w, h))
    return image, h, w

content_image_, content_image_h, content_image_w = image_resize_with_upper_bound(content_image_)
style_image_w, style_image_h = style_image_.size

# 여기를 직접 채워 넣으시면 됩니다.
content_image_p = tf.placeholder(dtype=tf.float32, shape = [1, content_image_h, content_image_w, 3], name = 'content_image_p')
style_image_p = tf.placeholder(dtype=tf.float32, shape = [1, style_image_h, style_image_w, 3], name = 'style_image_p')
# content_image, style_image를 tf.Variable로 바꾸기 위해 tf.placeholder와 같은 shape의 zero Tensor를 만듦
content_image = tf.get_variable(name = 'content_image', shape = [1, content_image_h, content_image_w, 3]\
                                , initializer = tf.zeros_initializer)
style_image = tf.get_variable(name = 'style_image' , shape = [1, style_image_h, style_image_w, 3]\
                              , initializer = tf.zeros_initializer)
# 생성할 이미지 크기 설정
generated_image = tf.get_variable(name='generated_image',
                                  shape=[1, content_image_h, content_image_w, 3],
                                  initializer=tf.random_uniform_initializer(minval=-0.2, maxval=0.2))

# tf.placeholder를 tf.Variable로 바꿈
content_image_op = content_image.assign(content_image_p)
style_image_op = style_image.assign(style_image_p)
# 초기 이미지는 content_image에 random noise를 섞음, 이거 때문에 이미지 사이즈를 (1, 576, 1024, 3)으로 맞추기.
generated_image_op = generated_image.assign(generated_image * noise_ratio + \
                                            content_image_p * (1.0 - noise_ratio))
# 여기를 직접 채워 넣으시면 됩니다.
# generated_image는 매 update 후에 [-1, 1] 사이로 clipping
generated_image_clipping = generated_image.assign(tf.clip_by_value(generated_image, -1, 1))

# 여기를 직접 채워 넣으시면 됩니다.
_, feature_maps_c = vgg_16(content_image, reuse = tf.AUTO_REUSE) # input: content_image
_, feature_maps_s = vgg_16(style_image, reuse = tf.AUTO_REUSE) # input: style_image
_, feature_maps_g = vgg_16(generated_image, reuse = tf.AUTO_REUSE) # input: generated_image

with tf.Session() as sess:
  writer = tf.summary.FileWriter("./graphs/02_style_transfer", sess.graph)
  writer.close()

content_layers = feature_maps_c['vgg_16/conv4/conv4_2']
style_layers = [feature_maps_s['vgg_16/conv1/conv1_1'],
                feature_maps_s['vgg_16/conv2/conv2_1'],
                feature_maps_s['vgg_16/conv3/conv3_1'],
                feature_maps_s['vgg_16/conv4/conv4_1'],
                feature_maps_s['vgg_16/conv5/conv5_1']]
generated_layers = [feature_maps_g['vgg_16/conv4/conv4_2'],
                    feature_maps_g['vgg_16/conv1/conv1_1'],
                    feature_maps_g['vgg_16/conv2/conv2_1'],
                    feature_maps_g['vgg_16/conv3/conv3_1'],
                    feature_maps_g['vgg_16/conv4/conv4_1'],
                    feature_maps_g['vgg_16/conv5/conv5_1']]


def content_loss(P, F, scope):
    """Calculate the content loss function between
    the feature maps of content image and generated image.

    Args:
      P: the feature maps of the content image
      F: the feature maps of the generated image
      scope: scope

    Returns:
      loss: content loss (mean squared loss)
    """
    # 여기를 직접 채워 넣으시면 됩니다.
    loss = 0.5 * tf.reduce_mean(tf.pow(P-F, 2))
    return loss


def style_loss(style_layers, generated_layers, scope):
    """Calculate the style loss function between
    the gram matrix of feature maps of style image and generated image.

    Args:
      style_layers: list of the feature maps of the style image
      generated_layers: list of the feature maps of the generated image
      scope: scope

    Returns:
      loss: style loss (mean squared loss)
    """

    def _style_loss_one_layer(feature_map_s, feature_map_g):
        """Calculate the style loss for one layer.

        Args:
          feature_map_s: the feature map of the style image
            - G: the gram matrix of the feature_map_s
          feature_map_g: the feature map of the generated image
            - A: the gram matrix of the feature_map_g

        Returns:
          loss: style loss for one layer (mean squared loss)
        """
        _, h, w, c = feature_map_s.get_shape().as_list()
        G, _ = _gram_matrix(feature_map_s)
        print("피쳐 맵 shape" + str(feature_map_s.shape))
        print(G.shape)
        A, num_weight = _gram_matrix(feature_map_g)
        print(A.shape)
        # 여기를 직접 채워 넣으시면 됩니다.
        loss =  tf.pow(G-A, 2)                  # 1/4 * N^2 * M^2 연산은 밑에서 함.
        return loss, num_weight

    def _gram_matrix(feature_map):
        """Calculate the gram matrix for the feature map

        Args:
          feature_map: 4-rank Tensor [1, height, width, channels]
            - F = 2-rank Tensor [h * w, channels]

        Returns:
          gram_matrix: 2-rank Tensor [c, c] (F.transpose x F)
        """
        # 여기를 직접 채워 넣으시면 됩니다.
        F = tf.squeeze(feature_map, axis=0)
        h, w, c = F.shape
        F = tf.reshape(F, [-1, h*w])
        return tf.matmul(F, tf.transpose(F)), h*w*c  # the num of weight 도  함께 반환

    assert len(style_layers) == len(generated_layers)

    loss = 0.0
    for i in range(len(style_layers)):
        loss_one, num_weight = _style_loss_one_layer(style_layers[i], generated_layers[i])
        C = int(loss_one.shape[0])       # 각 층의 필터 수.
        M = int(num_weight)        # 각 필터마다 weight의 수.
        print("C " + str(type(C)))
        print("M " + str(type(M)))
        loss_one = tf.reduce_sum(loss_one)
        loss_one = 0.25 * (1 / (C^2 * M^2)) * loss_one
        loss += loss_one * style_loss_weight[i]

    return loss

loss_c = content_loss(content_layers, generated_layers[0],
                      scope='content_loss')
loss_s = style_loss(style_layers, generated_layers[1:],
                    scope='style_loss')

with tf.variable_scope('total_loss'):
  total_loss = content_weight * loss_c + style_weight * loss_s

# 여기를 직접 채워 넣으시면 됩니다.
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)


def image_preprocessing(image):
    """image preprocessing
    transform image pixel value: int [0, 255] -> float [-1.0, 1.0]

    Args:
      image: PIL image format

    Returns:
      image: float type numpy array with shape [1, image_h, image_w, 3] which is in [-1, 1]
    """
    image = np.asarray(image) / 255.
    image -= 0.5
    image *= 2.0

    image = np.expand_dims(image, axis=0)
    return image


def print_image(image):
    """print image

    Args:
      image: 4-rank np.array [1, h, w, 3]
    """
    print_image = np.squeeze(image, axis=0)
    print_image = np.clip(print_image, -1.0, 1.0)
    print_image += 1.0
    print_image *= 0.5

    plt.axis('off')
    plt.imshow(print_image)
    plt.show()

content_image_ = image_preprocessing(content_image_)
style_image_ = image_preprocessing(style_image_)

v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_16')

saver = tf.train.Saver(var_list=v)

with tf.Session(config=sess_config) as sess:
    sess.run(tf.global_variables_initializer())
    # content_image_와 style_image_를 tf.placeholder에 넣고 tf.Variable로 assign
    sess.run([content_image_op, style_image_op, generated_image_op],
             feed_dict={content_image_p: content_image_,
                        style_image_p: style_image_})

    _, generated_image_ = sess.run([generated_image_clipping, generated_image])
    print_image(content_image_)
    print_image(style_image_)
    print_image(generated_image_)  # initial_image = content_image + small noise

    # use saver object to load variables from the saved model
    saver.restore(sess, "vgg_16_ckpt/vgg_16.ckpt")

    start_time = time.time()
    for step in range(max_steps + 1):

        _, loss_, _, generated_image_ = \
            sess.run([train_op, total_loss, generated_image_clipping, generated_image])

        if step % print_steps == 0:
            duration = time.time() - start_time
            start_time = time.time()
            print("step: {}  loss: {}  duration: {}".format(step,
                                                            loss_,
                                                            duration))

            print_image(generated_image_)
    print('training done!')


def save_image(image, content_image_name, style_image_name):
    """print image

    Args:
      image: 4-rank np.array [1, h, w, 3]
      filename: name of saved image
    """
    save_image = np.squeeze(image, axis=0)
    save_image = np.clip(save_image, -1.0, 1.0)
    save_image += 1.0
    save_image /= 2.0

    save_image = Image.fromarray(np.uint8(save_image * 255))
    filename = os.path.splitext(os.path.basename(content_image_name))[0] + '_'
    filename += os.path.splitext(os.path.basename(style_image_name))[0] + '.jpg'
    save_image.save(filename)

save_image(generated_image_, content_image_name, style_image_name)