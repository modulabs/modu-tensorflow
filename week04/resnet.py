from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

slim = tf.contrib.slim

sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

np.random.seed(21)
tf.set_random_seed(21)

# train 6만개, test 1만개
(train_data, train_labels), (test_data, test_labels) = \
        tf.keras.datasets.mnist.load_data()

train_data = train_data/ 255.
train_labels = np.asarray(train_labels, dtype=np.int32)

test_data = test_data/ 255.
test_labels = np.asarray(test_labels, dtype=np.int32)

#train_data = train_data[:10000]
#train_labels = train_labels[:10000]

test_data = train_data[:500]
test_labels = train_labels[:500]

batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
train_dataset = train_dataset.shuffle(buffer_size= 10000)
train_dataset = train_dataset.batch(batch_size = batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
test_dataset = test_dataset.shuffle(buffer_size = 10000)
test_dataset = test_dataset.batch(batch_size = len(test_data))

print(train_dataset.output_shapes)
print(train_dataset.output_types)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle,
                                                train_dataset.output_types,
                                                train_dataset.output_shapes)
x, y = iterator.get_next()
x = tf.cast(x, dtype= tf.float32)
y = tf.cast(y, dtype= tf.int32)


def resnet_block(input, num_filter, kernel_size, n, scope = '',stride = 2):
    for i in range(n):
        if i == 0 :
            net = slim.conv2d(input, num_filter, kernel_size, stride = stride,\
                              scope = scope + "_" + str(i) + "_1" )
            net = slim.conv2d(net, num_filter, kernel_size,\
                              scope = scope + "_" + str(i) + "_2" )
        else:
            tem_net = net
            net = slim.conv2d(net, num_filter, kernel_size, \
                              scope = scope + "_" + str(i) + "_1")
            net = slim.conv2d(net, num_filter, kernel_size, \
                              scope = scope + "_" + str(i) + "_2")
            net = net + tem_net
    return net

def cnn_model_fn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    net = resnet_block(x_image, 16, [3, 3], 3, scope="first", stride= 1)
    print(net)
    net = resnet_block(net, 32, [3, 3], 5, scope="second")
    print(net)
    net = resnet_block(net, 64, [3, 3], 7, scope="third")
    print(net)

    flat = slim.flatten(net, scope='flatten')
    logits = slim.fully_connected(flat, 10, activation_fn=None, scope='logits')

    return logits, x_image

logits, x_image = cnn_model_fn(x)
cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits= logits)
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_iterator = train_dataset.make_initializable_iterator()
train_handle = sess.run(train_iterator.string_handle())

max_epochs = 20
step = 0

saver = tf.train.Saver()

for epoch in range(max_epochs):
    sess.run(train_iterator.initializer)

    try:
        ckpt_state = tf.train.get_checkpoint_state("my_resnet")
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        step = int(ckpt_state.model_checkpoint_path.split('-')[1])

    except Exception as ex :
        print("불러오기 실패")
        print(ex)

    start_time = time.time()
    while True:
        try:
            _, loss = sess.run([train_step, cross_entropy],
                               feed_dict={handle: train_handle})
            if step % 100 == 0 :
                print("step: {}, loss: {}".format(step, loss))
            step += 1
        except tf.errors.OutOfRangeError:
            print("End of dataset")
            break

    save_path = saver.save(sess, "my_resnet/resnet.ckpt", global_step= step)
    print("Model saved in file: %s" % save_path)
    print("Epochs: {}, Elasped time: {}".format(epoch, time.time()-start_time))

print("training done!!")




test_iterator = test_dataset.make_initializable_iterator()
test_handle = sess.run(test_iterator.string_handle())
sess.run(test_iterator.initializer)

accuracy, acc_op = tf.metrics.accuracy(labels=y, predictions=tf.argmax(logits, 1), name='accuracy')
sess.run(tf.local_variables_initializer())

sess.run(acc_op, feed_dict={handle: test_handle})
print("test accuracy:", sess.run(accuracy, feed_dict={handle: test_handle}))