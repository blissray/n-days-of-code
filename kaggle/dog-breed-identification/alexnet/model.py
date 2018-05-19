import tensorflow as tf

'''
AlexNet using TensorFlow library.
References:
- Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton.
  ImageNet Classification with Deep Convolutional Neural Networks. NIPS, 2012.
Link:
- https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
'''

__author__ = 'Sungchul Choi'


class AlexNet(object):
    def __init__(self, X, n_in=(224*224*3), n_out=120, dropout_prob=0.5):

        weights = {
            "wc1": [11, 11, 3, 96],
            "wc2": [5, 5, 96, 256],
            "wc3": [3, 3, 256, 384],
            "wc4": [3, 3, 384, 384],
            "wc5": [3, 3, 384, 256],
            "wf1": [6*6*256, 4096],
            "wf2": [4096, 4096],
            "wf3": [4096, n_out]
        }

        biases = {
            "bc1": [0.0, 96],
            "bc2": [1.0, 256],
            "bc3": [0.0, 384],
            "bc4": [1.0, 384],
            "bc5": [1.0, 256],
            "bf1": [1.0, 4096],
            "bf2": [1.0, 4096],
            "bf3": [1.0, n_out]
        }

        self.X = tf.reshape(X, [-1, 224, 224, 3])
        self.dropout_prob = dropout_prob

        # 1st convolutional layer
        # in - [224, 224, 3], out - [55, 55, 96]
        layer_name = "conv1"
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            self.w_c1 = self._weight_variable(layer_name, weights["wc1"])
            self.b_c1 = self._bias_variable(layer_name, biases["bc1"])

            self.conv1 = tf.nn.conv2d(self.X, self.w_c1, strides=[1, 4, 4, 1], padding="SAME", name="conv1")
            self.conv1 = tf.nn.bias_add(self.conv1, self.b_c1)
            self.conv1 = tf.nn.relu(self.conv1)
            self.conv1 = tf.nn.local_response_normalization(self.conv1, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
            self.conv1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        # 2nd convolutional layer
        # in - [55, 55, 96], out - [27, 27, 256]
        layer_name = "conv2"
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            self.w_c2 = self._weight_variable(layer_name, weights["wc2"])
            self.b_c2 = self._bias_variable(layer_name, biases["bc2"])

            self.conv2 = tf.nn.conv2d(self.conv1, self.w_c2, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
            self.conv2 = tf.nn.bias_add(self.conv2, self.b_c2)
            self.conv2 = tf.nn.relu(self.conv2)
            self.conv2 = tf.nn.local_response_normalization(self.conv2, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
            self.conv2 = tf.nn.max_pool(self.conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        # 3rd convolutional layer
        # in - [27, 27, 256], out - [13, 13, 384]
        layer_name = "conv3"
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            self.w_c3 = self._weight_variable(layer_name, weights["wc3"])
            self.b_c3 = self._bias_variable(layer_name, biases["bc3"])

            self.conv3 = tf.nn.conv2d(self.conv2, self.w_c3, strides=[1, 1, 1, 1], padding="SAME", name="conv3")
            self.conv3 = tf.nn.bias_add(self.conv3, self.b_c3)
            self.conv3 = tf.nn.relu(self.conv3)

        # 4th convolutional layer
        # in - [13, 13, 384], out - [13, 13, 384]
        layer_name = "conv4"
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            self.w_c4 = self._weight_variable(layer_name, weights["wc4"])
            self.b_c4 = self._bias_variable(layer_name, biases["bc4"])

            self.conv4 = tf.nn.conv2d(self.conv3, self.w_c4, strides=[1, 1, 1, 1], padding="SAME", name="conv4")
            self.conv4 = tf.nn.bias_add(self.conv4, self.b_c4)
            self.conv4 = tf.nn.relu(self.conv4)

        # 5th convolutional layer
        # in - [13, 13, 384], out - [6, 6, 256]
        layer_name = "conv5"
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            self.w_c5 = self._weight_variable(layer_name, weights["wc5"])
            self.b_c5 = self._bias_variable(layer_name, biases["bc5"])

            self.conv5 = tf.nn.conv2d(self.conv4, self.w_c5, strides=[1, 1, 1, 1], padding="SAME", name="conv5")
            self.conv5 = tf.nn.bias_add(self.conv5, self.b_c5)
            self.conv5 = tf.nn.relu(self.conv5)
            # self.conv5 = tf.nn.max_pool(self.conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
            self.conv5 = tf.nn.max_pool(self.conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")


        # stretching out the 5th convolutional layer into a long n-dimensional tensor
        shape = [-1, weights['wf1'][0]]
        print("---------------")
        print(shape)
        print(self.conv5 )
        print("---------------")

        self.flatten_vector = tf.reshape(self.conv5, shape)

        # 1st fully connected layer
        # in -  [28, 28, 256], out - [4096]
        layer_name = "dense1"
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            self.w_f1 = self._weight_variable(layer_name, weights["wf1"])
            self.b_f1 = self._bias_variable(layer_name, biases["bf1"])

            self.fc1 = tf.nn.bias_add(tf.matmul(self.flatten_vector, self.w_f1), self.b_f1)
            self.fc1 = tf.nn.relu(self.fc1)
            self.fc1 = tf.nn.dropout(self.fc1, keep_prob=self.dropout_prob)

        # 2nd fully connected layer
        # in -  [4096], out - [4096]
        layer_name = "dense2"
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            self.w_f2 = self._weight_variable(layer_name, weights["wf2"])
            self.b_f2 = self._bias_variable(layer_name, biases["bf2"])

            self.fc2 = tf.nn.bias_add(tf.matmul(self.fc1, self.w_f2), self.b_f2)
            self.fc2 = tf.nn.relu(self.fc2)
            self.fc2 = tf.nn.dropout(self.fc2, keep_prob=self.dropout_prob)

        # 3rd fully connected layer
        layer_name = "output"
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            self.w_f3 = self._weight_variable(layer_name, weights["wf3"])
            self.b_f3 = self._bias_variable(layer_name, biases["bf3"])
            self.hypothesis = tf.matmul(self.fc2 , self.w_f3) + self.b_f3

    def _weight_variable(self, layer_name, shape):
        init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        return tf.get_variable(
            layer_name + "_w", shape=shape,
            initializer=init)

    def _bias_variable(self, layer_name, shape_value):
        init = tf.constant_initializer(value=shape_value[0])
        return tf.get_variable(
            layer_name + "_bias", shape=shape_value[1],
            initializer=init)
