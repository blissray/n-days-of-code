import tensorflow as tf
from model import AlexNet
import numpy as np
import os

def loss_function(hypothesis, Y):
    with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=hypothesis, labels=Y), name="softmax_cross_entropy")
    return loss


def training(loss, learning_rate_init_value):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_init_value, momentum=0.9)
    train_step = optimizer.minimize(loss)
    return train_step


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
          'images': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.string),
      })


    image = tf.decode_raw(features['images'], tf.uint8)
    # image = tf.reshape(image, [-1, 128,128,3])
    # image = tf.image.rgb_to_grayscale(image)

    image = tf.cast(image, tf.float32)

    label = tf.decode_raw(features['label'], tf.uint8)
    label = tf.cast(label, tf.float32)

    return image, label


def train(config_dict):
    n_out = 120
    n_in = 224*224*3

    X = tf.placeholder(tf.float32, [None, n_in])
    Y = tf.placeholder(tf.float32, [None, n_out])
    dropout_prob = tf.placeholder(tf.float32, shape=(), name="init")


    mlp_model = AlexNet(X, n_in=n_in, n_out=n_out, dropout_prob=dropout_prob)
    loss = loss_function(mlp_model.hypothesis, Y)
    train_step = training(loss,
        learning_rate_init_value=config_dict["learning_rate"])

    import glob
    filenames = glob.glob("./tfrecords/cropping/train/*.tfrecords")
    train_dataset = tf.data.TFRecordDataset(filenames)
    train_dataset = train_dataset.map(decode)
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(config_dict["batch_size"])
    iterator = tf.data.Iterator.from_structure(
                            train_dataset.output_types,
                            train_dataset.output_shapes)
    next_element = iterator.get_next()
    train_init_op = iterator.make_initializer(train_dataset)

    filenames = glob.glob("./tfrecords/cropping/test/*.tfrecords")
    test_dataset = tf.data.TFRecordDataset(filenames)
    test_dataset = test_dataset.map(decode)
    test_dataset = test_dataset.batch(512)
    test_iterator = tf.data.Iterator.from_structure(
                            test_dataset.output_types,
                            test_dataset.output_shapes)
    test_next_element = test_iterator.get_next()
    test_init_op = test_iterator.make_initializer(test_dataset)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(config_dict["training_epochs"]):
            # initialize the iterator on the training data
            sess.run(train_init_op)

            # get each element of the training dataset until the end is reached
            cost = 0
            total_batch = 0


            while True:
                try:
                    batch = sess.run(next_element)
                    batch_xs = batch[0]
                    # print(batch_xs.shape)
                    batch_ys = batch[1]

                    feed_dict = {X: batch_xs, Y: batch_ys, dropout_prob: 0.5}
                    c, _ = sess.run([loss, train_step], feed_dict=feed_dict)
                    cost += c
                    total_batch += 1
                except tf.errors.OutOfRangeError:
                    print("End of training dataset.")
                    break

            avg_cost = cost / total_batch
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(
                    avg_cost))

            sess.run(test_init_op)
            total_correct = 0
            total_data = 0
            while True:
                try:
                    batch = sess.run(test_next_element)
                    batch_x = batch[0]
                    batch_y = batch[1]

                    correct_prediction = tf.equal(
                            tf.argmax(mlp_model.hypothesis, axis=1),
                            tf.argmax(Y, axis=1))
                    # correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
                    result = sess.run(correct_prediction, feed_dict={
                          X: batch_x, Y: batch_y, dropout_prob: 1.0})
                    total_correct += np.sum(result)
                    total_data += len(result)

                    print('result:', sess.run(tf.argmax(mlp_model.hypothesis, axis=1)[:50], feed_dict={
                          X: batch_x, Y: batch_y, dropout_prob: 1.0}))
                except tf.errors.OutOfRangeError:
                    print("End of test dataset.")
                    break
            print("Accuracy : " , (total_correct/total_data))


if __name__ == "__main__":
    config_dict = {
        "learning_rate": 0.0001,
        "training_epochs": 5000,
        "batch_size": 64
        }

    train(config_dict)
