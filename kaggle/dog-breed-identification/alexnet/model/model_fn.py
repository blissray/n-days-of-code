"""Define the model."""

import tensorflow as tf

def _weight_variable(layer_name, shape):
    init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    return tf.get_variable(
        layer_name + "_w", shape=shape,
        initializer=init)

def _bias_variable(layer_name, shape_value):
    init = tf.constant_initializer(value=shape_value[0])
    return tf.get_variable(
        layer_name + "_bias", shape=shape_value[1],
        initializer=init)

def build_model(is_training, inputs, params):
    weights = {
        "wc1": [11, 11, 3, 96],
        "wc2": [5, 5, 96, 256],
        "wc3": [3, 3, 256, 384],
        "wc4": [3, 3, 384, 384],
        "wc5": [3, 3, 384, 256],
        "wf1": [6*6*256, 4096],
        "wf2": [4096, 4096],
        "wf3": [4096, params.num_labels]
    }

    biases = {
        "bc1": [0.0, 96],
        "bc2": [1.0, 256],
        "bc3": [0.0, 384],
        "bc4": [1.0, 384],
        "bc5": [1.0, 256],
        "bf1": [1.0, 4096],
        "bf2": [1.0, 4096],
        "bf3": [1.0, params.num_labels]
    }

    images = inputs['images']

    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 3]

    X = images
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool

    X = tf.reshape(X, [-1, 224, 224, 3])

    # 1st convolutional layer
    # in - [224, 224, 3], out - [55, 55, 96]
    layer_name = "conv1"
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        w_c1 = _weight_variable(layer_name, weights["wc1"])
        b_c1 = _bias_variable(layer_name, biases["bc1"])

        conv1 = tf.nn.conv2d(X, w_c1, strides=[1, 4, 4, 1],
                             padding="SAME", name="conv1")
        conv1 = tf.nn.bias_add(conv1, b_c1)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.local_response_normalization(
                conv1, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="VALID")

    # 2nd convolutional layer
    # in - [55, 55, 96], out - [27, 27, 256]
    layer_name = "conv2"
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        w_c2 = _weight_variable(layer_name, weights["wc2"])
        b_c2 = _bias_variable(layer_name, biases["bc2"])

        conv2 = tf.nn.conv2d(
            conv1, w_c2, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
        conv2 = tf.nn.bias_add(conv2, b_c2)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.local_response_normalization(
            conv2, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
        conv2 = tf.nn.max_pool(
            conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # 3rd convolutional layer
    # in - [27, 27, 256], out - [13, 13, 384]
    layer_name = "conv3"
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        w_c3 = _weight_variable(layer_name, weights["wc3"])
        b_c3 = _bias_variable(layer_name, biases["bc3"])

        conv3 = tf.nn.conv2d(
            conv2, w_c3, strides=[1, 1, 1, 1], padding="SAME", name="conv3")
        conv3 = tf.nn.bias_add(conv3, b_c3)
        conv3 = tf.nn.relu(conv3)

    # 4th convolutional layer
    # in - [13, 13, 384], out - [13, 13, 384]
    layer_name = "conv4"
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        w_c4 = _weight_variable(layer_name, weights["wc4"])
        b_c4 = _bias_variable(layer_name, biases["bc4"])

        conv4 = tf.nn.conv2d(
            conv3, w_c4, strides=[1, 1, 1, 1], padding="SAME", name="conv4")
        conv4 = tf.nn.bias_add(conv4, b_c4)
        conv4 = tf.nn.relu(conv4)

    # 5th convolutional layer
    # in - [13, 13, 384], out - [6, 6, 256]
    layer_name = "conv5"
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        w_c5 = _weight_variable(layer_name, weights["wc5"])
        b_c5 = _bias_variable(layer_name, biases["bc5"])

        conv5 = tf.nn.conv2d(
            conv4, w_c5, strides=[1, 1, 1, 1], padding="SAME", name="conv5")
        conv5 = tf.nn.bias_add(conv5, b_c5)
        conv5 = tf.nn.relu(conv5)
        # conv5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
        conv5 = tf.nn.max_pool(
            conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    dense_shape = [-1, weights['wf1'][0]]
    flatten_vector = tf.reshape(conv5, dense_shape)

    # 1st fully connected layer
    # in -  [6 * 6 * 256], out - [4096]
    layer_name = "dense1"
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        w_f1 = _weight_variable(layer_name, weights["wf1"])
        b_f1 = _bias_variable(layer_name, biases["bf1"])

        fc1 = tf.nn.bias_add(tf.matmul(flatten_vector, w_f1), b_f1)
        fc1 = tf.nn.relu(fc1)
        if is_training:
            fc1 = tf.nn.dropout(fc1, keep_prob=params.dropout_prob)

    # 2nd fully connected layer
    # in -  [4096], out - [4096]
    layer_name = "dense2"
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        w_f2 = _weight_variable(layer_name, weights["wf2"])
        b_f2 = _bias_variable(layer_name, biases["bf2"])

        fc2 = tf.nn.bias_add(tf.matmul(fc1, w_f2), b_f2)
        fc2 = tf.nn.relu(fc2)
        if is_training:
            fc2 = tf.nn.dropout(fc2, keep_prob=params.dropout_prob)

    # 3rd fully connected layer
    layer_name = "output"
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        w_f3 = _weight_variable(layer_name, weights["wf3"])
        b_f3 = _bias_variable(layer_name, biases["bf3"])
        logits = tf.matmul(fc2, w_f3) + b_f3

    return logits


def model_fn(mode, inputs, params, reuse=False):
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(is_training, inputs, params)
        predictions = tf.argmax(logits, 1)

    # Define loss and accuracy
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency
            # to update the moving mean and variance for batch normalization
            with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['images'])

    #TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    for label in range(0, params.num_labels):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
