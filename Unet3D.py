import tensorflow as tf
import numpy as np
from collections import OrderedDict
import os
import shutil
import logging
from tensorflow.python import debug as tf_debug

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

_gpu_options = tf.ConfigProto()
_gpu_options.gpu_options.allow_growth = True

"""utils"""


def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].

    :param data: the array to crop
    :param shape: the target shape
    """
    offset0 = (data.shape[1] - shape[1]) // 2
    offset1 = (data.shape[2] - shape[2]) // 2
    offset2 = (data.shape[3] - shape[3]) // 2
    return data[:, offset0:(-offset0), offset1:(-offset1), offset2:(-offset2)]


def combine_img_prediction(data, gt, pred):
    """
    Combines the data, grouth thruth and the prediction into one rgb image

    :param data: the data tensor
    :param gt: the ground thruth tensor
    :param pred: the prediction tensor

    :returns img: the concatenated rgb image
    """
    img = np.concatenate((to_rgb(data[..., 0]), to_rgb(gt[..., 1]), to_rgb(pred[..., 1])), axis=1)
    return img


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255)

    :param img: the array to convert [nx, ny, channels]

    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= (np.amax(img) + 0.0005)
    img *= 255
    return img


def save_image(img, path):
    """
    Writes the image to disk

    :param img: the rgb image to save
    :param path: the target path
    """
    # Image.fromarray(img.round().astype(np.uint8)).save(path, 'JPEG', dpi=[300, 300], quality=90)
    return 0


""" layers """


def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def weight_variable_devonc(shape, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv3d(x, W, keep_prob_):
    # x = tf.Print(x, [tf.shape(x)])
    conv_3d = tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='VALID', name='conv3d')
    print(conv_3d)
    return tf.nn.dropout(conv_3d, keep_prob_)


def deconv3d(x, W, stride):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] * 2, x_shape[4] // 2])
    a = tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, stride, stride, stride, 1], padding='VALID',
                               name='deconv3d')
    print(a)
    return a


def max_pool(x, n):
    return tf.nn.max_pool3d(x, ksize=[1, n, n, n, 1], strides=[1, n, n, n, 1], padding='VALID')


def crop_and_concat(x1, x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2,
               (x1_shape[3] - x2_shape[3]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 4)


def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map, tf.reverse(exponential_map, [False, False, False, False, True]))
    return tf.div(exponential_map, evidence, name="pixel_wise_softmax")


def pixel_wise_softmax_2(output_map):
    out_flatten = tf.reshape(output_map, [-1, tf.shape(output_map)[4]])
    res = tf.nn.softmax(out_flatten)
    return tf.reshape(res, tf.shape(output_map))


def cross_entropy(y_, output_map):
    return -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(output_map, 1e-10, 1.0)), name="cross_entropy")


#####################################

##model##

def create_conv_net(x, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3,summaries=True, pool_size=2):
    """
    Creates a new convolutional unet for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    logging.info(
        "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))

    # Placeholder for the input image
    nz = tf.shape(x)[1]
    nx = tf.shape(x)[2]
    ny = tf.shape(x)[3]

    x_image = tf.reshape(x, tf.stack([-1, nz, nx, ny, channels]))
    in_node = x_image
    batch_size = tf.shape(x_image)[0]

    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()

    in_size = 1000
    size = in_size
    # down layers
    for layer in range(0, layers):
        features = 2 ** layer * features_root
        stddev = np.sqrt(2 / (filter_size ** 2 * features))
        if layer == 0:
            w1 = weight_variable([filter_size, filter_size, filter_size, channels, features], stddev)
        else:
            w1 = weight_variable([filter_size, filter_size, filter_size, features // 2, features], stddev)

        w2 = weight_variable([filter_size, filter_size, filter_size, features, features], stddev)
        b1 = bias_variable([features])
        b2 = bias_variable([features])

        conv1 = conv3d(in_node, w1, keep_prob)
        tmp_h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv3d(tmp_h_conv, w2, keep_prob)
        dw_h_convs[layer] = tf.nn.relu(conv2 + b2)

        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))

        size -= 4
        if layer < layers - 1:
            pools[layer] = max_pool(dw_h_convs[layer], pool_size)
            in_node = pools[layer]
            size /= 2

    in_node = dw_h_convs[layers - 1]

    # up layers
    for layer in range(layers - 2, -1, -1):
        features = 2 ** (layer + 1) * features_root
        stddev = np.sqrt(2 / (filter_size ** 2 * features))

        wd = weight_variable_devonc([pool_size, pool_size, pool_size, features // 2, features], stddev)
        bd = bias_variable([features // 2])
        h_deconv = tf.nn.relu(deconv3d(in_node, wd, pool_size) + bd)
        h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
        deconv[layer] = h_deconv_concat

        w1 = weight_variable([filter_size, filter_size, filter_size, features, features // 2], stddev)
        w2 = weight_variable([filter_size, filter_size, filter_size, features // 2, features // 2], stddev)
        b1 = bias_variable([features // 2])
        b2 = bias_variable([features // 2])

        conv1 = conv3d(h_deconv_concat, w1, keep_prob)
        h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv3d(h_conv, w2, keep_prob)
        in_node = tf.nn.relu(conv2 + b2)
        up_h_convs[layer] = in_node

        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))

        size *= 2
        size -= 4

    # Output Map
    weight = weight_variable([1, 1, 1, features_root, n_class], stddev)  # !!!
    bias = bias_variable([n_class])
    conv = conv3d(in_node, weight, tf.constant(1.0))
    output_map = tf.nn.relu(conv + bias)
    up_h_convs["out"] = output_map

    if summaries:
        for i, (c1, c2) in enumerate(convs):
            tf.summary.image('summary_conv_%03d_01' % i, get_image_summary(c1))
            tf.summary.image('summary_conv_%03d_02' % i, get_image_summary(c2))

        for k in pools.keys():
            tf.summary.image('summary_pool_%03d' % k, get_image_summary(pools[k]))

        for k in deconv.keys():
            tf.summary.image('summary_deconv_concat_%03d' % k, get_image_summary(deconv[k]))

        for k in dw_h_convs.keys():
            tf.summary.histogram("dw_convolution_%03d" % k + '/activations', dw_h_convs[k])

        for k in up_h_convs.keys():
            tf.summary.histogram("up_convolution_%s" % k + '/activations', up_h_convs[k])

    variables = []
    for w1, w2 in weights:
        variables.append(w1)
        variables.append(w2)

    for b1, b2 in biases:
        variables.append(b1)
        variables.append(b2)

    return output_map, variables, int(in_size - size)


class Unet3D(object):
    """
    A unet implementation

    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """

    def __init__(self, channels=3, n_class=2, cost="cross_entropy", cost_kwargs={}, **kwargs):
        tf.reset_default_graph()

        self.n_class = n_class
        self.summaries = kwargs.get("summaries", True)

        self.x = tf.placeholder("float", shape=[None, None, None, None, channels], name='x')
        self.y = tf.placeholder("float", shape=[None, None, None, None, n_class], name='y')
        # self.x = tf.placeholder("float", shape=[None, None, None, None, channels])
        # self.y = tf.placeholder("float", shape=[None, None, None, None, n_class])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        logits, self.variables, self.offset = create_conv_net(
            self.x, self.keep_prob, channels, n_class, **kwargs)

        self.logits = logits
        self.cost = self._get_cost(logits, cost, cost_kwargs)

        self.gradients_node = tf.gradients(self.cost, self.variables)

        #logits = tf.Print(logits, [tf.shape(logits)], message="LOGITS shape: ", summarize=100)
        #self_y = tf.Print(self.y, [tf.shape(self.y)], message="LABELS shape: ", summarize=100)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.y, [-1, self.n_class]),
            logits=tf.reshape(logits, [-1, self.n_class])), axis=-1)

        self.predicter = pixel_wise_softmax_2(logits)
        self.correct_pred = tf.equal(tf.argmax(self.predicter, 4), tf.argmax(self.y, 4))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def _get_cost(self, logits, cost_name, cost_kwargs):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are:
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """
        #logits = tf.Print(logits, [tf.shape(logits)], message='Logits:', summarize=100)
        #y_printed = tf.Print(self.y, [tf.shape(self.y)], message='Labels:', summarize=100)


        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(self.y, [-1, self.n_class])
        if cost_name == "cross_entropy":
            class_weights = cost_kwargs.pop("class_weights", None)

            softmax_crossent = tf.nn.softmax_cross_entropy_with_logits(
                logits=flat_logits, labels=flat_labels
            )

            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

                weight_map = tf.multiply(flat_labels, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)

                loss_map = softmax_crossent
                weighted_loss = tf.multiply(loss_map, weight_map)

                loss = tf.reduce_mean(weighted_loss)

            else:
                loss = tf.reduce_mean(softmax_crossent)

        elif cost_name == "dice_coefficient":
            eps = 1e-5
            #logits = tf.Print(logits, [tf.shape(logits)], message="logits shape = ", summarize=100)
            prediction = pixel_wise_softmax_2(logits)
            intersection = tf.reduce_sum(prediction * self.y)
            union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
            loss = -(2 * intersection / (union))

        else:
            raise ValueError("Unknown cost function: " % cost_name)

        regularizer = cost_kwargs.pop("regularizer", None)
        if regularizer is not None:
            regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
            loss += (regularizer * regularizers)
        #loss_printed = tf.Print(loss, [loss], message="loss = ", summarize=100)
        #loss = loss_printed
        return loss

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2)
        """

        init = tf.global_variables_initializer()
        with tf.Session(config=_gpu_options) as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})

        return prediction

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)


class Trainer(object):
    """
        Trains a unet instance

        :param net: the unet instance to train
        :param batch_size: size of training batch
        :param norm_grads: (optional) true if normalized gradients should be added to the summaries
        :param optimizer: (optional) name of the optimizer to use (momentum or adam)
        :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer

        """

    verification_batch_size = 1

    def __init__(self, net, batch_size=1, norm_grads=False, optimizer="momentum", opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.norm_grads = norm_grads
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs

    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)

            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=training_iters,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)

            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.cost,
                                                                               global_step=global_step)
        elif self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            self.learning_rate_node = tf.Variable(learning_rate)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                               **self.opt_kwargs).minimize(self.net.cost,
                                                                           global_step=global_step)

        return optimizer

    def _initialize(self, training_iters, output_path, restore, prediction_path):
        global_step = tf.Variable(0)

        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]))

        if self.net.summaries and self.norm_grads:
           tf.summary.histogram('norm_grads', self.norm_gradients_node)

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('cross_entropy', self.net.cross_entropy)
        tf.summary.scalar('accuracy', self.net.accuracy)

        self.optimizer = self._get_optimizer(training_iters, global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)

        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init

    def train(self, data_provider, output_path, training_iters=10, epochs=100, dropout=0.75, display_step=1,
              restore=False, write_graph=False, prediction_path='prediction'):
        """
            Lauches the training process

            :param data_provider: callable returning training and verification data
            :param output_path: path where to store checkpoints
            :param training_iters: number of training mini batch iteration
            :param epochs: number of epochs
            :param dropout: dropout probability
            :param display_step: number of steps till outputting stats
            :param restore: Flag if previous model should be restored
            :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
            :param prediction_path: path where to save predictions on each epoch
            """
        save_path = os.path.join(output_path, "model.cpkt")
        if epochs == 0:
            return save_path

        init = self._initialize(training_iters, output_path, restore, prediction_path)
        check = tf.add_check_numerics_ops()

        with tf.Session(config=_gpu_options) as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            test_x, test_y = data_provider()


            pred_shape = self.store_prediction(sess, test_x, test_y, "_init")

            summary_writer = tf.summary.FileWriter(os.path.join(output_path, "summary/train"),
                                                   graph=sess.graph)
            summary_writer_test = tf.summary.FileWriter(os.path.join(output_path, "summary/test"),
                                                   graph=sess.graph)
            summary_writer_const = tf.summary.FileWriter(os.path.join(output_path, "summary/test_const"),
                                                        graph=sess.graph)

            logging.info("Start optimization")

            avg_gradients = None
            for epoch in range(epochs):
                total_loss = 0
                for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
                    batch_x, batch_y = data_provider()

                    # Run optimization op (backprop)
                    *_, loss, lr, gradients = sess.run(
                        (check, self.optimizer, self.net.cost, self.learning_rate_node,
                         self.net.gradients_node),
                        feed_dict={self.net.x: batch_x,
                                   self.net.y: crop_to_shape(batch_y, pred_shape),
                                   self.net.keep_prob: dropout})

                    if self.net.summaries and self.norm_grads:
                        avg_gradients = _update_avg_gradients(avg_gradients, gradients, step)
                        norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                        self.norm_gradients_node.assign(norm_gradients).eval()

                    if step % display_step == 0:
                        val_x, val_y = data_provider()
                        self.output_minibatch_stats(sess, summary_writer, step, batch_x,
                                                    crop_to_shape(batch_y, pred_shape))
                        self.output_minibatch_stats_test(sess, summary_writer_test, step, val_x,
                                                    crop_to_shape(val_y, pred_shape))
                        self.output_minibatch_stats_const(sess, summary_writer_const, step, test_x,
                                                         crop_to_shape(test_y, pred_shape))

                    total_loss += loss

                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                self.store_prediction(sess, test_x, test_y, "epoch_%s" % epoch)

                save_path = self.net.save(sess, save_path)

            logging.info("Optimization Finished!")
            return save_path

    def store_prediction(self, sess, batch_x, batch_y, name):
        # t = tf.get_default_graph().get_tensor_by_name('deconv3d:0')
        # print('shapes', sess.run(tf.shape(t), feed_dict={
        #    self.net.x: batch_x, self.net.y: batch_y, self.net.keep_prob: 1.
        # }))

        prediction = sess.run(self.net.predicter, feed_dict={self.net.x: batch_x,
                                                             self.net.y: batch_y,
                                                             self.net.keep_prob: 1.})
        pred_shape = prediction.shape
        y_cropped = crop_to_shape(batch_y, pred_shape)
        x_cropped = crop_to_shape(batch_x, pred_shape)
        # print(y_cropped.shape, "KKKKKKKKKKKKKK")
        loss = sess.run(self.net.cost, feed_dict={self.net.x: batch_x,
                                                  self.net.y: y_cropped,
                                                  self.net.keep_prob: 1.})

        logging.info("Verification error= {:.1f}%, loss= {:.4f}".format(error_rate(prediction,
                                                                                   crop_to_shape(batch_y,
                                                                                                 prediction.shape)),
                                                                        loss))
        img = combine_img_prediction(x_cropped, y_cropped, prediction)
        save_image(img, "%s/%s.jpg" % (self.prediction_path, name))  # TODO: save 3D image

        return pred_shape

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info(
            "Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters),
                                                                            lr))

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        check = tf.add_check_numerics_ops()
        _, summary_str, loss, acc, predictions = sess.run([check, self.summary_op,
                                                        self.net.cost,
                                                        self.net.accuracy,
                                                        self.net.predicter],
                                                       feed_dict={self.net.x: batch_x,
                                                                  self.net.y: batch_y,
                                                                  self.net.keep_prob: 1.})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info(
            "Iter {:}, Minibatch Loss= {:.4f}, "
            "Training Accuracy= {:.4f}, "
            "Minibatch error= {:.1f}%".format(step, loss, acc, error_rate(predictions, batch_y))
        )

    def output_minibatch_stats_test(self, sess, summary_writer, step, batch_x, batch_y):
            # Calculate batch loss and accuracy
        check = tf.add_check_numerics_ops()
        _, summary_str, loss, acc, predictions = sess.run([check, self.summary_op,
                                                           self.net.cost,
                                                           self.net.accuracy,
                                                           self.net.predicter],
                                                          feed_dict={self.net.x: batch_x,
                                                                     self.net.y: batch_y,
                                                                     self.net.keep_prob: 1.})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info(
                "Iter {:}, Test Loss= {:.4f}, "
                "Test Accuracy= {:.4f}, "
                "Test error= {:.1f}%".format(step, loss, acc, error_rate(predictions, batch_y))
        )

    def output_minibatch_stats_const(self, sess, summary_writer, step, batch_x, batch_y):
            # Calculate batch loss and accuracy
        check = tf.add_check_numerics_ops()
        _, summary_str, loss, acc, predictions = sess.run([check, self.summary_op,
                                                           self.net.cost,
                                                           self.net.accuracy,
                                                           self.net.predicter],
                                                          feed_dict={self.net.x: batch_x,
                                                                     self.net.y: batch_y,
                                                                     self.net.keep_prob: 1.})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info(
                "Iter {:}, Val Loss= {:.4f}, "
                "Val Accuracy= {:.4f}, "
                "Val error= {:.1f}%".format(step, loss, acc, error_rate(predictions, batch_y))
        )


def _update_avg_gradients(avg_gradients, gradients, step):
    if avg_gradients is None:
        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
    for i in range(len(gradients)):
        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step + 1)))) + (gradients[i] / (step + 1))
        return avg_gradients


def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """

    return 100.0 - (
            100.0 *
            np.sum(np.argmax(predictions, 4) == np.argmax(labels, 4)) /
            (predictions.shape[0] * predictions.shape[1] * predictions.shape[2]* predictions.shape[3]))


def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, 0, idx), (1, -1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_d = tf.shape(img)[1]
    img_w = tf.shape(img)[2]
    img_h = tf.shape(img)[3]

    V = tf.reshape(V, tf.stack((img_d, img_w, img_h, 1)))
    V = tf.transpose(V, (3, 0, 1, 2))
    V = tf.reshape(V, tf.stack((-1, img_d, img_w, img_h, 1)))
    return V
