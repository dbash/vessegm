import tensorflow as tf
import numpy as np
from collections import OrderedDict
import os
import shutil
import logging

from matplotlib import pyplot

"""utils"""
def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].

    :param data: the array to crop
    :param shape: the target shape
    """
    offset0 = (data.shape[1] - shape[1]) // 2
    offset1 = (data.shape[2] - shape[2]) // 2
    return data[:, offset0:(-offset0), offset1:(-offset1)]


def combine_img_prediction(data, gt, pred):
    """
    Combines the data, grouth thruth and the prediction into one rgb image

    :param data: the data tensor
    :param gt: the ground thruth tensor
    :param pred: the prediction tensor

    :returns img: the concatenated rgb image
    """
    ny = pred.shape[2]
    ch = data.shape[3]
    img = np.concatenate((to_rgb(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)),
                          to_rgb(crop_to_shape(gt[..., 1], pred.shape).reshape(-1, ny, 1)),
                          to_rgb(pred[..., 1].reshape(-1, ny, 1))), axis=1)
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
    img /= np.amax(img)
    img *= 255
    return img


def save_image(img, path):
    """
    Writes the image to disk

    :param img: the rgb image to save
    :param path: the target path
    """
    #Image.fromarray(img.round().astype(np.uint8)).save(path, 'JPEG', dpi=[300, 300], quality=90)
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
    conv_3d = tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='VALID')
    return tf.nn.dropout(conv_3d, keep_prob_)


def deconv3d(x, W, stride):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] * 2, x_shape[4] // 2])
    return tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, stride, stride, stride, 1], padding='VALID')


def max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, n, 1], strides=[1, n, n, n, 1], padding='VALID')


def crop_and_concat(x1, x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, abs(x1_shape[1] - x2_shape[1]) // 2, abs(x1_shape[2] - x2_shape[2]) // 2,
               abs(x1_shape[3] - x2_shape[3]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)


def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map, tf.reverse(exponential_map, [False, False, False, False, True]))
    return tf.div(exponential_map, evidence, name="pixel_wise_softmax")


def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 4, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, 1, tf.shape(output_map)[4]]))
    return tf.div(exponential_map, tensor_sum_exp)


def cross_entropy(y_, output_map):
    return -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(output_map, 1e-10, 1.0)), name="cross_entropy")

#####################################

##model##

def create_conv_net(x, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3, pool_size=2):
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

    # Placeholder for the input image
    nz = tf.shape(x)[1]
    nx = tf.shape(x)[2]
    ny = tf.shape(x)[3]

    x_image = tf.reshape(x, tf.stack([-1,nz, nx, ny, channels]))
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
            w1 = weight_variable([filter_size, filter_size,filter_size, channels, features], stddev)
        else:
            w1 = weight_variable([filter_size, filter_size,filter_size, features // 2, features], stddev)

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

        wd = weight_variable_devonc([pool_size, pool_size,pool_size, features // 2, features], stddev)
        bd = bias_variable([features // 2])
        h_deconv = tf.nn.relu(deconv3d(in_node, wd, pool_size) + bd)
        h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
        deconv[layer] = h_deconv_concat

        w1 = weight_variable([filter_size, filter_size,filter_size, features, features // 2], stddev)
        w2 = weight_variable([filter_size, filter_size,filter_size, features // 2, features // 2], stddev)
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
    weight = weight_variable([1, 1,1, features_root, n_class], stddev) #!!!
    bias = bias_variable([n_class])
    conv = conv3d(in_node, weight, tf.constant(1.0))
    output_map = tf.nn.relu(conv + bias)
    up_h_convs["out"] = output_map


    variables = []
    for w1, w2 in weights:
        variables.append(w1)
        variables.append(w2)

    for b1, b2 in biases:
        variables.append(b1)
        variables.append(b2)

    return output_map, variables, int(in_size - size)


class Unet(object):
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

        self.x = tf.placeholder("float", shape=[None,None, None, None, channels])
        self.y = tf.placeholder("float", shape=[None,None, None, None, n_class])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        logits, self.variables, self.offset = create_conv_net(self.x, self.keep_prob, channels, n_class, **kwargs)

        self.cost = self._get_cost(logits, cost, cost_kwargs)

        self.gradients_node = tf.gradients(self.cost, self.variables)

        self.cross_entropy = tf.reduce_mean(cross_entropy(tf.reshape(self.y, [-1, n_class]),
                                                          tf.reshape(pixel_wise_softmax_2(logits), [-1, n_class])))

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

        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(self.y, [-1, self.n_class])
        if cost_name == "cross_entropy":
            class_weights = cost_kwargs.pop("class_weights", None)

            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

                weight_map = tf.multiply(flat_labels, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)

                loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                   labels=flat_labels)
                weighted_loss = tf.multiply(loss_map, weight_map)

                loss = tf.reduce_mean(weighted_loss)

            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                              labels=flat_labels))
        elif cost_name == "dice_coefficient":
            eps = 1e-5
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

        return loss

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2)
        """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2],x_test.shape[3], self.n_class))
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

class Trainer(object):
        """
        Trains a unet instance

        :param net: the unet instance to train
        :param batch_size: size of training batch
        :param norm_grads: (optional) true if normalized gradients should be added to the summaries
        :param optimizer: (optional) name of the optimizer to use (momentum or adam)
        :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer

        """

        verification_batch_size = 4

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
                shutil.rmtree(abs_prediction_path, ignore_errors=True)
                shutil.rmtree(output_path, ignore_errors=True)

            if not os.path.exists(abs_prediction_path):
                os.makedirs(abs_prediction_path)

            if not os.path.exists(output_path):
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

            with tf.Session() as sess:
                if write_graph:
                    tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

                sess.run(init)

                if restore:
                    ckpt = tf.train.get_checkpoint_state(output_path)
                    if ckpt and ckpt.model_checkpoint_path:
                        self.net.restore(sess, ckpt.model_checkpoint_path)

                test_x, test_y = data_provider(self.verification_batch_size)
                pred_shape = self.store_prediction(sess, test_x, test_y, "_init")

                summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)

                avg_gradients = None
                for epoch in range(epochs):
                    total_loss = 0
                    for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
                        batch_x, batch_y = data_provider(self.batch_size)

                        # Run optimization op (backprop)
                        _, loss, lr, gradients = sess.run(
                            (self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node),
                            feed_dict={self.net.x: batch_x,
                                       self.net.y: crop_to_shape(batch_y, pred_shape),
                                       self.net.keep_prob: dropout})

                        if self.net.summaries and self.norm_grads:
                            avg_gradients = _update_avg_gradients(avg_gradients, gradients, step)
                            norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                            self.norm_gradients_node.assign(norm_gradients).eval()

                        if step % display_step == 0:
                            self.output_minibatch_stats(sess, summary_writer, step, batch_x,
                                                        crop_to_shape(batch_y, pred_shape))

                        total_loss += loss

                    self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                    self.store_prediction(sess, test_x, test_y, "epoch_%s" % epoch)

                    save_path = self.net.save(sess, save_path)

                return save_path

        def store_prediction(self, sess, batch_x, batch_y, name):
            prediction = sess.run(self.net.predicter, feed_dict={self.net.x: batch_x,
                                                                 self.net.y: batch_y,
                                                                 self.net.keep_prob: 1.})
            pred_shape = prediction.shape

            loss = sess.run(self.net.cost, feed_dict={self.net.x: batch_x,
                                                      self.net.y: crop_to_shape(batch_y, pred_shape),
                                                      self.net.keep_prob: 1.})


            img = combine_img_prediction(batch_x, batch_y, prediction)
            save_image(img, "%s/%s.jpg" % (self.prediction_path, name)) #TODO: save 3D image

            return pred_shape

        def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
            logging.info(
                "Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters),
                                                                                lr))

        def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
            # Calculate batch loss and accuracy
            summary_str, loss, acc, predictions = sess.run([self.summary_op,
                                                            self.net.cost,
                                                            self.net.accuracy,
                                                            self.net.predicter],
                                                           feed_dict={self.net.x: batch_x,
                                                                      self.net.y: batch_y,
                                                                      self.net.keep_prob: 1.})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
            logging.info(
                "Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(step,
                                                                                                               loss,
                                                                                                               acc,
                                                                                                               error_rate(
                                                                                                                   predictions,
                                                                                                                   batch_y)))

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
            np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
            (predictions.shape[0] * predictions.shape[1] * predictions.shape[2]))

def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V
