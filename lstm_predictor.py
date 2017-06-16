import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


def x_sin(x):
    return x * np.sin(x)

numLabels = 1
numEpochs = 5000
LEARNING_RATE = 0.001
batch_size = 20
KEEP_PROB = 0.8
TIMESTEPS = 5
hidden_size = 10
num_layers_rnn = 2
num_steps = 5

def sin_cos(x):
    return pd.DataFrame(dict(a=np.sin(x), b=np.cos(x)), index=x)


def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [3, 4, 5]
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float32)


def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits data to training, validation and testing parts
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test


def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))


def load_csvdata(rawdata, time_steps, seperate=False):
    data = rawdata
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


def generate_data(fct, x, time_steps, seperate=False):
    """generates data with based on a function fct"""
    data = fct(x)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)

model_params = {"learning_rate": LEARNING_RATE, 'steps': TIMESTEPS, "keep_prob": KEEP_PROB, 'num_units': 5}


def lstm_model(features, targets, mode, params):
    """
    Creates a deep model based on:
        * stacked lstm cells
        * an optional dense layers
    :param num_units: the size of the cells.
    :param rnn_layers: list of int or dict
                         * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                         * list of dict: [{steps: int, keep_prob: int}, ...]
    :param dense_layers: list of nodes for each layer
    :return: the model definition
    """

    def lstm_cells(layers):
        if isinstance(layers[0], dict):
            return [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(5,state_is_tuple=True),layer['keep_prob'])
                    if layer.get('keep_prob') else tf.contrib.rnn.BasicLSTMCell(5,state_is_tuple=True) for layer in layers]

        return [tf.contrib.rnn.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]

    def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return tflayers.stack(input_layers, tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers

    def _lstm_model(features, y):

        # features = tf.reshape(features, [-1, 10, 1])
        # cell = tf.contrib.rnn.LSTMCell(256, state_is_tuple=True)
        # outputs, state = tf.nn.dynamic_rnn(cell,
        #                                    features,
        #                                    # sequence_length=[10] * batch_size,
        #                                    dtype=tf.float32)
        #
        # predictions = tf.contrib.layers.fully_connected(state.h,
        #                                                 num_outputs=1,
        #                                                 activation_fn=None)
        # loss = tf.reduce_sum(tf.pow(predictions - targets[-1], 2))


        # cell = tf.contrib.rnn.BasicLSTMCell(5, state_is_tuple=True)
        # cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(5, state_is_tuple=True), 0.9)
        # stacked_lstm = tf.contrib.rnn.MultiRNNCell([cell for _ in range(1)], state_is_tuple=True)

        layers=[]
        layers.append(params)
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells(layers), state_is_tuple=True)


        features = tf.unstack(features, axis=1, num=10)

        output, layers = tf.contrib.rnn.static_rnn(stacked_lstm, features, dtype=dtypes.float32)
        output = dnn_layers(output[-1], [10,10])
        prediction, loss = tflearn.models.linear_regression(output, y)
        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
            learning_rate=params["learning_rate"])
        eval_metric_ops = {
            "rmse": tf.metrics.root_mean_squared_error(
                tf.cast(targets, tf.float32), prediction)
        }
        predictions_dict = {"classes":prediction}

        return model_fn_lib.ModelFnOps(
            mode=mode,
            predictions=predictions_dict,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)

    return _lstm_model(features, targets)


def sine_model(features, targets, mode, params):

    def lstm_cell(cell_size):
        return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(cell_size, forget_bias=0.0,
                                                                          state_is_tuple=True),
                                             output_keep_prob=0.8)

    def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return tflayers.stack(input_layers, tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers

    def _lstm_model(features, targets):




        cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(5, state_is_tuple=True), 0.9)
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([cell for _ in range(1)], state_is_tuple=True)
        features = tf.unstack(features, axis=1, num=10)

        output, layers = tf.contrib.rnn.static_rnn(stacked_lstm, features, dtype=dtypes.float32)
        output = dnn_layers(output[-1], [10, 10])

        prediction, loss = tflearn.models.linear_regression(output, targets)

        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
            learning_rate=params["learning_rate"])
        eval_metric_ops = {
            "rmse": tf.metrics.root_mean_squared_error(
                tf.cast(targets, tf.float32), prediction)
        }
        predictions_dict = {"classes": prediction}

        return model_fn_lib.ModelFnOps(
            mode=mode,
            predictions=predictions_dict,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)

        # # batch_size sequences of length 10 with 2 values for each timestep
        #
        # # features = tf.unstack(features, num=params["num_steps"], axis=1)
        # # Create LSTM cell with state size 256. Could also use GRUCell, ...
        # # Note: state_is_tuple=False is deprecated;
        # # the option might be completely removed in the future
        #
        # # multi_cell=tf.contrib.rnn.MultiRNNCell([lstm_cell(256) for _ in range(5)],state_is_tuple=True)
        # features = tf.reshape(features, [-1, 10, 1])
        # cell = tf.contrib.rnn.LSTMCell(256, state_is_tuple=True)
        # outputs, state = tf.nn.dynamic_rnn(cell,
        #                                    features,
        #                                    # sequence_length=[10] * batch_size,
        #                                    dtype=tf.float32)
        #
        # predictions = tf.contrib.layers.fully_connected(state.h,
        #                                                 num_outputs=1,
        #                                                 activation_fn=None)
        # loss = tf.reduce_sum(tf.pow(predictions - targets[-1], 2))
        #
        # # lstm_fw_multicell = tf.contrib.rnn.MultiRNNCell([lstm_forward_cell() for _ in range(params['num_layers_rnn'])],
        # #                                                  state_is_tuple=True)
        # # lstm_bw_multicell = tf.contrib.rnn.MultiRNNCell([lstm_backword_cell() for _ in range(params['num_layers_rnn'])],
        # #                                                  state_is_tuple=True)
        # # features = tf.unstack(features, num=params["num_steps"], axis=1)
        # # with tf.variable_scope("RNN"):
        # #     output, state = tf.contrib.rnn.static_rnn(lstm_fw_multicell, features, dtype=tf.float32)
        # # #     output, state = tf.contrib.learn.models.bidirectional_rnn(lstm_fw_multicell, lstm_bw_multicell, features,
        # # #                                                                dtype='float32')
        # # # # output = dnn_layers(output[-1], [params['dnn_layer_size'], params['dnn_layer_size']])
        # # first_hidden_layer = tf.contrib.layers.fully_connected(output[-1], num_outputs=5, activation_fn=None)
        # # output = tf.contrib.layers.fully_connected(first_hidden_layer, num_outputs=5, activation_fn=None)
        # #
        # # output = self.extract(output, 'input')
        # # labels = self.extract(targets, 'labels')
        # #
        # # W = tf.Variable(tf.random_normal([5, 1]), name="Theta")
        # # lambda_val = tf.constant(0.1)
        # # y_predicted = tf.matmul(output, W, name="y_predicted")
        # #
        # # for pow_i in range(1, 1):
        # #     W = tf.Variable(tf.random_normal([5, 1]), name='weight_%d' % pow_i)
        # #     y_predicted = tf.matmul(tf.pow(output, pow_i), W)+ y_predicted
        # #
        # # with tf.name_scope('cost') as scope:
        # #     # loss = (tf.nn.l2_loss(y_predicted - labels) + lambda_val * tf.nn.l2_loss(W)) / float(self.batch_size)
        # #     # loss_summary = tf.summary.scalar('cost', loss)
        # #     loss = tf.reduce_sum(tf.pow(y_predicted - labels, 2)) / (self.batch_size - 1)
        # #
        # train_op = tf.contrib.layers.optimize_loss(
        #     loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
        #     learning_rate=params["learning_rate"])
        #
        # # correct_prediction = tf.equal(tf.argmax(train_prediction, 1), train_labels)
        #
        # # predictions_dict = {"classes":y_predicted}
        # predictions_dict = {"classes": tf.argmax(input=predictions, axis=1, name="angles")}
        #
        # eval_metric_ops = {
        #     "rmse": tf.metrics.root_mean_squared_error(tf.cast(predictions, tf.float32), tf.cast(targets, tf.float32))
        # }
        #
        # return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions_dict, loss=loss, train_op=train_op
        #                                , eval_metric_ops=eval_metric_ops)


        # layers=[]
        # layers.append(params)
        # stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(5) for _ in range(params['num_layers_rnn'])],
        #                                                 state_is_tuple=True)
        #
        # stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(5) for _ in range(params['num_layers_rnn'])],
        #                                            state_is_tuple=True)
        #
        # x_ = tf.unstack(X, axis=1, num=10)
        # output, layers = tf.contrib.rnn.static_rnn(stacked_lstm, x_, dtype=dtypes.float32)
        # output = dnn_layers(output[-1], [10,10])
        # prediction, loss = tflearn.models.linear_regression(output, y)
        # train_op = tf.contrib.layers.optimize_loss(
        #     loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
        #     learning_rate=params["learning_rate"])
        # eval_metric_ops = {
        #     "rmse": tf.metrics.root_mean_squared_error(
        #         tf.cast(targets, tf.float32), prediction)
        # }
        # predictions_dict = {"classes":prediction}
        #
        # return model_fn_lib.ModelFnOps(
        #     mode=mode,
        #     predictions=predictions_dict,
        #     loss=loss,
        #     train_op=train_op,
        #     eval_metric_ops=eval_metric_ops)

    return _lstm_model(features, targets)

