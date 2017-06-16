
# coding: utf-8

# In[13]:

# get_ipython().magic('matplotlib inline')
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lstm_predictor import generate_data, lstm_model, sine_model
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

LOG_DIR = './ops_logs'
TIMESTEPS = 10
RNN_LAYERS = [{'steps': TIMESTEPS, "learning_rate": 0.9, "keep_prob": 0.8, 'num_units': 5}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 100
BATCH_SIZE = 10
PRINT_STEPS = TRAINING_STEPS / 10

model_params = {"learning_rate": 0.004}
regressor = learn.Estimator(model_fn=sine_model, params = model_params)

X, y = generate_data(np.sin, np.linspace(0, 100, 1000), TIMESTEPS, seperate=False)
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                      every_n_steps=1,
                                                      early_stopping_rounds=1000)

tensors_to_log = {"classes": "angles"}
# logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)

regressor.fit(X['train'], y['train'], monitors=[validation_monitor])
predicted = regressor.predict(X['test'])
mse = mean_squared_error(y['test'], predicted)
print ("Error: %f" % mse)

plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(y['test'], label='test')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()
