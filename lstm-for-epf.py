import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lstm_predictor import generate_data, load_csvdata, lstm_model, sine_model
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


LOG_DIR = './ops_logs'
TIMESTEPS = 10
RNN_LAYERS = [{'steps': TIMESTEPS}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 100000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100
LEARNING_RATE = 0.004

dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M')
# rawdata = pd.read_csv("/home/runge/openbci/git/OpenBCI_Python/visualization/TensorFlow-Tutorials-for-Time-Series/RealMarketPriceDataPT.csv",
#                    parse_dates={'timeline': ['date', '(UTC)']},
#                    index_col='timeline', date_parser=dateparse)

model_params = {"learning_rate": LEARNING_RATE}

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def load_data(filename, seq_len, normalise_window):
	f = open(filename, 'rb').read()
	data = f.decode().split('\n')

	sequence_length = seq_len + 1
	result = []
	for index in range(len(data) - sequence_length):
		result.append(data[index: index + sequence_length])

	if normalise_window:
		result = normalise_windows(result)

	result = np.array(result, dtype=np.float32)

	row = round(0.9 * result.shape[0])
	train = result[:int(row), :]
	np.random.shuffle(train)
	x_train = train[:, :-1]
	y_train = train[:, -1]
	x_test = result[int(row):, :-1]
	y_test = result[int(row):, -1]

	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

	return [x_train, y_train, x_test, y_test]

# X, y = load_csvdata(rawdata, TIMESTEPS, seperate=False)
x_train, y_train, x_test, y_test = load_data('sp500.csv', 10, True)
regressor = learn.Estimator(model_fn=sine_model,params = model_params)

# n_classes=0,
# verbose=1,
# steps=TRAINING_STEPS,
# optimizer='Adagrad',
# learning_rate=0.03,
# batch_size=BATCH_SIZE

validation_monitor = learn.monitors.ValidationMonitor(x_test, y_test,
                                                      every_n_steps=PRINT_STEPS,
                                                      early_stopping_rounds=1000)

# regressor.fit(X['train'], y['train'], monitors=[validation_monitor], logdir=LOG_DIR)
regressor.fit(x=x_train, y=y_train, steps=1000)


test_results = regressor.evaluate(x=x_test, y=y_test, steps=1)
print("Loss: %s" % test_results["loss"])
print("Root Mean Squared Error: %s" % test_results["rmse"])

predictions = regressor.predict(x=x_test, as_iterable=True)

predictions_rsl = []

for i,p in enumerate(predictions):
    print("Prediction %s: %s" % (y_test[i], p["classes"]))
    predictions_rsl.append(p["classes"])
    # plot_predicted, = plt.plot(p["classes"], label='predicted')
    # plot_test, = plt.plot(y['test'], label='test')
    # plt.legend(handles=[plot_predicted])
#
plot_results(predictions_rsl, y_test)


