import logging

# Importing the Libraries
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.strategy import IStrategy

import talib.abstract as ta

logger = logging.getLogger(__name__)

logger.info('Starting CryptoPrediction Strategy')


def _load_data(config, pair, timeframe, timerange, window_size):
    timerange = TimeRange.parse_timerange(timerange)
    logger.info('Loading data for pair %s and timeframe %s', pair, timeframe)

    return history.load_data(
        datadir=config['datadir'],
        pairs=[pair],
        timeframe=timeframe,
        timerange=timerange,
        startup_candles=window_size + 1,
        fail_without_data=True,
        data_format=config.get('dataformat_ohlcv', 'json'),
    )


class CryptoPredictionTraining(IStrategy):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

        self.pair = 'BTC/USDT'
        self.timeframe = '1h'
        self.timerange = config['timerange']
        self.window_size = 200
        self.data = _load_data(config, self.pair, self.timeframe, self.timerange, self.window_size)
        self.data = self.data[self.pair]
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        output_var = 'close'
        features = ['open', 'high', 'low', 'volume', 'sar']

        self.data['sar'] = ta.SAR(self.data['high'], self.data['low'], acceleration=0.02, maximum=0.2)
        # fill nan values with 0
        self.data['sar'].fillna(0, inplace=True)

        logger.info('Training data shape: %s', self.data.shape)

        # Create the training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[features],
                                                                                self.data[output_var], test_size=0.2,
                                                                                shuffle=False)

        # data processing for lstm
        self.X_train = self.X_train.values
        self.X_test = self.X_test.values
        self.y_train = self.y_train.values
        self.y_test = self.y_test.values
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)

        # convert to 2d array
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1])
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1])
        # self.y_train = self.y_train.reshape(self.y_train.shape[0], self.y_train.shape[1])
        # self.y_test = self.y_test.reshape(self.y_test.shape[0], self.y_test.shape[1])

        # Normalize the data
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.y_train = self.scaler.fit_transform(self.y_train.reshape(-1, 1))
        self.y_test = self.scaler.transform(self.y_test.reshape(-1, 1))
        self.y_train = self.y_train.reshape(-1)
        self.y_test = self.y_test.reshape(-1)

        # Create and fit the LSTM network
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        self.model.add(LSTM(units=50))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=2, verbose=2)

        # Save the model
        self.model.save('model.h5')
        self.model.summary()

        # Plot the model
        # plot_model(self.model, to_file='model.png')

        # Load the model
        self.model = load_model('model.h5')

        # Make predictions
        self.predictions = self.model.predict(self.X_test)
        self.predictions = self.scaler.inverse_transform(self.predictions)
        self.y_test = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

        # plot actual and predicted values
        plt.plot(self.y_test, color='red', label='Actual')
        plt.plot(self.predictions, color='blue', label='Predicted')
        plt.title('BTC/USDT')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        plt.savefig('prediction.png')

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, 'buy'] = 0
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, 'sell'] = 0
        return dataframe
