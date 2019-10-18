"""
The idea is to take some of the previous bars, put it in a neural network, and
predict the next n bars.  It also provides room for inputting other data, such
as orderbook data.
"""

import sys
sys.path.append('../')

import talib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bitfinex_ohlc_import import load_candle_data as lcd

df = lcd.load_data(candle_size='1m')


df_1h = lcd.resample_data(df, '1H')

# get pct changes for each OHLCV targets
# TODO: try normalizing OHL as ratios to close
# TODO: try standardizing targets
# low and high need to be ratios to close in order to work properly
pct_df = df_1h.copy()
pct_df = pct_df.pct_change(1)

# first datapoint will be NA, so drop first row
df_1h = df_1h.iloc[1:]
pct_df = pct_df.iloc[1:]

# fraction of dataset to use for training
train_frac = 0.8
train_idx = int(train_frac * df_1h.shape[0])

# number of historical bars to use for predictions
num_hist_bars = 25
num_predict_bars = 5

train_data = df_1h.iloc[:train_idx]
test_data = df_1h.iloc[train_idx:]

train_pct_df = pct_df.iloc[:train_idx]
test_pct_df = pct_df.iloc[train_idx:]

# get features and targets -- must be a more elegant way to do this
# todo: make into function
train_feats_index = train_pct_df.iloc[::num_hist_bars].index
train_feats_ilocs = [train_pct_df.index.get_loc(t) for t in train_feats_index]

train_feats = []
train_targs = []
raw_train_feats = []
raw_train_targs = []
for i, j in zip(train_feats_ilocs[:-1], train_feats_ilocs[1:]):
    train_feats.append(train_pct_df.iloc[i:j].values.flatten())
    train_targs.append(train_pct_df.iloc[j:j + num_predict_bars].values.flatten())

    raw_train_feats.append(train_data.iloc[i:j].values.flatten())
    raw_train_targs.append(train_data.iloc[j:j + num_predict_bars].values.flatten())

train_feats = np.array(train_feats)
train_targs = np.array(train_targs)
raw_train_feats = np.array(raw_train_feats)
raw_train_targs = np.array(raw_train_targs)

# make model, evaluate performance

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import EarlyStopping


model = Sequential()
model.add(BatchNormalization(input_shape=train_feats[0].shape))
model.add(Dense(128, activation='elu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dense(64, activation='elu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dense(32, activation='elu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dense(true_train_targs[0].shape[0], activation='elu', kernel_initializer='glorot_normal'))

es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_absolute_error', optimizer='sgd', metrics=['mse', 'mae'])

history = model.fit(train_feats, train_targs, epochs=1000, validation_split=0.15, callbacks=[es])

# todo: interactive plotting, plotly live dashboard

"""
Notes: noticed when using raw feats and targs, it seems to predict the average
when using feats and raw targs, doesn't seem to do well
With pct feats and targs, seems to do much better
"""
predictions = model.predict(train_feats)

# todo: incorporate with live data, plot live with dashboard