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

# fraction of dataset to use for training
train_frac = 0.8
train_idx = int(train_frac * df_1h.shape[0])

# number of historical bars to use for predictions
num_hist_bars = 25
num_predict_bars = 5

train_data = df_1h.iloc[:train_idx]
test_data = df_1h.iloc[train_idx:]

# get features and targets -- must be a more elegant way to do this
# todo: make into function
train_feats_index = train_data.iloc[::num_hist_bars].index
train_feats_ilocs = [train_data.index.get_loc(t) for t in train_feats_index]

train_feats = []
train_targs = []
for i, j in zip(train_feats_ilocs[:-1], train_feats_ilocs[1:]):
    train_feats.append(train_data.iloc[i:j])
    train_targs.append(train_data.iloc[j:j+num_predict_bars])

# todo: make model, evaluate performance

# todo: interactive plotting, plotly live dashboard

# todo: incorporate with live data, plot live with dashboard