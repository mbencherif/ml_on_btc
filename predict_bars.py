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
# or some other way to ensure low is below OC and high is above OC
pct_df = df_1h.copy()
pct_df = pct_df.pct_change(1)

# first datapoint will be NA, so drop first row
df_1h = df_1h.iloc[1:]
pct_df = pct_df.iloc[1:]

# number of historical bars to use for predictions
num_hist_bars = 25
num_predict_bars = 5

# get features and targets -- must be a more elegant way to do this
# todo: make into function
# train_feats_index = train_pct_df.iloc[::num_hist_bars].index
# train_feats_ilocs = [train_pct_df.index.get_loc(t) for t in train_feats_index]

from tqdm import tqdm

feats = []
targs = []
raw_feats = []
raw_targs = []
for i in tqdm(range(num_hist_bars, df_1h.shape[0] - num_hist_bars - num_predict_bars)):
    feats.append(pct_df.iloc[i - num_hist_bars:i].values.flatten())
    targs.append(pct_df.iloc[i:i + num_predict_bars].values.flatten())

    raw_feats.append(df_1h.iloc[i - num_hist_bars:i].values.flatten())
    raw_targs.append(df_1h.iloc[i:i + num_predict_bars].values.flatten())

feats = np.array(feats)
targs = np.array(targs)
raw_feats = np.array(raw_feats)
raw_targs = np.array(raw_targs)

# fraction of dataset to use for training
train_frac = 0.8
train_idx = int(train_frac * df_1h.shape[0])

train_feats = feats[:train_idx]
train_targs = targs[:train_idx]

test_feats = feats[train_idx:]
test_targs = targs[train_idx:]

# make model, evaluate performance
import datetime

import tensorflow as tf
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard


model = Sequential()
model.add(BatchNormalization(input_shape=train_feats[0].shape))
model.add(Dense(128, activation='elu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dense(64, activation='elu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dense(32, activation='elu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dense(targs[0].shape[0], activation='elu', kernel_initializer='glorot_normal'))

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = TensorBoard(log_dir=log_dir)

es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
sgd = optimizers.SGD(lr=0.5, momentum=0.9, nesterov=True) # decay=1e-6, 
model.compile(loss='mean_absolute_error', optimizer='sgd', metrics=['mse'])

history = model.fit(train_feats, train_targs, epochs=1000, validation_split=0.15, callbacks=[es, tb])

print(model.evaluate(train_feats, train_targs))
print(model.evaluate(test_feats, test_targs))

# todo: interactive plotting, plotly live dashboard

"""
Notes: noticed when using raw feats and targs, it seems to predict the average
when using feats and raw targs, doesn't seem to do well
With pct feats and targs, seems to do much better
"""
train_preds = model.predict(train_feats)

# todo: back-calculate actual values from predictions
# test re-calc on raw_targs

rs_targs = targs.reshape(-1, num_predict_bars, num_predict_bars)
first_point = df_1h.iloc[num_hist_bars - 1]
first_point * (1 + rs_targs[0][0])

preds = [first_point * (1 + rs_targs[0][0])]
for i in range(1, num_predict_bars):
    preds.append(preds[-1] * (1 + rs_targs[0][i]))

pr = pd.concat(preds, axis=1).T


def ensure_low_high(df):
    """
    some prediction methods can result in lows and highs above/below
    the open/close.  This fixes that
    params:
    df: pd.DataFrame, should have columns ['open', 'high', 'low', 'close'] at least
    """
    df['low'] = min(df['low'], df['open'], df['close'])
    df['high'] = max(df['high'], df['open'], df['close'])
    return df


def back_calc_one_point_pct(first_point, pct_targs, timeunit='1h'):
    """
    calculates OHLCV from the first OHLCV point and 
    pct change values for num_predict_bars

    params:
    first_point: pd.Series; the first raw OHLCV point
    pct_targs: np.array; pct change for the columns in first_point
    """
    first_pr = first_point * (1 + pct_targs[0])
    first_pr = ensure_low_high(first_pr)
    preds = [first_pr]
    # it's a pd.Series, so the datetime is in the name, not index
    index = [first_pr.name]

    for i in range(1, pct_targs.shape[-1]):
        prd = preds[-1] * (1 + pct_targs[i])
        prd = ensure_low_high(prd)
        preds.append(prd)
        index.append(index[-1] + pd.Timedelta(timeunit))

    pr = pd.concat(preds, axis=1).T
    pr.index = index
    return pr

# backcalculate all values from targets
all_preds = []
for i in tqdm(range(rs_targs.shape[0])):
    point = df_1h.iloc[i + num_hist_bars - 1]
    pr = back_calc_one_point_pct(point, rs_targs[i])
    all_preds.append(pr)

# test that calculations are working as expected
for i in tqdm(range(len(all_preds))):
    init_loc = i + num_hist_bars
    if not np.allclose(all_preds[i], df_1h.iloc[init_loc:init_loc + 5]):
        print('uhoh')
        break

rs_train_preds = train_preds.reshape(-1, num_predict_bars, 5)

all_preds = []
for i in tqdm(range(rs_train_preds.shape[0])):
    point = df_1h.iloc[i + num_hist_bars - 1]
    pr = back_calc_one_point_pct(point, rs_train_preds[i])
    all_preds.append(pr)
    break


import plotly.graph_objects as go
# may need to do this to make sure it shows up in browser
# https://plot.ly/python/renderers/
# import plotly.io as pio
# pio.renderers.default = "browser"

# https://plot.ly/~jackp/17421.embed
INCREASING_COLOR = '#17BECF'
DECREASING_COLOR = '#7F7F7F'
colors = []

for i in range(df1.shape[0]):
    if i != 0:
        if df1['close'][i] > df1['close'][i-1]:
            colors.append(INCREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)
    else:
        colors.append(DECREASING_COLOR)


df1 = all_preds[0]
fig = go.Figure(data=go.Candlestick(x=df1.index,
                                    open=df1['open'],
                                    high=df1['high'],
                                    low=df1['low'],
                                    close=df1['close']))

fig.show()

data = [ dict(
    type = 'candlestick',
    open = df['open'],
    high = df['high'],
    low = df['low'],
    close = df['close'],
    x = df.index,
    yaxis = 'y2',
    name = 'GS',
    increasing = dict( line = dict( color = INCREASING_COLOR ) ),
    decreasing = dict( line = dict( color = DECREASING_COLOR ) ),
) ]

layout=dict()

fig = dict( data=data, layout=layout )

fig['layout'] = dict()
fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
fig['layout']['xaxis'] = dict( rangeselector = dict( visible = True ) )
fig['layout']['yaxis'] = dict( domain = [0, 0.2], showticklabels = False )
fig['layout']['yaxis2'] = dict( domain = [0.2, 0.8] )
fig['layout']['legend'] = dict( orientation = 'h', y=0.9, x=0.3, yanchor='bottom' )
fig['layout']['margin'] = dict( t=40, b=40, r=40, l=40 )

rangeselector=dict(
    visibe = True,
    x = 0, y = 0.9,
    bgcolor = 'rgba(150, 200, 250, 0.4)',
    font = dict( size = 13 ),
    buttons=list([
        dict(count=1,
             label='reset',
             step='all'),
        dict(count=1,
             label='1yr',
             step='year',
             stepmode='backward'),
        dict(count=3,
            label='3 mo',
            step='month',
            stepmode='backward'),
        dict(count=1,
            label='1 mo',
            step='month',
            stepmode='backward'),
        dict(step='all')
    ]))
    
fig['layout']['xaxis']['rangeselector'] = rangeselector

fig['data'].append( dict( x=df.index, y=df1['volmue'],                         
                         marker=dict( color=colors ),
                         type='bar', yaxis='y', name='Volume' ) )

fig.show()

import cufflinks as cf

df = cf.datagen.ohlc()
qf = cf.QuantFig(all_preds[0])

# todo: incorporate with live data, plot live with dashboard