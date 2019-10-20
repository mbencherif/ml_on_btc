"""
The idea is to take some of the previous bars, put it in a neural network, and
predict the next n bars.  It also provides room for inputting other data, such
as orderbook data.
"""
import datetime
import sys
sys.path.append('../')

import talib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard

from bitfinex_ohlc_import import load_candle_data as lcd

def create_simple_model(feats, targs):
    model = Sequential()
    model.add(BatchNormalization(input_shape=feats[0].shape))
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

    return model



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


def test_backcalc(df, targs, num_hist_bars, num_predict_bars):
    """
    checks to make sure backcalculation of targets is working properly

    df: pd.DataFrame of original data
    targs: np.ndarray of target values, should be percent changes
    """
    rs_targs = targs.reshape(-1, num_predict_bars, num_predict_bars)
    first_point = df.iloc[num_hist_bars - 1]
    first_point * (1 + rs_targs[0][0])

    preds = [first_point * (1 + rs_targs[0][0])]
    for i in range(1, num_predict_bars):
        preds.append(preds[-1] * (1 + rs_targs[0][i]))

    pr = pd.concat(preds, axis=1).T
    # backcalculate all values from targets
    all_preds = []
    for i in tqdm(range(rs_targs.shape[0])):
        point = df.iloc[i + num_hist_bars - 1]
        pr = back_calc_one_point_pct(point, rs_targs[i])
        all_preds.append(pr)

    # test that calculations are working as expected
    for i in tqdm(range(len(all_preds))):
        init_loc = i + num_hist_bars
        if not np.allclose(all_preds[i], df_1h.iloc[init_loc:init_loc + 5]):
            print('uhoh, backcalculation didn\'t match actual values')
            break


def backcalculate_predictions(df, predictions, num_hist_bars, num_predict_bars):
    """
    takes percent change predictions and calculates actual values
    df: pd.DataFrame, original data
    predictions: np.ndarray, percent change predictions
    """
    rs_preds = predictions.reshape(-1, num_predict_bars, 5)

    all_preds = []
    for i in tqdm(range(rs_preds.shape[0])):
        point = df.iloc[i + num_hist_bars - 1]
        pr = back_calc_one_point_pct(point, rs_preds[i])
        all_preds.append(pr)

    return all_preds


def create_feats_targs_pct(df, num_hist_bars=25, num_predict_bars=5):
    """
    Creates features and targets from raw data.
    Uses percent change for features and targets.
    num_hist_bars: number of historical bars to use for predictions
    num_predict_bars: number of bars in the future to predict
    """
    # get pct changes for each OHLCV targets
    # TODO: try normalizing OHL as ratios to close
    # TODO: try standardizing targets
    # low and high need to be ratios to close in order to work properly
    # or some other way to ensure low is below OC and high is above OC
    pct_df = df.copy()
    pct_df = pct_df.pct_change(1)

    # first datapoint will be NA, so drop first row
    new_df = df.iloc[1:].copy()
    pct_df = pct_df.iloc[1:]

    # get features and targets -- must be a more elegant way to do this
    feats = []
    targs = []
    raw_feats = []
    raw_targs = []
    for i in tqdm(range(num_hist_bars, pct_df.shape[0] - num_hist_bars - num_predict_bars)):
        feats.append(pct_df.iloc[i - num_hist_bars:i].values.flatten())
        targs.append(pct_df.iloc[i:i + num_predict_bars].values.flatten())

        raw_feats.append(new_df.iloc[i - num_hist_bars:i].values.flatten())
        raw_targs.append(new_df.iloc[i:i + num_predict_bars].values.flatten())

    feats = np.array(feats)
    targs = np.array(targs)
    raw_feats = np.array(raw_feats)
    raw_targs = np.array(raw_targs)

    return feats, targs


def create_train_test(feats, targs, train_frac=0.8)
    """
    train_frac: float between 0 and 1, fraction of dataset to use for training
    """
    train_idx = int(train_frac * feats.shape[0])

    train_feats = feats[:train_idx]
    train_targs = targs[:train_idx]

    test_feats = feats[train_idx:]
    test_targs = targs[train_idx:]

    return train_feats, train_targs, test_feats, test_targs


# load data
df = lcd.load_data(candle_size='1m')
df_1h = lcd.resample_data(df, '1H')

feats, targs = create_feats_targs_pct(df_1h)

# make model, evaluate performance
model = create_simple_model(feats, targs)
history = model.fit(train_feats, train_targs, epochs=1000, validation_split=0.15, callbacks=[es, tb])

print(model.evaluate(train_feats, train_targs))
print(model.evaluate(test_feats, test_targs))

"""
Notes: noticed when using raw feats and targs, it seems to predict the average
when using feats and raw targs, doesn't seem to do well
With pct feats and targs, seems to do much better
"""


train_preds = model.predict(train_feats)

test_backcalc(df_1h, targs, num_hist_bars, num_predict_bars)

true_preds = backcalculate_predictions(df_1h, train_preds, num_hist_bars, num_predict_bars)
