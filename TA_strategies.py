"""
rules:

buy when macdhist changes to positive and MACD is positive
close out buy when macd turns negative

sell when macdhist changes to negative and MACD is negative
close out sell when macd turns positive

"""


import sys
sys.path.append('../')

import talib
import numpy as np
import matplotlib.pyplot as plt

from bitfinex_ohlc_import import load_candle_data as lcd

df = lcd.load_data(candle_size='1m')

macd, macdsignal, macdhist = talib.MACD(df.close.values, fastperiod=30, slowperiod=90, signalperiod=9)

df['macd'] = macd
df['macdsignal'] = macdsignal
df['macdhist'] = macdhist

df.dropna(inplace=True)






fig, ax = plt.subplots(2, 1, sharex=True)

df['close'].plot(ax=ax[0])
df[['macd', 'macdsignal', 'macdhist']].plot(ax=ax[1])
plt.show()


# get future price
df['close_pct'] = df['close'].pct_change(60)
df['1h_close_pct'] = df['close_pct'].shift(-60)

asign = np.sign(df['macdhist'])
signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
signchange[0] = 0
signchange = signchange * asign

df['macdhist_sign_change'] = signchange
# annyingly sets signs of zeros
df.loc[df['macdhist_sign_change'] == 0, 'macdhist_sign_change'] = 0
df['buy_signals'] = 0
df['sell_signals'] = 0
df.loc[(df['macdhist_sign_change'] > 0) & (df['macd'] > 0), 'buy_signals'] = 1
df.loc[(df['macdhist_sign_change'] < 0) & (df['macd'] < 0), 'sell_signals'] = 1

# get first buy/sell signal
first_buy = df[df['buy_signals'] != 0].index[0]
first_sell = df[df['sell_signals'] != 0].index[0]

buy_first = True
if first_buy > first_sell:
    buy_first = False

# close out positions for analysis
for i, r in df.iterrows():
    
