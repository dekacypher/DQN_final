#Heatmap 
import pandas as pd

from binance.client import Client 
import numpy as np 
 
client = Client()



info = client.get_exchange_info()

symbols = [x['symbol'] for x in info['symbols']]
#symbols

relevant = [symbol for symbol in symbols if symbol.endswith('BTC')]
relevant

# for symbol in symbols:
#     if symbol.endswith('BTC'):
#         print(symbol)

def getdailydata(symbol):
    frame = pd.DataFrame(client.get_historical_klines(symbol, '1d', '30 day ago UTC'))

    if len(frame) > 0:
        frame = frame.iloc[:,:6]
        frame.columns = ['Time', 'Open', 'High', 'Low', 'Close','Volume']
        frame = frame.set_index('Time')
        frame.index = pd.to_datetime(frame.index, unit='ms')
        frame = frame.astype(float)
        return frame

    
    #getdailydata('BTCUSDT')

dfs = list()

for coin in relevant:
    dfs.append(getdailydata(coin))

dfs