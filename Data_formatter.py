import csv
from operator import index 
from pathlib import Path
from tkinter.ttk import Separator
from nbformat import read
import pandas as pd
import warnings
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
#from PatternDetectionInCandleStick.LabelPatterns import label_candles




DATASET_NAME = 'LUNA-USDT'
DATASET_FOLDER = r'LUNA-USDT'
FILE = r'LUNA-USDT.csv'
#format = Dataformatter(DATASET_NAME)

def seperator(filename,new_sep):
    data = pd.read_csv(filename)
    reader = list(csv.reader(open(data, "rU"), delimiter=';'))
    writer = csv.writer(open(data, 'w'), delimiter=new_sep)
    writer.writerows(row for row in reader)
    print (writer)

#seperator(FILE,',')

import csv


"""reader = reader.replace(';',',')
    reader.columns=reader.columns.str.replace('time','Date')
    for i in reader.columns:
        print(i)
    reader.to_csv('/Users/dekahalane/Desktop/DQN-Trading-Master/Data/LUNA-USDT/LUNA-USDT.csv')
    #newcontents.to_csv(path_or_buf='/Users/dekahalane/Desktop/DQN-Trading-Master/Data/LUNA-USDT/LUNA-USDT.csv')
    """
with open('/Users/dekahalane/Desktop/DQN-Trading-Master/Data/LUNA-USDT/LUNA-USDT.csv','r',encoding='utf-8') as file:
    reader = pd.read_csv(file, delimiter=';')
    reader = reader.replace(';',',')
    #reader.swaplevel(6,0)
   #first=reader.columns[0][:5] #gir output: trash
    #print(reader)
    #for i in reader:
        #data.iloc[i] = data[i].delimeter
    new = reader["time"].str.split("T", n = 1, expand = True)
    reader["Date"] = new[0]
    
    #ny.reset_index()
    reader.drop(columns = ['time'], inplace = True)
  
    #reader = reader.drop(reader.columns[[0][:5]], axis=1)
    print(reader)
    reader.to_csv('/Users/dekahalane/Desktop/DQN-Trading-Master/Data/LUNA-USDT/LUNA-USDT.csv', mode='w+',index=False)
 
    #for i in reader.columns[0]:
        #print(i)


fig = go.Figure(data=[go.Candlestick(x=reader['Date'],
                open=reader['open'], high=reader['high'],
                low=reader['low'], close=reader['close'])
                     ])

fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()