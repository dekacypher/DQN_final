from concurrent.futures import ProcessPoolExecutor
from pydoc import ErrorDuringImport
import warnings
import pandas as pd
import pickle
import sys
sys.path.insert(0,'/content/DQN/')
from PatternDetectionInCandleStick.LabelPatterns import label_candles
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import numpy as np
import ast
from pathlib import Path
import datetime
import requests

class YahooFinanceDataLoader:
    """ Dataset form GOOGLE"""
    def __init__(self, dataset_folder, file_name,interval, split_point, begin_date=None, end_date=None, load_from_file=False,load_from_binance=True):
        """
        :param dataset_folder
            folder name in './Data' directory
        :param file_name
            csv file name in the Data directory
        :param load_from_file
            if False, it would load and process the data from the beginning
            and save it again in a file named 'data_processed.csv'
            else, it has already processed the data and saved in 'data_processed.csv', so it can load
            from file. If you have changed the original .csv file in the Data directory, you should set it to False
            so that it will rerun the preprocessing process on the new data.
        :param begin_date
            This is the beginning date in the .csv file that you want to consider for the whole train and test
            processes
        :param end_date
            This is the end date in the .csv file of the original data to to consider for the whole train and test
            processes
        :param split_point
            The point (date) between begin_date and end_date that you want to split the train and test sets.
        """
        print('Loading data for ',dataset_folder)
        warnings.filterwarnings('ignore')
        self.DATA_NAME = dataset_folder
        self.DATA_PATH = os.path.join(Path(os.path.abspath(os.path.dirname(__file__))).parent,
                                      f'Data/{dataset_folder}') + '/'
        self.OBJECT_PATH = os.path.join(Path(os.path.abspath(os.path.dirname(__file__))).parent, 'Objects') + '/'
        self.DATA_FILE = file_name
        date = datetime.datetime.strptime(split_point, "%Y-%m-%d")
        self.split_point = int(date.strftime("%Y%m%d"))
        self.interval = interval
        self.begin_date = begin_date
        self.end_date = end_date
        self.load_from_binance = load_from_binance
        self.file_path = os.path.join(f'{self.DATA_PATH}{self.interval}', f'{self.interval}.csv')
        if not os.path.exists(f'{self.DATA_PATH}{self.interval}'):
            os.makedirs(os.path.join(self.DATA_PATH,self.interval))
        f = open(self.file_path, 'a')
        
        if not load_from_file:
            self.data, self.patterns = self.load_data()
            self.save_pattern()
            self.normalize_data()
            if(self.load_from_binance==False):
                self.data.to_csv(f'{self.DATA_PATH}data_processed.csv', index=True)
            if begin_date is not None:
                self.data = self.data[self.data.index >= begin_date]
            if end_date is not None:
                self.data = self.data[self.data.index <= end_date]
            if type(split_point) == str:
                print("INDEX: ")
                print (self.data.index)
                print('\nTYPE: ')
                print((type(self.data.index)))
                print("INDEX: ")
                print(self.split_point)
                print('\nTYPE: ')
                print(type(self.split_point))
                self.data_train = self.data[self.data.index < self.split_point]
                self.data_test = self.data[self.data.index >= self.split_point]
            elif type(split_point) == int:
                self.data_train = self.data[:self.split_point]
                self.data_test = self.data[self.split_point:]
            else:
                raise ValueError('Split point should be either int or date!')
            self.data_train_with_date = self.data_train.copy()
            self.data_test_with_date = self.data_test.copy()
            self.data_train.reset_index(drop=True, inplace=True)
            self.data_test.reset_index(drop=True, inplace=True)
            # self.data.reset_index(drop=True, inplace=True)
        else:
            self.data = pd.read_csv(f'{self.DATA_PATH}data_processed.csv')
            self.data.set_index('Time', inplace=True)
            labels = list(self.data.label)
            labels = [ast.literal_eval(l) for l in labels]
            self.data['label'] = labels
            self.load_pattern()
            self.normalize_data()
            if begin_date is not None:
                self.data = self.data[self.data.index >= begin_date]
            if end_date is not None:
                self.data = self.data[self.data.index <= end_date]
            if type(split_point) == str:
                self.data_train = self.data[self.data.index < split_point]
                self.data_test = self.data[self.data.index >= split_point]
            elif type(split_point) == int:
                self.data_train = self.data[:split_point]
                self.data_test = self.data[split_point:]
            else:
                raise ValueError('Split point should be either int or date!')
            self.data_train_with_date = self.data_train.copy()
            self.data_test_with_date = self.data_test.copy()
            self.data_train.reset_index(drop=True, inplace=True)
            self.data_test.reset_index(drop=True, inplace=True)
            self.plot_data()
            # self.data.reset_index(drop=True, inplace=True)
    def ms_to_dt_utc(self,ms: int) -> datetime:
        return datetime.datetime.fromtimestamp(ms)

    def get_klines_iter(self,symbol, interval, start, end, limit=1000):
        url = 'https://api.kucoin.com/api/v1/market/candles?type=8hour&symbol=ASD-USDT&startAt=1577833200&endAt=1657983710'
        startDate = 1577833200
        end = 1657983710  
        df = pd.DataFrame()
        while startDate <= end:
            #time.sleep(200)
                data = requests.get(url)
                df = pd.json_normalize(data.json())
                df2 = pd.DataFrame(df['data'].iloc[0])

                columns_names = ["Time", "Open", "Close", "High", "Low", "Volume", "Transaction Amount"]
                df2.columns = columns_names

                df2["Time"] = df2["Time"].apply(lambda x: self.ms_to_dt_utc(int(x)))
                #df2.reset_index(drop=True, inplace=True)   
                df2[['Open','Close','High','Low', 'Volume']] = df2[['Open','Close','High','Low', 'Volume']].apply(pd.to_numeric)
                df2.set_index('Time')

                return df2

        # url = 'https://api.kucoin.com/api/v1/market/candles?type=' + interval + '&symbol=' + \
        #         symbol + '&startAt=' + str(startDate) + '&endAt=' + str(end)
        # df = pd.DataFrame()
        # startDate = start

        # while startDate <= end:
      
        #         data = requests.get(url)
        #         df = pd.json_normalize(data.json())
        #         df2 = pd.DataFrame(df['data'].iloc[0])

        #         columns_names = ["Time", "Open", "Close", "High", "Low", "Volume", "Transaction Amount"]
        #         df2.columns = columns_names

        #         df2["Time"] = df2["Time"].apply(lambda x: self.ms_to_dt_utc(int(x)))
        #         df2.reset_index(drop=True, inplace=True)   
        #         #print(df2)
            
        #         return df2
            
       
     
    def get_last_date(self) -> datetime.datetime:
        with open(self.file_path, 'r') as file:
            for line in file:
                line = line.strip().split(",")
                if "-" in line[0]:
                    date = datetime.datetime.strptime(line[0], "%Y-%m-%d %H:%M:%S")
                    return date
        file.close()

    def load_data(self):
        """
        This function is used to read and clean data from .csv file.
        @return:
        """
        if self.load_from_binance :
          if os.stat(self.file_path).st_size == 0:
              datefrom = datetime.datetime(2020, 1, 1 , 0, 0)
          else:
              try:
                  datefrom = self.get_last_date()
              except:
                  raise ValueError('date is not of type datetime.datetime')
            #datefrom = datetime.datetime(2020, 1, 1, 0, 0)
          timestamp_datefrom = int(round(datefrom.timestamp()))
          dateto = datetime.datetime.now()
          timestamp_dateto = int(round(dateto.timestamp()))
          data = self.get_klines_iter(self.DATA_NAME,self.interval,timestamp_datefrom ,timestamp_dateto)
          print(data)
          data.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume':'volume'}, inplace=True)
          data = data.drop(['Transaction Amount'], axis=1)
          data['mean_candle'] = data.close
          patterns = label_candles(data)
          print('Loaded data',data)
          #data.set_index('Time', inplace=True)
          data.to_csv(self.file_path,index=False)
          return data, list(patterns.keys())
        else:
            data = pd.read_csv(self.file_path, index=True)
            data.dropna(inplace=True)
            #data.set_index('Time', inplace=True)
            data.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume':'volume'}, inplace=True)
            data = data.drop(['Transaction Amount'], axis=1)
            data['mean_candle'] = data.close
            patterns = label_candles(data)
            return data, list(patterns.keys())

    def plot_data(self):
        """
        This function is used to plot the dataset (train and test in different colors).
        @return:
        """
        sns.set(rc={'figure.figsize': (9, 5)})
        df1 = pd.Series(self.data_train_with_date.close, index=self.data.index)
        df2 = pd.Series(self.data_test_with_date.close, index=self.data.index)
        ax = df1.plot(color='b', label='Train')
        df2.plot(ax=ax, color='r', label='Test')
        ax.set(xlabel='Time', ylabel='Close Price')
        ax.set_title(f'Train and Test sections of dataset {self.DATA_NAME}')
        plt.legend()
        plt.savefig(f'{Path(self.DATA_PATH).parent}/DatasetImages/{self.DATA_NAME}.jpg', dpi=300)
    def save_pattern(self):
        with open(
                f'{self.OBJECT_PATH}pattern.pkl', 'wb') as output:
            pickle.dump(self.patterns, output, pickle.HIGHEST_PROTOCOL)
    def load_pattern(self):
        with open(self.OBJECT_PATH + 'pattern.pkl', 'rb') as input:
            self.patterns = pickle.load(input)

    def normalize_data(self):
        """
        This function normalizes the input data
        @return:
        """
        min_max_scaler = MinMaxScaler()
        self.data['open_norm'] = min_max_scaler.fit_transform(self.data.open.values.reshape(-1, 1))
        self.data['high_norm'] = min_max_scaler.fit_transform(self.data.high.values.reshape(-1, 1))
        self.data['low_norm'] = min_max_scaler.fit_transform(self.data.low.values.reshape(-1, 1))
        self.data['close_norm'] = min_max_scaler.fit_transform(self.data.close.values.reshape(-1, 1))