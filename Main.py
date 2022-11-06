# Importing DataLoaders for each model. These models include rule-based, vanilla DQN and encoder-decoder DQN.
from DataLoader_F.DataLoader import YahooFinanceDataLoader
from DataLoader_F.DataForPatternBasedAgent import DataForPatternBasedAgent
from DataLoader_F.DataAutoPatternExtractionAgent import DataAutoPatternExtractionAgent
from DataLoader_F.DataSequential import DataSequential

from DeepRLAgent.MLPEncoder.Train import Train as SimpleMLP
from DeepRLAgent.SimpleCNNEncoder.Train import Train as SimpleCNN
from EncoderDecoderAgent.GRU.Train import Train as GRU
from EncoderDecoderAgent.CNN.Train import Train as CNN
from EncoderDecoderAgent.CNN2D.Train import Train as CNN2d
from EncoderDecoderAgent.CNNAttn.Train import Train as CNN_ATTN
from EncoderDecoderAgent.CNN_GRU.Train import Train as CNN_GRU
from Sensitivity import SensitivityRun
# Imports for Deep RL Agent
from DeepRLAgent.VanillaInput.Train import Train as DeepRL

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import argparse
from tqdm import tqdm
import os
#from utils import save_pkl, load_pkl

def choose_model():
  list = ['1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d','3d','1w','1M']
  for value in list:
    print(value)
  print("Choose only one interval of the following formatted values:\n")
  interval = input("Interval: ")
  
  if (interval not in list):
    print("Incorrect input {} interval doesn't exist! ".format(interval))
    
  else:
    print("You have chosen the interval {}.".format(interval))

    return interval 

interval = choose_model()

parser = argparse.ArgumentParser(description='DQN-Trader arguments')
parser.add_argument('--dataset-name', default="KDAUSDT",
                    help='Name of the data inside the Data folder')
parser.add_argument('--interval', default=interval, 
                    help='Number of interval for plotting chart')
parser.add_argument('--nep', type=int, default=30,
                    help='Number of episodes')
parser.add_argument('--window_size', type=int, default=3,
                    help='Window size for sequential models')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()


if __name__ == '__main__':
    gamma_list = [0.9, 0.8, 0.7]
    batch_size_list = [8, 32, 64]
    replay_memory_size_list = [16, 64, 256]
    n_step = 8
    window_size = args.window_size
    dataset_name = args.dataset_name
    interval = args.interval
    n_episodes = args.nep
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    feature_size = 64
    target_update = 5

    gamma_default = 0.9
    batch_size_default = 16
    replay_memory_size_default = 32

    pbar = tqdm(len(gamma_list) + len(replay_memory_size_list) + len(batch_size_list))

    # test gamma

    run = SensitivityRun(
        dataset_name,
        interval,
        gamma_default,
        batch_size_default,
        replay_memory_size_default,
        feature_size,
        target_update,
        n_episodes,
        n_step,
        window_size,
        device,
        evaluation_parameter='gamma',
        transaction_cost=0)

    for gamma in gamma_list:
        run.gamma = gamma
        run.reset()
        run.train()
        run.evaluate_sensitivity()
        pbar.update(1)

    run.save_experiment()

    # test batch-size
    run = SensitivityRun(
        dataset_name,
        interval,
        gamma_default,
        batch_size_default,
        replay_memory_size_default,
        feature_size,
        target_update,
        n_episodes,
        n_step,
        window_size,
        device,
        evaluation_parameter='batch size',
        transaction_cost=0)

    for batch_size in batch_size_list:
        run.batch_size = batch_size
        run.reset()
        run.train()
        run.evaluate_sensitivity()
        pbar.update(1)

    run.save_experiment()

    # test replay memory size
    run = SensitivityRun(
        dataset_name,
        interval,
        gamma_default,
        batch_size_default,
        replay_memory_size_default,
        feature_size,
        target_update,
        n_episodes,
        n_step,
        window_size,
        device,
        evaluation_parameter='replay memory size',
        transaction_cost=0)

    for replay_memory_size in replay_memory_size_list:
        run.replay_memory_size = replay_memory_size
        run.reset()
        run.train()
        run.evaluate_sensitivity()
        pbar.update(1)

    run.save_experiment()
    pbar.close()
