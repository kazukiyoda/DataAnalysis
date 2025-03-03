# 初手
# データ分析系の導入
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
# 機械学習系の導入
import lightgbm as lgb
import optuna
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
#ランダム変数
import random
np.random.seed(8513)
random.seed(8513)