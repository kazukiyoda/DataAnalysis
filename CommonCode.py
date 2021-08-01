# Common code
# データ実行時にだいたい行う共通コード

import pandas as pd
import numpy as np
import pandas_profiling
import lightgbm as lgb
import optuna
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

'''
基本的な流れ
- データの概要を観察
- 利用する変数を決める?
- 学習用データから検証用データを作成する
- 

予測に関して
- 予測器の学習を行う
- 

ハイパーパラメータ―のチューニング
- paramsについて
- 

クロスバリデーション
- foldのスコア
    - accuracy
    - logloss
        - 低いほど良い指標
'''


# 訓練データの読み込み
train = pd.read()
# 訓練データから目的変数を抜いたテーブル
# x_trainを作成する
x_train = train.drop(['目的変数'],axis = 1)
# 訓練データから目的変数だけの配列
# y_trainを作成する
y_train = train['目的変数']

# テストデータの読み込み
test = pd.read()
# この結果が提出する答えとなる
x_test = test.copy()

# 単純に訓練データと検証データに分ける場合
# train_test_splitを用いる
# kfoldを用いるときはその都度validを作成するので下のコードは実行しない
X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                    test_size=0.3,
                                                    random_state=42,
                                                    stratify=y_train)


# cross validationをして実行していく
# kfoldでシンプルなクロスバリデーションを行う

# 各foldのスコアを保存するリスト
# 正解率
scores_accuracy = []
# logloss:モデルの性能
scores_logloss = []
# mean_squared_error
scores_RMSE = []

# 実行する
kf = KFold(n_splits=4,shuffle=True,random_state=42)
for tr_idx, va_idx in kf.split(x_train):
    # 学習用データを学習データとバリデーションデータに分ける
    X_train, X_valid = x_train.iloc[tr_idx], x_train.iloc[va_idx]
    y_train, y_valid = y_train.iloc[tr_idx], y_train.iloc[va_idx]
    # モデルの学習
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    # パラメータチューニングの箇所はここになる
    params = {
    'objective':'regression'
    }
    model = lgb.train(params, lgb_train,
                valid_sets = [lgb_train, lgb_eval],
                verbose_eval = 10,
                num_boost_round = 1000,
                early_stopping_rounds = 10)
    # 検証データ
    y_pred_valid = model.predict(X_valid,num_iteration=model.best_iteration)
    # 検証データの正解と検証で予測したものを利用する
    logloss = log_loss(y_valid, y_pred_valid)
    accuracy = accuracy_score(y_valid, y_pred_valid)
    RMSE = mean_squared_error(Y_valid, y_pred_valid)
    # そのfoldのスコアを保存する
    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)
    scores_RMSE.append(RMSE)
# 各foldのスコアの平均を出力する
logloss = np.mean(scores_logloss)
accuracy = np.mean(scores_accuracy)
print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')

# 基本形
# trainとsplitで分けたのちに行っていく行動
# 予測器の学習から予測まで:not cross validation
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

params = {
    'objective'
}

model = lgb.train(params, lgb_train,
                valid_sets = [lgb_train, lgb_eval],
                verbose_eval = 10,
                num_boost_round = 1000,
                early_stopping_rounds = 10)

# テストデータで予測値を出す
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
# log_lossでモデルの性能を見れる

score = log_loss(y_valid, y_pred)
print(score)


# ハイパーパラメーターしてから予測する
def objective(trial):
    paramas = {
        'objective':'binary',
        'max_bin':trial.suggest_int('max_bin', 255, 500),
        'learning_rate':0.05,
        'num_leaves':trial.suggest_int('num_leaves', 32, 128)
    }
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    model = lgb.train(params, lgb_train,
                valid_sets = [lgb_train, lgb_eval],
                verbose_eval = 10,
                num_boost_round = 1000,
                early_stopping_rounds = 10)
    y_pred_valid = model.predict(X_valid,num_iteration=model.best_iteration)
    score = log_loss(y_valid, y_pred_valid)
    return score