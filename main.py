import time

import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

public_holidays = ['1/1/2018', '16/2/2018', '17/2/2018', '19/2/2018', '30/3/2018', '31/3/2018', '2/4/2018',
                   '5/4/2018',
                   '1/5/2018', '22/5/2018', '18/6/2018', '2/7/2018', '25/9/2018', '1/10/2018', '17/10/2018',
                   '25/12/2018', '26/12/2018', '2/1/2017', '28/1/2017', '30/1/2017', '31/1/2017', '4/4/2017',
                   '14/4/2017', '15/4/2017', '17/4/2017', '1/5/2017', '3/5/2017', '30/5/2017', '1/7/2017',
                   '2/10/2017',
                   '5/10/2017', '28/10/2017', '25/12/2017', '26/12/2017']

no = 0
splitRatio = 0.2
shuffled = True


def load_data():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    return train_data, test_data

def preprocess_data3():
    train_data, test_data = load_data()
    X_org = train_data['date'].values
    y = train_data['speed'].values
    test_X_org = test_data['date'].values

    X = []
    test_X = []

    for x in X_org:
        x_f = []
        strs = x.split(' ')
        day, mon, year = strs[0].split('/')
        hour = strs[1].split(':')[0]
        x_f.append(int(day) / 31)
        x_f.append(int(mon) / 12)
        x_f.append(int(year) / 2018)
        x_f.append(int(hour) / 23)
        weekday = datetime.datetime.strptime(strs[0], "%d/%m/%Y").weekday()
        x_f.append(int(weekday) / 6)
        week_num = datetime.datetime.strptime(strs[0], "%d/%m/%Y").strftime('%W')
        x_f.append(int(week_num) / 52)
        season = (int(mon) - 1) // 3
        x_f.append(season / 3)
        ph = 0
        if strs[0] in public_holidays:
            ph = 1
        x_f.append(float(ph))
        X.append(x_f)

    for x in test_X_org:
        x_f = []
        strs = x.split(' ')
        day, mon, year = strs[0].split('/')
        hour = strs[1].split(':')[0]
        x_f.append(int(day) / 31)
        x_f.append(int(mon) / 12)
        x_f.append(int(year) / 2018)
        x_f.append(int(hour) / 23)
        weekday = datetime.datetime.strptime(strs[0], "%d/%m/%Y").weekday()
        x_f.append(int(weekday) / 6)
        week_num = datetime.datetime.strptime(strs[0], "%d/%m/%Y").strftime('%W')
        x_f.append(int(week_num) / 52)
        season = (int(mon) - 1) // 3
        x_f.append(season / 3)
        ph = 0
        if strs[0] in public_holidays:
            ph = 1
        x_f.append(float(ph))
        test_X.append(x_f)

    X = np.array(X)
    test_X = np.array(test_X)

    return X, y, test_X, test_X_org


def trainAndPredict(val, model):
    X, y, test_X, test_X_org = preprocess_data3()
    X_train, y_train = X, y
    if val:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=splitRatio, shuffle=shuffled)
    model.fit(X_train, y_train)
    if val:
        print(mean_squared_error(y_valid, model.predict(X_valid)))
    else:
        ans = model.predict(test_X)
        write_tocsv(ans)


def trainAndPredictMul(val, models, ratio):
    X, y, test_X, test_X_org = preprocess_data3()
    X_train, y_train = X, y
    if val:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=splitRatio, shuffle=shuffled,
                                                              random_state=12)
    ans_m = []
    for model in models:
        model.fit(X_train, y_train)
        new_X = X_valid if val else test_X
        ans_m.append(model.predict(new_X))
    ans = ans_m[0] * ratio[0]

    for i in range(1, len(ratio)):
        ans += ans_m[i] * ratio[i]

    if val:
        print(mean_squared_error(y_valid, ans))
    else:
        write_tocsv(ans)


def cv_mse(models, ratio):
    X, y, test_X, test_X_org = preprocess_data3()
    kfold = KFold(n_splits=5, shuffle=True)
    cvans = []
    for train, test in kfold.split(X, y):
        ans_m = []
        for model in models:
            model.fit(X[train], y[train])
            new_X = X[test]
            ans_m.append(model.predict(new_X))
        ans = ans_m[0] * ratio[0]
        for i in range(1, len(ratio)):
            ans += ans_m[i] * ratio[i]
        cvans.append(mean_squared_error(y[test], ans))
    print(cvans)
    print(np.mean(cvans))


def write_tocsv(ans):
    output = []
    for row in range(0, len(ans)):
        output.append(ans[row])
    np_data = np.array(output)
    # 写入文件
    pd_data = pd.DataFrame(np_data, columns=['speed'])
    # print(pd_data)
    pd_data.to_csv(f'submit{no}.csv', index=True, index_label='id')


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # 遍历拟合原始模型
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # 得到基模型，并用基模型对out_of_fold做预估，为学习stacking的第2层做数据准备
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # 学习stacking模型
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # 做stacking预估
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


def combineModel(val):
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3, max_iter=1500))
    KRR = KernelRidge(alpha=0.1, kernel='polynomial', degree=3, coef0=2.5)
    GBoost = GradientBoostingRegressor(max_depth=10, learning_rate=0.1, n_estimators=700)
    stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR), meta_model=lasso)

    model_xgb = xgb.XGBRegressor(max_depth=10, learning_rate=0.1, subsample=0.8, min_child_weight=0, gamma=0.05)
    model_lgb = lgb.LGBMRegressor(max_depth=10, learning_rate=0.1, num_leaves=100)
    model_cat = CatBoostRegressor(learning_rate=0.08, depth=10, l2_leaf_reg=2)
    models = [stacked_averaged_models, model_xgb, model_lgb, model_cat]
    ratio = [0.20, 0.25, 0.25, 0.30]
    trainAndPredictMul(val, models, ratio)

if __name__ == '__main__':
    start_time = time.time()
    no = 68
    splitRatio = 0.2
    shuffled = True
    combineModel(False)

    print(f'Times:{time.time() - start_time}')
