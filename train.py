import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from utils import *
from helpers import *
from time import process_time
import glob
from os.path import splitext, basename

import lightgbm as lgb

start_time = process_time()

parameters = {
    'n_estimators': [3000],
    'metric': ['mse'],
    'n_jobs': [4],
    'boosting_type': ['goss'],
    'learning_rate': [0.1, 0.01, 0.02],
    'colsample_bytree': [0.3],
    'reg_lambda': [0.011685550612519125],
    'reg_alpha': [0.04502045156737212],
    'min_child_weight': [16.843316711276092],
    'min_child_samples': [412],
    'num_leaves': [546],
    'max_depth': [5, 8, 16, 32],
    'cat_smooth': [36.40200359200525],
    'cat_l2': [12.979520035205597]
}

datasets = list(glob.glob(str(processed_path) + "/**.csv"))

for dataset_name in datasets:
    model_name = splitext(basename(dataset_name))[0]
    estimator = lgb.LGBMRegressor()
    estimator = GridSearchCV(estimator, parameters, verbose=3, n_jobs=2)

    train_x = pd.read_csv(Path(processed_path, dataset_name))
    train_x.dropna(inplace=True)
    train_y = train_x[["price_change"]]
    train_x = train_x[[x for x in train_x.columns if x not in exclude_for_x]]

    train_x = train_x[list(train_x.columns)[:-1]]
    train_x, test_x, train_y, test_y = train_test_split(train_x.values, train_y.values, random_state=42)
    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
    estimator.fit(np.array(train_x), np.array(train_y).ravel())

    save_object(estimator, "lgb-"+model_name)
    estimator = load_object("lgb-"+model_name)

    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
    y_hat = estimator.predict(test_x)

    print_losses(y_hat, np.array(test_y).ravel())

    print("time", process_time() - start_time)
