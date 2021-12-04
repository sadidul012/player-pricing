import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from sklearn.model_selection import train_test_split
from helpers import *
from utils import *
from time import process_time
import numpy as np
from tensorflow.keras.models import load_model


start_time = process_time()
train_x = pd.read_csv(Path(processed_path, dataset_name))
train_x.dropna(inplace=True)
train_y = train_x[["price_change"]]
train_x = train_x[[x for x in train_x.columns if x not in exclude_for_x]]

train_x = train_x[list(train_x.columns)[:-1]]
train_x, test_x, train_y, test_y = train_test_split(train_x.values, train_y.values, random_state=42)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

estimator = load_object("lgb")
y_hat = estimator.predict(test_x)
print("\nLGB metrics\n", "*"*20)
print_losses(y_hat, np.array(test_y).ravel())

transformer = load_model(Path(data_root, "models", "transformer"))
y_hat = transformer.predict(np.expand_dims(test_x, -2)).squeeze()
print("\nTransformer metrics\n", "*"*20)
print_losses(y_hat, np.array(test_y).ravel())

print("\ntime", process_time() - start_time)
