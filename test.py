import pandas as pd
from sklearn.model_selection import train_test_split
from helpers import *
from utils import *
from time import process_time
import numpy as np

start_time = process_time()
train_x = pd.read_csv(Path(processed_path, "dataset_for_2022.csv"))
train_x = train_x.loc[train_x.price > 0]
train_y = train_x[["price"]]
train_x = train_x[list(train_x.columns)[:-1]]
train_x, test_x, train_y, test_y = train_test_split(train_x.values, train_y.values, random_state=42)

estimator = load_object("lgb")

y_hat = estimator.predict(test_x)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

print_losses(y_hat, np.array(test_y).ravel())

print("time", process_time() - start_time)
