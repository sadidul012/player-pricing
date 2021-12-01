import pandas as pd
from helpers import *
from utils import *
from sklearn.preprocessing import OrdinalEncoder

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

dataset_2022 = pd.read_csv(Path(data_root, "Database_2022.csv"))

pc_prices = dataset_2022[["PC dialy prices"]]
pc_prices["PC dialy prices"] = pc_prices["PC dialy prices"].apply(lambda x: split_price_history(x))
pc_prices = pc_prices.explode("PC dialy prices")
price_history = pc_prices["PC dialy prices"].apply(pd.Series)
price_history.columns = ["date", "price"]
pc_prices = pd.concat([pc_prices, price_history], axis=1)

pc_players = dataset_2022[["Data-version", "Rating", "Name"]]
pc_players = pd.merge(pc_players, pc_prices, left_index=True, right_index=True)

train_columns = [x for x in dataset_2022.columns if x not in exclude_columns]
train_data = dataset_2022[train_columns]

pc_games = dataset_2022['GAMES(PS,Xbox,PC)'].apply(lambda x: float_convert(x, 0))
pc_goals = dataset_2022['GOALS(PS,Xbox,PC)'].apply(lambda x: float_convert(x, 0))
pc_assists = dataset_2022['ASSISTS(PS,Xbox,PC)'].apply(lambda x: float_convert(x, 0))
# train_data = pd.merge(train_data, pc_games, left_index=True, right_index=True)
train_data = pd.merge(train_data, pc_players[["price"]], left_index=True, right_index=True)
train_data.fillna(0, inplace=True)

oe = OrdinalEncoder()
train_data[["Data-version", "Position"]] = oe.fit_transform(train_data[["Data-version", "Position"]])

save_object(oe, "oe")
train_data = train_data.loc[train_data.price > 0]
train_data.drop_duplicates(inplace=True)


def shift(x):
    x["future_price"] = x["price"].shift(-1)
    return x


shifted_data = train_data.groupby('Player ID').apply(shift)
shifted_data["price_change"] = shifted_data["price"] - shifted_data["future_price"]
# shifted_data.dropna(inplace=True)

print(shifted_data.head(10))
print(shifted_data.shape)

shifted_data.to_csv(Path(processed_path, "Database_2022.csv"), index=False)
