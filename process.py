import pandas as pd
from helpers import *
from utils import *
from sklearn.preprocessing import OrdinalEncoder

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


def shift(x):
    x["future_price"] = x["price"].shift(-1)
    return x


def process(dataset, price_column_name, index):
    pc_prices = dataset[[price_column_name]]
    pc_prices[price_column_name] = pc_prices[price_column_name].apply(lambda x: split_price_history(x))
    pc_prices = pc_prices.explode(price_column_name)
    price_history = pc_prices[price_column_name].apply(pd.Series)
    price_history.columns = ["date", "price"]
    pc_prices = pd.concat([pc_prices, price_history], axis=1)

    pc_players = dataset[["Data-version", "Rating", "Name"]]
    pc_players = pd.merge(pc_players, pc_prices, left_index=True, right_index=True)

    train_columns = [x for x in dataset.columns if x not in exclude_columns]
    train_data = dataset[train_columns]

    pc_games = dataset['GAMES(PS,Xbox,PC)'].apply(lambda x: float_convert(x, index))
    pc_goals = dataset['GOALS(PS,Xbox,PC)'].apply(lambda x: float_convert(x, index))
    pc_assists = dataset['ASSISTS(PS,Xbox,PC)'].apply(lambda x: float_convert(x, index))
    train_data = pd.merge(train_data, pc_games, left_index=True, right_index=True)
    train_data = pd.merge(train_data, pc_goals, left_index=True, right_index=True)
    train_data = pd.merge(train_data, pc_assists, left_index=True, right_index=True)
    train_data = pd.merge(train_data, pc_players[["price"]], left_index=True, right_index=True)
    train_data.fillna(0, inplace=True)

    oe = OrdinalEncoder()
    train_data[["Data-version", "Position"]] = oe.fit_transform(train_data[["Data-version", "Position"]])

    save_object(oe, "oe")
    train_data = train_data.loc[train_data.price > 0]
    train_data.drop_duplicates(inplace=True)
    train_data = train_data[sorted(train_data.columns)]

    train_data = train_data.groupby('Player ID').apply(shift)
    train_data["price_change"] = train_data["price"] - train_data["future_price"]

    return train_data


raw_dataset = pd.read_csv(Path(data_root, dataset_name))
pc_data = process(raw_dataset, price_column, game_index)

print(pc_data.head(10))
print(pc_data.shape)

pc_data.to_csv(Path(processed_path, dataset_name), index=False)
