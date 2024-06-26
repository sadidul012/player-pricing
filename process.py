import pandas as pd
from helpers import *
from utils import *
from sklearn.preprocessing import OrdinalEncoder
import glob
from os.path import splitext, basename

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


def shift(x):
    x["future_price"] = x["price"].shift(-1)
    return x


def process(dataset, price_column_name, index, m_name):
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

    save_object(oe, "oe-"+m_name)
    train_data = train_data.loc[train_data.price > 0]
    train_data.drop_duplicates(inplace=True)
    train_data = train_data[sorted(train_data.columns)]

    train_data = train_data.groupby('Player ID').apply(shift)
    train_data["price_change"] = train_data["price"] - train_data["future_price"]

    return train_data


if __name__ == '__main__':
    datasets = list(glob.glob(str(data_root) + "/datasets/**.csv"))

    for dataset_name in datasets:
        model_name = splitext(basename(dataset_name))[0]
        print(model_name)
        try:
            raw_dataset = pd.read_csv(Path(data_root, dataset_name))
            pc_data = process(raw_dataset, price_column, game_index, model_name)
            pc_data.to_csv(Path(processed_path, basename(dataset_name)), index=False)
            print(model_name, "saved")
        except Exception as e:
            print(e)

    # raw_dataset = raw_dataset.iloc[2249:2260]
    # raw_dataset = raw_dataset.loc[raw_dataset["Rating"] == "Rating"]
    # pc_data = process(raw_dataset, price_column, game_index)
    # print("dataset:", dataset_name)
    # print(pc_data.head(10))
    # print(pc_data.shape)
    #
    # pc_data.to_csv(Path(processed_path, dataset_name), index=False)
