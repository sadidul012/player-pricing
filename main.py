import pandas as pd
from utils import *
from helpers import *
from tensorflow.keras.models import load_model


inference_data = pd.read_csv(Path(processed_path, dataset_name))

meta_data = inference_data[['Player ID', 'Name', 'Data-version', 'Position']].drop_duplicates()
oe = load_object("oe")
meta_data[["Data-version", "Position"]] = oe.inverse_transform(meta_data[["Data-version", "Position"]].astype(np.float))

inference_y = inference_data[["price_change"]]
inference_x = inference_data[[x for x in inference_data.columns if x not in exclude_for_x]]
inference_x = inference_x[list(inference_x.columns)[:-1]]

inference_x = inference_x[inference_y["price_change"].isna()]
inference_data = inference_data[['Rating', 'Player ID', 'price', 'price_change']].groupby("Player ID").agg("mean")
# inference_data = inference_data.join(names, on="Player ID", how="left")
estimator = load_object("lgb")
inference_data["lgb"] = estimator.predict(inference_x.values)

transformer = load_model(Path(data_root, "models", "transformer"))
inference_data["transformer"] = transformer.predict(np.expand_dims(inference_x.values, -2)).squeeze()

inference_data = meta_data.join(inference_data, on="Player ID", lsuffix='_caller', rsuffix='_other')
inference_data.sort_values(by=['price_change', 'lgb', 'price'], ascending=False, inplace=True)
inference_data.columns = ["Player ID", "Name", "Data-version", "Rating", "Position", "Current Price", "Average Price Change", "LGB Prediction", "Transformer Prediction"]
inference_data["Action"] = ((inference_data["Current Price"] + inference_data["LGB Prediction"]) * 0.05)
inference_data.loc[(inference_data["Current Price"] >= 150) & (inference_data["Current Price"] < 1000), "min_change"] = 50
inference_data.loc[(inference_data["Current Price"] >= 1000) & (inference_data["Current Price"] < 10000), "min_change"] = 100
inference_data.loc[(inference_data["Current Price"] >= 10000) & (inference_data["Current Price"] < 50000), "min_change"] = 250
inference_data.loc[(inference_data["Action"] < inference_data["LGB Prediction"]) & (inference_data["Action"] >= inference_data["min_change"]), "Action"] = "Buy"
inference_data.loc[inference_data["Action"] != "Buy", "Action"] = "Sell"
inference_data[["Player ID", "Name", "Data-version", "Rating", "Position", "Current Price", "Average Price Change", "LGB Prediction", "Transformer Prediction", "Action"]].to_csv("output.csv", index=False)
print(inference_data[inference_data["Current Price"] < 50000][["Current Price", "Average Price Change", "LGB Prediction", "action", "min_change"]].head())
