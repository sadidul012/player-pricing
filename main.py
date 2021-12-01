import pandas as pd
from utils import *
from helpers import *
from tensorflow.keras.models import load_model


inference_data = pd.read_csv(Path(processed_path, "Database_2022.csv"))
names = inference_data[['Player ID', 'Name']].drop_duplicates()
inference_y = inference_data[["price_change"]]
inference_x = inference_data[[x for x in inference_data.columns if x not in exclude_for_x]]
inference_x = inference_x[list(inference_x.columns)[:-1]]

inference_x = inference_x[inference_y["price_change"].isna()]
inference_data = inference_data[['Data-version', 'Rating', 'Position', 'Player ID', 'price', 'price_change']].groupby("Player ID").agg("mean")
# inference_data = inference_data.join(names, on="Player ID", how="left")
estimator = load_object("lgb")
inference_data["lgb"] = estimator.predict(inference_x.values)

transformer = load_model("/mnt/Cache/data/player-pricing/models/transformer")
inference_data["transformer"] = transformer.predict(np.expand_dims(inference_x.values, -2)).squeeze()

inference_data.sort_values(by=['price_change', 'lgb', 'price'], ascending=False, inplace=True)
inference_data = names.join(inference_data, on="Player ID", lsuffix='_caller', rsuffix='_other')
print(inference_data.columns)
