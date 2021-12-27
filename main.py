import glob
from os.path import splitext, basename
import pandas as pd

from utils import *

from inference import inference
from process import process
from train import train_lgb
from train_transformer import train_transformer


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

    d = list(glob.glob(str(processed_path) + "/**.csv"))
    train_transformer(d)

    d = list(glob.glob(str(processed_path) + "/**.csv"))
    train_lgb(d)

    d = list(glob.glob(str(processed_path) + "/**.csv"))
    inference(d)

