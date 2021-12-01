from pathlib import Path
import pickle

# data_root = Path("/input/player-pricing")
data_root = Path("/mnt/Cache/data/player-pricing")

objects_path = Path(data_root, "objects")
processed_path = Path(data_root, "processed")
processed_path.mkdir(parents=True, exist_ok=True)

exclude_columns = [
    'Scrape Date',
    'Added on',
    'Club ID',
    'League ID',
    'Nation ID (nationality)',
    'All versions resource ID',
    'GAMES(PS,Xbox,PC)',
    'GOALS(PS,Xbox,PC)',
    'ASSISTS(PS,Xbox,PC)',
    'YELLOW(PS,Xbox,PC)',
    'RED(PS,Xbox,PC)',
    'TOP 3 CHEM(PS,Xbox,PC)',
    'PC dialy prices',
    'Xbox dialy prices',
    'PS dialy prices',
    'Scraped Url'
]
exclude_for_x = ['Name', 'Player ID', "price_change", "future_price"]


def save_object(obj, obj_name):
    objects_path.mkdir(exist_ok=True, parents=True)
    with open(Path(objects_path, "{}.pkl".format(obj_name)), "wb") as file:
        pickle.dump(obj, file)


def load_object(obj_name):
    with open(Path(objects_path, "{}.pkl".format(obj_name)), "rb") as file:
        return pickle.load(file)
