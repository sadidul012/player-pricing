import pandas as pd

old_data = "/mnt/Cache/data/player-pricing/Database_2022.csv"
new_data = "/mnt/Cache/data/player-pricing/updated_price (1).csv"

old_data = pd.read_csv(old_data)
new_data = pd.read_csv(new_data)
print(new_data.shape, old_data.shape)

merged = pd.merge(old_data, new_data[["Player ID", 'New PC dialy prices', 'New Xbox dialy prices', 'New PS dialy prices']], on="Player ID")

print(merged.shape)
# print(merged.columns)
# print(new_data["New PS dialy prices"])
merged[['PC dialy prices', 'Xbox dialy prices', 'PS dialy prices']] = merged[['New PC dialy prices', 'New Xbox dialy prices', 'New PS dialy prices']]
merged = merged[list(merged.columns)[:-3]]
merged.to_csv("updated_data.csv", index=False)
print(merged.columns)
print(merged.shape)
