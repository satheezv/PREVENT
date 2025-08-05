import torch
from torch.utils.data import random_split
import pandas as pd

df1 = pd.read_csv("capping_dataset.csv")
df2 = pd.read_csv("chemspeed_dataset.csv")
df3 = pd.read_csv("navigation_dataset.csv")
all_data = pd.concat([df1, df2, df3], ignore_index=True)


# Shuffle the entire dataset first
all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split sizes
train_size = int(0.8 * len(all_data))
eval_size = int(0.1 * len(all_data))
test_size = len(all_data) - train_size - eval_size

# Perform the random split (returns Subset objects with .indices)
train_subset, eval_subset, test_subset = random_split(all_data, [train_size, eval_size, test_size], generator=torch.Generator().manual_seed(42))

# Extract actual DataFrames from the subsets
train_df = all_data.iloc[train_subset.indices].reset_index(drop=True)
eval_df = all_data.iloc[eval_subset.indices].reset_index(drop=True)
test_df = all_data.iloc[test_subset.indices].reset_index(drop=True)

#  Save to CSV for reproducibility
train_df.to_csv("split_train.csv", index=False)
eval_df.to_csv("split_val.csv", index=False)
test_df.to_csv("split_test.csv", index=False)
