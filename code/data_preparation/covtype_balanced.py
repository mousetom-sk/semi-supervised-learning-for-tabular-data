import os

import pandas as pd
import matplotlib.pyplot as plt


# prepare the path to data
data_prep_dir = os.path.dirname(__file__)
code_dir = os.path.dirname(data_prep_dir)
project_dir = os.path.dirname(code_dir)
data_dir = os.path.join(project_dir, "data")

# load the original and the already created imbalanced datasets
df = pd.read_csv(os.path.join(data_dir, "covtype.csv"), delimiter=',')
df_val = pd.read_csv(os.path.join(data_dir, "covtype-val.csv"), delimiter=',')
df_test = pd.read_csv(os.path.join(data_dir, "covtype-test.csv"), delimiter=',')

# exclude the validation and test samples
df.drop(df_val.index)
df.drop(df_test.index)

# create balanced training set and write it
train = df.groupby("Cover_Type", group_keys=False).apply(lambda x: x.sample(1700))

train.to_csv(os.path.join(data_dir, "covtype-train-balanced.csv"))

# plot the label distribution
ax = plt.subplot()
plot = train["Cover_Type"].value_counts().plot.bar(ax=ax)
plt.show()
