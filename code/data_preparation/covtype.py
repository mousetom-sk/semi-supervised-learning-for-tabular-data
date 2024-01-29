import os

import pandas as pd
import matplotlib.pyplot as plt


# prepare the path to data
data_prep_dir = os.path.dirname(__file__)
code_dir = os.path.dirname(data_prep_dir)
project_dir = os.path.dirname(code_dir)
data_dir = os.path.join(project_dir, "data")

# load the original dataset (not provided on GitHub)
df = pd.read_csv(os.path.join(data_dir, "covtype.csv"), delimiter=',')

# create training, validation, and test set with original proportions of labels
train = df.groupby("Cover_Type", group_keys=False).apply(lambda x: x.sample(frac=0.02))
df = df.drop(train.index)
val = df.groupby("Cover_Type", group_keys=False).apply(lambda x: x.sample(frac=0.005))
df = df.drop(val.index)
test = df.groupby("Cover_Type", group_keys=False).apply(lambda x: x.sample(frac=0.02))

# write the create datasets
train.to_csv(os.path.join(data_dir, "covtype-train.csv"))
val.to_csv(os.path.join(data_dir, "covtype-val.csv"))
test.to_csv(os.path.join(data_dir, "covtype-test.csv"))

#plot the label distribution
fig, axes = plt.subplots(ncols=3)

plot = train["Cover_Type"].value_counts().plot.bar(ax=axes[0])
plot = val["Cover_Type"].value_counts().plot.bar(ax=axes[1])
plot = test["Cover_Type"].value_counts().plot.bar(ax=axes[2])

plt.show()
