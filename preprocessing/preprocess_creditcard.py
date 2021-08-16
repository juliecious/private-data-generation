import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/creditcard.csv")
df.dropna(how='any', inplace=True)

train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)
train_df.to_csv('../data/creditcard_processed_train.csv', index=False)
test_df.to_csv('../data/creditcard_processed_test.csv', index=False)
