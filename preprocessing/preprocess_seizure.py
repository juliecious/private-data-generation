import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/Epileptic_Seizure_Recognition.csv')

df = df.drop(columns='Unnamed')

dic = {5: 0, 4: 0, 3: 0, 2: 0, 1: 1}
df['y'] = df['y'].map(dic)

X, y = df.drop('y', axis=1), df['y']

train, test = train_test_split(df, test_size=0.2, random_state=42)

train.to_csv('./data/seizure_processed_train.csv', index=False)
test.to_csv('./data/seizure_processed_test.csv', index=False)