import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    df = pd.read_csv('../data/cardio_train.csv', delimiter=';')

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train.to_csv('../data/cardio_processed_train.csv', index=False)
    test.to_csv('../data/cardio_processed_test.csv', index=False)