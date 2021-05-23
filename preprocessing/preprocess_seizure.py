import pandas as pd
from sklearn.model_selection import train_test_split

def convert_seizure(df):
    df = df.drop('Unnamed', axis=1)

    dic = {5: 0, 4: 0, 3: 0, 2: 0, 1: 1}
    df['y'] = df['y'].map(dic)

    return df

if __name__ == "__main__":
    df = pd.read_csv('https://raw.githubusercontent.com/juliecious/sml-dataset/master/dataSets/Epileptic_Seizure_Recognition.csv')

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train.to_csv('./data/seizure_train.csv', index=False)  # for ctgan

    train = convert_seizure(train)
    test = convert_seizure(test)

    train.to_csv('./data/seizure_processed_train.csv', index=False)
    test.to_csv('./data/seizure_processed_test.csv', index=False)