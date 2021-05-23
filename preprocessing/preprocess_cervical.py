# https://www.kaggle.com/randyrose2017/for-beginners-using-keras-to-build-models#1.-Data-observation

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def convert_cervical(df):
    df = df.replace('?', np.nan)
    df = df.apply(pd.to_numeric)

    # Data Imputation
    # for continuous variable
    df['Number of sexual partners'] = df['Number of sexual partners'].fillna(df['Number of sexual partners'].median())
    df['First sexual intercourse'] = df['First sexual intercourse'].fillna(df['First sexual intercourse'].median())
    df['Num of pregnancies'] = df['Num of pregnancies'].fillna(df['Num of pregnancies'].median())
    df['Smokes'] = df['Smokes'].fillna(1)
    df['Smokes (years)'] = df['Smokes (years)'].fillna(df['Smokes (years)'].median())
    df['Smokes (packs/year)'] = df['Smokes (packs/year)'].fillna(df['Smokes (packs/year)'].median())
    df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].fillna(1)
    df['Hormonal Contraceptives (years)'] = df['Hormonal Contraceptives (years)'].fillna(df['Hormonal Contraceptives (years)'].median())
    df['IUD'] = df['IUD'].fillna(0) # Under suggestion
    df['IUD (years)'] = df['IUD (years)'].fillna(0) #Under suggestion
    df['STDs'] = df['STDs'].fillna(1)
    df['STDs (number)'] = df['STDs (number)'].fillna(df['STDs (number)'].median())
    df['STDs:condylomatosis'] = df['STDs:condylomatosis'].fillna(df['STDs:condylomatosis'].median())
    df['STDs:cervical condylomatosis'] = df['STDs:cervical condylomatosis'].fillna(df['STDs:cervical condylomatosis'].median())
    df['STDs:vaginal condylomatosis'] = df['STDs:vaginal condylomatosis'].fillna(df['STDs:vaginal condylomatosis'].median())
    df['STDs:vulvo-perineal condylomatosis'] = df['STDs:vulvo-perineal condylomatosis'].fillna(df['STDs:vulvo-perineal condylomatosis'].median())
    df['STDs:syphilis'] = df['STDs:syphilis'].fillna(df['STDs:syphilis'].median())
    df['STDs:pelvic inflammatory disease'] = df['STDs:pelvic inflammatory disease'].fillna(df['STDs:pelvic inflammatory disease'].median())
    df['STDs:genital herpes'] = df['STDs:genital herpes'].fillna(df['STDs:genital herpes'].median())
    df['STDs:molluscum contagiosum'] = df['STDs:molluscum contagiosum'].fillna(df['STDs:molluscum contagiosum'].median())
    df['STDs:AIDS'] = df['STDs:AIDS'].fillna(df['STDs:AIDS'].median())
    df['STDs:HIV'] = df['STDs:HIV'].fillna(df['STDs:HIV'].median())
    df['STDs:Hepatitis B'] = df['STDs:Hepatitis B'].fillna(df['STDs:Hepatitis B'].median())
    df['STDs:HPV'] = df['STDs:HPV'].fillna(df['STDs:HPV'].median())
    df['STDs: Time since first diagnosis'] = df['STDs: Time since first diagnosis'].fillna(df['STDs: Time since first diagnosis'].median())
    df['STDs: Time since last diagnosis'] = df['STDs: Time since last diagnosis'].fillna(df['STDs: Time since last diagnosis'].median())

    # for categorical variable
    df = pd.get_dummies(data=df, columns=['Smokes','Hormonal Contraceptives','IUD','STDs',
                                          'Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Citology','Schiller'])
    return df

if __name__ == "__main__":
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv')
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train.to_csv('./data/cervical_train.csv', index=False) # for ctgan

    train = convert_cervical(train)
    test = convert_cervical(test)

    train.to_csv('./data/cervical_processed_train.csv', index=False)
    test.to_csv('./data/cervical_processed_test.csv', index=False)
