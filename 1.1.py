import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


df = pd.read_csv("train.csv")


nul_matrix = df.isnull()
print(nul_matrix.sum())



cabin_mode = df['Cabin'].mode()[0]

age_median = df['Age'].median()

fare_mean = df['Fare'].mean()



df['Cabin'].fillna(cabin_mode, inplace=True)

df['Age'].fillna(age_median, inplace=True)

df['Fare'].fillna(fare_mean, inplace=True)



nul_matrix = df.isnull()
print (nul_matrix.sum())


scaler = MinMaxScaler()
scaler.fit(df[['Fare']])
df['Fare'] = scaler.transform(df[['Fare']])


df = pd.get_dummies(df, drop_first=True)



df.to_csv('ok_train.csv', index=False)

print(df.head())
