import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


df = pd.read_csv("coffe.csv")


nul_matrix = df.isnull()
print(nul_matrix.sum())


money_mode = df['money'].mode()[0]
room_median = df['hour_of_day'].median()
age_mean = df['Weekdaysort'].mean()

df['money'].fillna(money_mode, inplace=True)
df['hour_of_day'].fillna(room_median, inplace=True)
df['Weekdaysort'].fillna(age_mean, inplace=True)


nul_matrix = df.isnull()
print (nul_matrix.sum())


scaler = MinMaxScaler()
scaler.fit(df[['Weekdaysort']])
df['Weekdaysort'] = scaler.transform(df[['Weekdaysort']])


df = pd.get_dummies(df, drop_first=True)


df.to_csv('coffe.csv', index=False)

print(df.head())
