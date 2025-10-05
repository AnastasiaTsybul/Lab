import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Пример: загрузим DataFrame (замени путь к CSV при необходимости)
df = pd.read_csv("coffe.csv")

# Проверим пропуски
nul_matrix = df.isnull()
print(nul_matrix.sum())

# Заполнение пропусков
cabin_mode = df['Cabin'].mode()[0]
room_median = df['RoomService'].median()
age_mean = df['Age'].mean()

df['Cabin'].fillna(cabin_mode, inplace=True)
df['RoomService'].fillna(room_median, inplace=True)
df['Age'].fillna(age_mean, inplace=True)

# Проверка после заполнения
nul_matrix = df.isnull()
print(nul_matrix.sum())

# Масштабирование возраста
scaler = MinMaxScaler()
scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])

# Преобразование категориальных признаков
df = pd.get_dummies(df, drop_first=True)

# Сохраним результат
df.to_csv('train.csv', index=False)

print(df.head())
