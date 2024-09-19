import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
url = 'https://raw.githubusercontent.com/diegogarcia122/clpa2/main/train.csv'

df = pd.read_csv(url)
df_original = pd.read_csv(url)
print(df.head())
print('\n')

print(df.info())
print('\n')
print(df.describe())
print('\n')

importantes = ['Make', 'Model Year', 'Electric Range', 'Electric Vehicle Type']
print(df[importantes].head())
print('\n')

missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
print('\n')
df['Electric Range'] = df['Electric Range'].fillna(df['Electric Range'].median(), inplace=True)

sns.boxplot(x=df['Model Year'])
plt.show()

df = df[(np.abs(stats.zscore(df['Model Year'])) < 3)]

sns.boxplot(x=df['Model Year'])
plt.show()

scaler = StandardScaler()
df['Model Year_scaled'] = scaler.fit_transform(df[['Model Year']])

min_max_scaler = MinMaxScaler()
df['Model Year_normalized'] = min_max_scaler.fit_transform(df[['Model Year']])

print(df[['Model Year', 'Model Year_scaled', 'Model Year_normalized']].head())
print('\n')

Range_cut = pd.qcut(df['Model Year'], q=3, labels=['Old Car', 'Avg Car', 'New Car'])
df['Range_cat'] = Range_cut

category_mapping = {
    'Low Mileage': 0, 
    'Avg Mileage': 1, 
    'High Mileage': 2
}

print(df[['Model Year', 'Range_cat']].sample(8))
print('\n')

print("Antes del preprocesamiento:")
print(df_original.describe())
print('\n')
print("Después del preprocesamiento:")
print(df.describe())
