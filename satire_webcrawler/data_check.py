import pandas as pd

onion = pd.read_csv('babylonbee_urls.csv', quotechar="\"")
print(onion.head())
print(onion.iloc[0])