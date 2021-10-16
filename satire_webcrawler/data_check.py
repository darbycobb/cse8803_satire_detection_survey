import pandas as pd

onion = pd.read_csv('babylonbee_urls.csv', usecols=range(4), lineterminator='\n')
print(onion.head())