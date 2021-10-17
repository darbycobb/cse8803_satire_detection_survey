import pandas as pd

onion = pd.read_csv('spoof_urls_2021.csv', usecols=range(4), lineterminator='\n')
print(onion.head())