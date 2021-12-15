# A Survey of Methods for Satirical Fake News Detection

## Where Everything Is
The data is hosted on Google Drive and can be downloaded here: https://drive.google.com/file/d/1lYRTWBwGZoQV_P5slCMnt6Z1MGbzgeK3/view?usp=sharing
In the data folder, the zipped final data and the zipped WELFake dataset are included.
Code for each benchmarked model is in the models folder.

Code for data scraping and preprocessing is located in the satire_webcrawler folder. The respective data scraped from each website is in this folder as well.

## How to Run the Code
Each satirical website has its own scraping file in the satire_webcrawler folder. Each can be run using these simple commands:
```
python url_crawler_babylon.py
python url_crawler_onion.py
python url_crawler_spoof.py
```
After the data is collected, data_cleaning.ipynb can be run on a local Jupyter Notebook, VS Code, or Google Colaboratory. Note that the WELFake dataset is needed for this file. The data_cleaning.ipynb outputs the final dataset.

FNDNet and Bag of Words can be run in their respective folders using the command 
```
python main.py
```

DistilBERT can be run on its IPython notebook on Google Colaboratory with the appropriate data file uploaded.
