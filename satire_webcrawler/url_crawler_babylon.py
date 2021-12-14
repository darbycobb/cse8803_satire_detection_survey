from bs4 import BeautifulSoup
import urllib.request,sys,time
import requests
import pandas as pd
import re

pagesToGet= 350

url = "https://babylonbee.com/news"

filename="babylonbee_urls.csv"
headers="Date,Link,Heading,Body\n"
with open(filename, 'a') as f:
    f.write(headers)

for page in range(1,pagesToGet+1):
    print('processing page :', page)
    
    url = 'https://babylonbee.com/news?page='+str(page)
    
    #an exception might be thrown, so the code should be in a try-except block
    try:
        #use the browser to get the url. This is suspicious command that might blow up.
        html_page=requests.get(url)                             # this might throw an exception if something goes wrong.
    
    except Exception as e:                                   # this describes what to do if an exception is thrown
        error_type, error_obj, error_info = sys.exc_info()      # get the exception information
        print ('ERROR FOR LINK:',url)                          #print the link that cause the problem
        print (error_type, 'Line:', error_info.tb_lineno)     #print error info and line that threw the exception
        continue                                              #ignore this page. Abandon this and go back.
    time.sleep(2) 

    soup=BeautifulSoup(html_page.text,'html5lib')

    links = soup.find_all('article-card')

    print(len(links))
    
    with open(filename, "a") as f:
        for j in links:
            
            heading = j[':title']
            link = "https://babylonbee.com" + j[':path'].replace("'", "")
            date = j[":published_on"].replace(',', '')
            print("heading: ", heading)
            #print('link: ', link)
            #print('date: ', date)
            #an exception might be thrown, so the code should be in a try-except block
            try:
                #use the browser to get the url. This is suspicious command that might blow up.
                inner_page=requests.get(link)                             # this might throw an exception if something goes wrong.
            
            except Exception as e:                                   # this describes what to do if an exception is thrown
                error_type, error_obj, error_info = sys.exc_info()      # get the exception information
                print ('ERROR FOR LINK:',url)                          #print the link that cause the problem
                print (error_type, 'Line:', error_info.tb_lineno)     #print error info and line that threw the exception
                continue
            time.sleep(2)

            inner_soup=BeautifulSoup(inner_page.text, 'html.parser')
            body = inner_soup.find_all("div", attrs={'class' : 'article-content mb-2'})
            #print("BODY: ", body)
            body_text = ""
            for i in body:
                for p in i.find_all('p', {'class' : None}):
                    body_text += p.text.strip()
            body_text = body_text.replace(',', '')
            #print("BODY TEXT: ", body_text)
            f.write(date + ',' + link  + ',' + heading + ',' + body_text + "\n")

        url = 'https://babylonbee.com/news?page='+str(page)
