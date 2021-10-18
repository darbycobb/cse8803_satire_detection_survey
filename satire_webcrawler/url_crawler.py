from bs4 import BeautifulSoup
import urllib.request,sys,time
import requests
import pandas as pd

pagesToGet= 100

#url = "https://www.theonion.com/breaking-news/news"
url = "https://www.theonion.com/breaking-news/news-in-brief"

filename="onion_urls_brief.csv"
headers="Date,Link,Heading,Body\n"
with open(filename, 'a') as f:
    f.write(headers)

for page in range(51,pagesToGet+1):
    print('processing page :', page)
    
    print(url)
    
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

    soup=BeautifulSoup(html_page.text,'html.parser')
    links=soup.find_all('article')

    print(len(links))
    
    with open(filename, "a") as f:
        for j in links:
            heading = j.find("h2").text.strip().replace(',', '')
            print("heading: ", heading)

            link = j.find("a",attrs={'class':'sc-1out364-0 hMndXN js_link'})['href'].strip()
            print("link: ", link)
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
            date = inner_soup.find("time")['datetime'].strip()
            print("date: ", date)
            body_text = ""
            for p in inner_soup.find_all("p", attrs={'class' : 'sc-77igqf-0 bOfvBY'}):
                body_text += p.text
            print("body: ", body_text)
            body_text = body_text.replace(',', '')
            f.write(date + ',' + link  + ',' + heading + ',' + body_text + "\n")

        url = 'https://www.theonion.com/breaking-news/news-in-brief?startIndex='+str(page * 20)
