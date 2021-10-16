from bs4 import BeautifulSoup
import urllib.request,sys,time
import requests
import pandas as pd

pagesToGet= 20

url = "https://www.thespoof.com/spoof-news/archive/2021/"
#months = ['jan/', 'feb/', 'mar/', 'apr/', 'may/', 'jun/', 'jul/', 'aug/', 'sep/', 'oct/', 'nov/', 'dec/']
months = ['jan/', 'feb/']
filename="spoof_urls_2021.csv"
headers="Date,Link,Heading,Body\n"
with open(filename, 'a') as f:
    f.write(headers)

for page in months:
    print('processing page :', page)
    
    month_url = url + page
    
    #an exception might be thrown, so the code should be in a try-except block
    try:
        #use the browser to get the url. This is suspicious command that might blow up.
        html_page=requests.get(month_url)                             # this might throw an exception if something goes wrong.
    
    except Exception as e:                                   # this describes what to do if an exception is thrown
        error_type, error_obj, error_info = sys.exc_info()      # get the exception information
        print ('ERROR FOR LINK:',url)                          #print the link that cause the problem
        print (error_type, 'Line:', error_info.tb_lineno)     #print error info and line that threw the exception
        continue                                              #ignore this page. Abandon this and go back.
    time.sleep(2) 

    soup=BeautifulSoup(html_page.text,'html.parser')
    links=soup.find_all('div', {'class' : 'bigStory story'})

    print(len(links))
    
    with open(filename, "a") as f:
        for j in links:
            link = 'https://www.thespoof.com' + j.find('a')['href'].strip()
            heading = j.find('h2').text
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
            date = inner_soup.find(id='writtenOn').text.strip().replace(',', '')
            body_text = inner_soup.find(id='articlebody').text.strip()

            body_text = body_text.replace(',', '')
            f.write(date + ',' + link  + ',' + heading + ',' + body_text + "\n")
