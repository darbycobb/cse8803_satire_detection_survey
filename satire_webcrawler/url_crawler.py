from bs4 import BeautifulSoup
import urllib.request,sys,time
import requests
import pandas as pd

pagesToGet= 10

upperframe=[]
url = "https://www.theonion.com/breaking-news/news"
for page in range(1,pagesToGet+1):
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
    print(links[0])
    print(len(links))
    filename="onion_urls.csv"
    headers="Heading,Link\n"
    
    with open(filename, "a") as f:
        f.write(headers)
        for j in links:
            heading = j.find("h2").text.strip()
            print("heading: ", heading)
            #link = "https://www.theonion.com/breaking-news"
            link = j.find("a",attrs={'class':'sc-1out364-0 hMndXN js_link'})['href'].strip()
            print("link: ", link)
            f.write(heading + ',' + link + "\n")
        print(page)
        url = 'https://www.theonion.com/breaking-news/news?startIndex='+str(page * 20)
