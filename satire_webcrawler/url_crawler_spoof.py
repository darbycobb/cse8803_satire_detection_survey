from bs4 import BeautifulSoup
import urllib.request,sys,time
import requests
import pandas as pd

pagesToGet= 20

url = "https://www.thespoof.com/spoof-news/archive/2019/"
months = {'jan/' : 31, 'feb/' : 28, 'mar/' : 31, 'apr/' : 30, 'may/' : 31, 'jun/' : 30, 'jul/' : 31, 'aug/' : 31, 'sep/' : 30, 'oct/' : 31, 'nov/' :30, 'dec/' : 31}
#months = ['jan/', 'feb/']
filename="spoof_urls_2019.csv"
headers="Date,Link,Heading,Body\n"
with open(filename, 'a') as f:
    f.write(headers)

for month, days in months.items():
    for day in range(1, days):
        
        month_url = url + month + str(day) + '/'
        print('processing page :', month_url)
        
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
                heading = j.find('h2').text.replace(',', '')
                print('heading', heading)
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
                body = inner_soup.find_all(id='articlebody')
                body_text = ""
                for i in body:
                    for p in i.find_all('p'):
                        body_text += p.text.strip()

                body_text = body_text.replace(',', '')
                f.write(date + ',' + link  + ',' + heading + ',' + body_text + "\n")
