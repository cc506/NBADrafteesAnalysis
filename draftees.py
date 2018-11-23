import requests
import json
from bs4 import BeautifulSoup

player = 'deandre-ayton'
isCBB = None
htmlData = requests.get('https://www.sports-reference.com/cbb/players/'+ player + '-1.html')

soup = BeautifulSoup(htmlData.text, 'lxml')

 #This is the index of any table of that page. If you change it you can get different tables.
table = soup.find_all('table')[0] 
tab_data = [[celldata.text for celldata in rowdata.find_all(["th","td"])]
                        for rowdata in table.find_all("tr")]

for data in tab_data:
     print(' '.join(data))