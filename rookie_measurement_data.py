
#%%
import requests
from bs4 import BeautifulSoup as bs 
import pandas as pd
import re

#%%
data = []
count = 1

for year in range(2013, 2019):
    url = requests.get('https://www.nbadraft.net/'+str(year)+'-nba-draft-combine-measurements')
    soup = bs(url.content, 'lxml')
    soup.prettify()
    table_nums = soup.find('div',{"tbody"})
    rows = table_nums.findAll('tr')
    for row in rows:
        column = row.findAll('td')
        cols=[x.text.strip() for x in cols]
        data.append(cols)
