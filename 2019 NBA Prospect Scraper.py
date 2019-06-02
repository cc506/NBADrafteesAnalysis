
#%%
import requests
import lxml.html as lh
import pandas as pd
from bs4 import BeautifulSoup


#%%
data = []
count = 1
for year in range(1980,2020):
    website_url = requests.get('https://www.basketball-reference.com/leagues/NBA_'+str(year)+'_per_game.html')
    soup = BeautifulSoup(website_url.content,'lxml')
    soup.prettify()
    My_table = soup.find('table',{'id':"per_game_stats"})
    tabledata = My_table.findAll('td')
    for cell in tabledata:
        count = count+1
        if (count%29 == 2):
            data.append(str(year))
        data.append(cell.get_text())


#%%
import numpy as np
data2 = np.array(data)
refined = np.reshape(data2, (-1, 30))


#%%
tableheader = My_table.findAll('th')
headers = []
count = 0
for item in tableheader:
    if (count < 30):
        headers.append(item.get_text())
        count=count+1
headers.remove('Rk')
headers = ['Year'] + headers
print(headers)


#%%
import pandas as pd
df1 = pd.DataFrame(refined, columns=headers)
df1.head()


#%%
df1 = df1.drop_duplicates(subset='Player', keep='first')
rookies = df1[df1['Year'] != '1980']
rookies.head()


#%%
rookies = rookies.set_index('Player')
rookies2 = rookies.drop(['Year','Pos', 'Age', 'Tm', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P','3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB','DRB', 'TOV', 'PF'], axis=1)
rookies2.head()


#%%
rookies2['NBATRB'] = rookies2['TRB']
rookies2['NBAAST'] = rookies2['AST']
rookies2['NBASTL'] = rookies2['STL']
rookies2['NBABLK'] = rookies2['BLK']
rookies2['NBAPTS'] = rookies2['PTS']
rookies2 = rookies2.drop(['TRB', 'AST', 'STL', 'BLK', 'PTS'], axis=1)
rookies2.head()


#%%
players = rookies2.index.values
print(len(players))


#%%
maindata = []
for name in players:
    realname = name
    name = name.lower()
    name = name.replace(" ", "-")
    website_url = requests.get('https://www.sports-reference.com/cbb/players/'+ name +'-1.html')
    soup = BeautifulSoup(website_url.content,'lxml')
    soup.prettify()
    My_table = soup.find('table',{'id':'players_per_game'})
    if My_table is not None:
        maindata = maindata + [realname]
        tabledata = My_table.findAll('td')
        data = []
        for cell in tabledata:
            data.append(cell.get_text())
        maindata = maindata +data[-28:]


#%%
import numpy as np
maindata2 = np.array(maindata)
refined2 = np.reshape(maindata2, (-1, 29))


#%%
tableheader = My_table.findAll('th')
colheaders = []
count = 0
for item in tableheader:
    if (count < 29):
        colheaders.append(item.get_text())
        count=count+1
colheaders.remove('Season')
colheaders = ['Name'] + colheaders
print(colheaders)


#%%
import pandas as pd
college = pd.DataFrame(refined2, columns=colheaders)
college.tail()


#%%
college = college.drop(['\xa0'], axis=1)


#%%
col = college.columns
cols = []
for co in col:
    cols.append(co)
cols.remove('Name')
cols.remove('School')
cols.remove('Conf')
print(cols)


#%%
for col in cols:
    college[col] = pd.to_numeric(college[col], errors='coerce')


#%%
college = college.dropna()
college = college.drop(['Conf'], axis=1)
college.head()


#%%
college.to_csv('CollegeStatLog2.csv')


#%%
college.reset_index(inplace=True)


#%%
# maybe add year later
#Also maybe add position


#%%
final = college.merge(rookies2, left_on='Name', right_on='Player')
final = final.drop(['index'], axis=1)
final = final.set_index('Name')
final.head()


#%%
cols = rookies2.columns
for col in cols:
    final[col] = pd.to_numeric(final[col], errors='coerce')
final.head()


#%%
final = final.dropna()
final.to_csv('CollegeRookieStatLog.csv')


#%%
#maybe combine with dummified college name too?


