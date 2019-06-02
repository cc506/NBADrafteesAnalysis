
#%%
import pandas as pd 
import numpy as np 
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn
import matplotlib.pyplot as plt


#%%
df = pd.read_csv('CollegeRookieStatLog2.csv')
df.head()


#%%
dfend = pd.read_csv('2019DraftClass.csv')
dfend.head()

#%% [markdown]
# ## College Trends

#%%
dfend['Year'] = 2020
df = df.drop(['Pos','Age','NBATRB','NBAAST','NBASTL','NBABLK','NBAPTS'], axis=1)
df1 = df.append(dfend, ignore_index = True)
df1 = df1[df1['Year']>2000]
df1.tail()


#%%
df1.columns


#%%
plt.rcParams.update({'font.size': 14})

#%% [markdown]
# ### Points

#%%
fig, ax = plt.subplots(figsize=(15,7))
plt.title('Year vs Points Scored')
seaborn.boxplot(df1['Year'], df1['PTS'], ax=ax)
plt.xlabel('Year')
plt.ylabel('Points')
plt.show()

#%% [markdown]
# ### Assists

#%%
fig, ax = plt.subplots(figsize=(15,7))
plt.title('Year vs Assists')
seaborn.boxplot(df1['Year'], df1['AST'], ax=ax)
plt.xlabel('Year')
plt.ylabel('Assists')
plt.show()

#%% [markdown]
# ### Rebounds

#%%
fig, ax = plt.subplots(figsize=(15,7))
plt.title('Year vs Rebounds')
seaborn.boxplot(df1['Year'], df1['TRB'], ax=ax)
plt.xlabel('Year')
plt.ylabel('Rebounds')
plt.show()

#%% [markdown]
# ### 3 Point to 2 Point Attempt Ratio

#%%
df1['ratio'] = df1['3PA']/df1['2PA']
fig, ax = plt.subplots(figsize=(15,7))
plt.title('Year vs 3pt:2pt ratio')
plt.ylim(-.2, 2) 
seaborn.boxplot(df1['Year'], df1['ratio'], ax=ax)
plt.xlabel('Year')
plt.ylabel('3pt:2pt ratio')
plt.show()

#%% [markdown]
# ## NBA Rookie Trends

#%%
df2 = pd.read_csv('NBARookieData.csv')
df2.head()


#%%
df2.loc[df2['3PA'] > 0, '3P%'] = df2['3P']/df2['3PA']
df2.loc[df2['3PA'] <= 0, '3P%'] = 0
df2.loc[df2['FGA'] > 0, 'FG%'] = df2['FG']/df2['FGA']
df2.loc[df2['FGA'] <= 0, 'FG%'] = 0
df2.loc[df2['2PA'] > 0, '2P%'] = df2['2P']/df2['2PA']
df2.loc[df2['2PA'] <= 0, '2P%'] = 0
df2.loc[df2['FTA'] > 0, 'FT%'] = df2['FT']/df2['FTA']
df2.loc[df2['FTA'] <= 0, 'FT%'] = 0
df2.head()

#%% [markdown]
# ### Points

#%%
fig, ax = plt.subplots(figsize=(32,7))
plt.title('Year vs Points Scored')
seaborn.boxplot(df2['Year'], df2['PTS'], ax=ax)
plt.xlabel('Year')
plt.ylabel('Points')
plt.show()

#%% [markdown]
# ### Assists

#%%
fig, ax = plt.subplots(figsize=(32,7))
plt.title('Year vs Assists')
seaborn.boxplot(df2['Year'], df2['AST'], ax=ax)
plt.xlabel('Year')
plt.ylabel('Assists')
plt.show()

#%% [markdown]
# ### Rebounds

#%%



#%%
df2[df2['Year']==2019]['PTS'].mean()


#%%
#do this for rookie points
ptavg = []
#for x in range(1981,2020):
    #find yearly means and scale up the predictions
