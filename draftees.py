import urllib
import csv
import json
from bs4 import BeautifulSoup

#will be automated later . (player: first-last)
players = ['deandre-ayton','marvin-bagleyiii','jaren-jacksonjr','trae-young','mohamed-bamba','wendell-carterjr','collin-sexton','kevin-knox','mikal-bridges']

stats = ['school_name', 'g', 'gs', 'mp_per_g', 'fg_per_g', 'fga_per_g', 'fg_pct', 'fg2_per_g', 'fg2a_per_g', 'fg2_pct', 'fg3_per_g', 'fg3a_per_g', 'fg3_pct', 
        'ft_per_g', 'fta_per_g', 'ft_pct', 'orb_per_g', 'drb_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g', 'sos']

data = []

isCBB = None

for player in players:
    htmlData = urllib.urlopen('https://www.sports-reference.com/cbb/players/'+ player + '-1.html')

    soup = BeautifulSoup(htmlData, 'html.parser')

    name_box = soup.find('h1', attrs={'itemprop': 'name'})
    name = name_box.text.strip()

    
    for x in stats:
        box = soup.find('td', attrs={'data-stat': x})
        stat = box.text.strip()

        data.append((name, stat))
    
        with open('CollegeStats.csv', 'a') as csv_file:
            writer = csv.writer(csv_file)
            for stat in data:
                writer.writerow([stat])
