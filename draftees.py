import json

player = ''

def check():
    htmlData = requests.get('https://www.sports-reference.com/cbb/players/'+ player + '-1.html')
