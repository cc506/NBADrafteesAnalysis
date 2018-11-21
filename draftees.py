import json
import requests

player = ''
isCBB = None

def checkPlayer():
    global player
    global isCBB

    

def check():
    htmlData = requests.get('https://www.sports-reference.com/cbb/players/'+ player + '-1.html')
