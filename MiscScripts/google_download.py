#!/usr/bin/env python
# coding: utf-8

# In[102]:


from googlesearch import search
from bs4 import BeautifulSoup
import requests
import re

def download_search(title,num=20):
    regex = re.compile('[^a-zA-Z.!0-9$%?,.]')
    def clean(line):
        line = regex.sub(' ',line)
        return(line)

    strings = {}
    for i in search(title, tld="com", num=num, stop=num, pause=3):
        r = requests.get(i)
        soup = BeautifulSoup(r.text, 'html.parser')
        text = ''
        for j in soup.find_all('p'):
            text += ' ' + j.get_text()
        
        strings[i] = (clean(soup.title.string), clean(text))
        
    return(strings)


# In[103]:


s = download_search('ford ')


# In[104]:


s

