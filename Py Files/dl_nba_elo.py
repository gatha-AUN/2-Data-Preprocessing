#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Download NBA ELO Dataset
import requests

download_url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv"
target_csv_path = "nba_all_elo.csv"

response = requests.get(download_url)
response.raise_for_status()    # Check that the request was successful
with open(target_csv_path, "wb") as f:
    f.write(response.content)
print("Download ready.")


# In[ ]:


# Kaggle Mushroom Classification Dataset
# https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/
# https://www.kaggle.com/uciml/mushroom-classification


# In[ ]:




