
# Source: https://realpython.com/pandas-python-explore-dataset/#querying-your-dataset

import pandas as pd

nba = pd.read_csv("./datasets/nba_all_elo.csv")
type(nba)

print ("Rows & columns: ", nba.shape, "\n")

print("Column names and their data types:")
nba.info()
print("\n")

# Take a peek into top rows
print("Top 5 rows in dataset:")
print(nba.head())
# Take a peek into N bottom rows
print("Bottom 5 rows in dataset:")
print(nba.tail())
print("\n")

print("Basic statistics:")
print (nba.describe())
print("\n")

print("Teams & their counts in the dataset:")
nba["team_id"].value_counts()


# In[10]:


# A DataFrame object has two axes: “axis 0” and “axis 1”.
# “axis 0” represents rows and “axis 1” represents columns
nba.axes


# In[19]:


# Boolean Indexing
prev_decade_df = nba[nba["year_id"] > 2010]
print ("Overall row count: ", nba.shape, "\n")
print ("Previous decade row count: ", prev_decade_df.shape, "\n")

#print("Top 5 rows in dataset:")
#print(nba.head())

# Positional indexing
print (nba.iloc[1], "\n")

# Label indexing
print (nba.loc[4])
#nba.loc[nba.year_id >= 2010]


# In[6]:


# Do a search for Baltimore games where both teams scored over 100 points
print (nba[(nba["_iscopy"] == 0) &
    (nba["pts"] > 100) &
    (nba["opp_pts"] > 100) &
    (nba["team_id"] == "BLB")])


# In[18]:


get_ipython().magic('matplotlib inline')
from matplotlib import pyplot

nba["fran_id"].value_counts().head(10).plot(kind="bar");


# In[17]:


# Show how many points Knicks scored throughout the seasons
nba[nba["fran_id"] == "Knicks"].groupby("year_id")["pts"].sum().plot();


# In[26]:


# ~~~~~~~~ Cleaning the data ~~~~~~~~~

nba.info()


# In[12]:


# Drop rows with NULL notes
rows_without_missing_notes = nba.dropna()
rows_without_missing_notes.shape


# In[13]:


# Drop the notes column
data_without_notes_columns = nba.dropna(axis=1)
data_without_notes_columns.shape


# In[29]:


# Replace with a default text
data_with_default_notes = nba.copy()
data_with_default_notes["notes"].fillna(
    value="no notes available",
    inplace=True)

data_with_default_notes["notes"].describe()


# In[34]:


# Check for invalid values
print (nba['year_id'].max())
print (nba['year_id'].min())


print (nba['pts'].max())
print (nba['pts'].min())


# In[35]:


nba[nba["pts"] == 0]


# In[14]:


# Data consistency
# Check for points scored by team vs by opponents vs the game outcome

# Check for wins
nba[(nba["pts"] > nba["opp_pts"]) & (nba["game_result"] != 'W')].empty

# Check for losses
nba[(nba["pts"] < nba["opp_pts"]) & (nba["game_result"] != 'L')].empty






