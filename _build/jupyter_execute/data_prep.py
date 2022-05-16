#!/usr/bin/env python
# coding: utf-8

# # Create data prep file

# In[1]:


_data_preparation_file = 'car_prices_data_prep.py'


# In[2]:


get_ipython().run_cell_magic('writefile', '{_data_preparation_file}', '\n# load packages\nimport pandas as pd\n\n# import dataset \nraw_dataset = pd.read_csv("car_prices.csv", on_bad_lines="skip")\n\ndf = raw_dataset.copy()\n\n# drop column with too many missing values\ndf = df.drop([\'transmission\'], axis=1)\n\n# drop remaining row with one missing value\ndf = df.dropna()\n\n# Drop irrelevant features\ndf = df.drop([\'trim\', \'vin\', \'mmr\'], axis=1)\n\n# rename columns\ndf = df.rename(columns={\n"make" : "brand",\n"body" : "type",\n"odometer" : "miles"} \n    )\n\n# transform into lowercase\ndf["brand"] = df["brand"].str.lower()\ndf["model"] = df["model"].str.lower()\ndf["type"] = df["type"].str.lower()\n\n# convert data types\nfor cat in ["year", "brand", "model", "type", "state", "condition", "color", "interior", "seller", "saledate"]:\n    df[cat] = df[cat].astype("category")\n')


# 
