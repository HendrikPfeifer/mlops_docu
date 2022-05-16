#!/usr/bin/env python
# coding: utf-8

# # Data preperation
# 
# ### Here you can find information about the data preperation of the "used car prices"-dataset

# # Conclusion:
# 
# 
# * ratingprice and sellingprice have a very high correlation, therefore I would remove the column "ratingprice" from the dataset.
# * code is not necessary, therefore I would remove the column "code" from the dataset.
# * saledate is also unnecessary, therefore I would remove the column "saledate" from the dataset.
# 
# * there are almost only automatic cars in "drivetrain" - not sure if I need this column for my model

# ### Load packages

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# import dataset and save it as df

df = pd.read_csv("car_prices.csv", on_bad_lines="skip")


# In[3]:


# drop missing vales (dataset is still big enough)

df = df.dropna()


# In[4]:


# rename colums for better understanding (as described above)

df = df.rename(columns={
"make" : "brand",
"body" : "type",
"trim" : "version",
"transmission" : "drivetrain",
"vin" : "code",
"odometer" : "miles",
"mmr" : "ratingprice"} 
    )


# In[5]:


# transform into lowercase

df["brand"] = df["brand"].str.lower()
df["model"] = df["model"].str.lower()
df["type"] = df["type"].str.lower()
df["drivetrain"] = df["drivetrain"].str.lower()
df["state"] = df["state"].str.lower()
df["version"] = df["version"].str.lower()
df["color"] = df["color"].str.lower()
df["interior"] = df["interior"].str.lower()
df["seller"] = df["seller"].str.lower()


# In[6]:


# transform into categorial variables

for cat in ["year", "brand", "model", "version", "type", "drivetrain", "code", "state", "condition", "color", "interior", "seller", "saledate"]:
    df[cat] = df[cat].astype("category")


# In[7]:


# drop irrelevant features

df = df.drop(["code", "ratingprice", "saledate"], axis=1)


# In[8]:


df.info()


# In[9]:


df.head()


# In[ ]:




