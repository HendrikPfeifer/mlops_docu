#!/usr/bin/env python
# coding: utf-8

# # 2nd SK-learn model

# ##  Data
# 

# 
# ## Load packages

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant  

sns.set_theme()


# ## Import Dataset

# In[2]:


raw_dataset = pd.read_csv("car_prices.csv", on_bad_lines="skip")


# In[3]:


df = raw_dataset.copy()


# ## Data inspection

# In[4]:


df.head(2)


# In[5]:


df.info()


# In[6]:


# show missing values (missing values - if present - will be displayed in yellow)
sns.heatmap(df.isnull(), 
            yticklabels=False,
            cbar=False, 
            cmap='viridis');


# In[7]:


print(df.isnull().sum())


# ## Data transformation

# In[8]:


# drop column with too many missing values
df = df.drop(['transmission'], axis=1)

# drop remaining row with one missing value
df = df.dropna()


# In[9]:


# Drop irrelevant features
df = df.drop(['trim', 'vin', 'mmr'], axis=1)


# In[10]:


print(df.isnull().sum())


# In[11]:


# rename columns

df = df.rename(columns={
"make" : "brand",
"body" : "type",
"odometer" : "miles"} 
    )


# In[12]:


df.info()


# In[13]:


# transform into lowercase

df["brand"] = df["brand"].str.lower()
df["model"] = df["model"].str.lower()
df["type"] = df["type"].str.lower()


# # Categorial or numeric?
# 
# * year = categorial
# * brand = categorial
# * model = categorial
# * type = categorial
# * state = categorial
# * condition = categorial
# * miles = numeric
# * color = categorial
# * interior = categorial
# * seller = categorial
# * ratingprice = numeric
# * sellingprice = numeric
# * saledate = categorial

# In[14]:


# In kategorische Variablen umwandeln:

for cat in ["year", "brand", "model", "type", "state", "condition", "color", "interior", "seller", "saledate"]:
    df[cat] = df[cat].astype("category")


# In[15]:


df.info()


# In[16]:


# summary statistics for all categorical columns
df.describe(include=['category']).transpose()


# ## Data splitting

# In[17]:


train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

train_dataset


# ## Exploratory data analysis

# In[18]:


# summary statistics for all numerical columns
#round(train_dataset.describe(),2).transpose()


# In[19]:


sns.pairplot(train_dataset);


# ## Correlation analysis

# In[20]:


# Create correlation matrix for numerical variables
corr_matrix = train_dataset.corr()
corr_matrix


# In[21]:


# Simple heatmap
heatmap = sns.heatmap(corr_matrix)


# In[22]:


# Make a pretty heatmap

# Use a mask to plot only part of a matrix
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)]= True

# Change size
plt.subplots(figsize=(11, 15))

# Build heatmap with additional options
heatmap = sns.heatmap(corr_matrix, 
                      mask = mask, 
                      square = True, 
                      linewidths = .5,
                      cmap = 'coolwarm',
                      cbar_kws = {'shrink': .6,
                                'ticks' : [-1, -.5, 0, 0.5, 1]},
                      vmin = -1,
                      vmax = 1,
                      annot = True,
                      annot_kws = {"size": 10})


# ## Data preprocessing pipeline

# In[23]:


# Modules
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import set_config
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# In[24]:


# for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])


# In[25]:


# for categorical features  
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])


# In[26]:


# Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, selector(dtype_exclude="category")),
    ('cat', categorical_transformer, selector(dtype_include="category"))
        ])


# In[27]:


df.head()


# ## Simple regression

# In[28]:


# Select features for simple regression
features = ['miles']
X = df[features]

# Create response
y = df["sellingprice"]


# In[29]:


# check feature
X.info()


# In[30]:


# check label
y


# In[31]:


# check for missing values
print("Missing values X:",X.isnull().any(axis=1).sum())

print("Missing values Y:",y.isnull().sum())


# ## Data splitting

# In[32]:


from sklearn.model_selection import train_test_split

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Modeling

# In[33]:


from sklearn.linear_model import LinearRegression

# Create pipeline with model
lm_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lm', LinearRegression())
                        ])


# In[34]:


# show pipeline
set_config(display="diagram")
# Fit model
lm_pipe.fit(X_train, y_train)


# In[35]:


# Obtain model coefficients
lm_pipe.named_steps['lm'].coef_


# ## Evaluation with training data

# In[36]:


X_train.head()


# In[37]:


y_pred = lm_pipe.predict(X_train)


# In[38]:


from sklearn.metrics import r2_score

r2_score(y_train, y_pred)  


# In[39]:


from sklearn.metrics import mean_squared_error

mean_squared_error(y_train, y_pred)


# In[40]:


# RMSE
mean_squared_error(y_train, y_pred, squared=False)


# In[41]:


from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_train, y_pred)


# In[42]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
sns.set_theme(style="ticks")

# Plot with Seaborn

# We first need to create a DataFrame
df_train = pd.DataFrame({'x': X_train['miles'], 'y':y_train})

sns.lmplot(x='x', y='y', data=df_train, line_kws={'color': 'darkred'}, ci=False);


# In[43]:


import plotly.io as pio
import plotly.offline as py
import plotly.express as px

# Plot with Plotly Express
fig = px.scatter(x=X_train['miles'], y=y_train, opacity=0.65, 
                trendline='ols', trendline_color_override='darkred');

fig.show()


# In[52]:


sns.residplot(x=y_pred, y=y_train, scatter_kws={"s": 80});


# In[53]:


# wrongest predictions

# create dataframe
df_error = pd.DataFrame(
    { "y": y_train,
      "y_pred": y_pred,
      "error": y_pred - y_train
    })

# sort by error, select top 10 and get index
error_index = df_error.sort_values(by=['error']).nlargest(10, 'error').index

# show corresponding data observations
df.iloc[error_index]


# ## Evaluation with test data

# In[54]:


y_pred = lm_pipe.predict(X_test)


# In[55]:


print('MSE:', mean_squared_error(y_test, y_pred))

print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))


# In[56]:


# Plot with Plotly Express
fig = px.scatter(x=X_test['miles'], y=y_test, opacity=0.65, 
                trendline='ols', trendline_color_override='darkred')

fig.show()


# # Multiple Regression

# In[57]:


# Select features for multiple regression
features= [
 'miles',
 'brand',
 'model',
 'type',
 'condition',
 'color'
  ]
X = df[features]

X.info()
print("Missing values:",X.isnull().any(axis = 1).sum())

# Create response
y = df["sellingprice"]


# In[59]:


# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[60]:


# Create pipeline with model
lm_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lm', LinearRegression())
                        ])


# In[61]:


# show pipeline
set_config(display="diagram")
# Fit model
lm_pipe.fit(X_train, y_train)


# In[62]:


# Obtain model coefficients
lm_pipe.named_steps['lm'].coef_


# In[67]:


y_pred = lm_pipe.predict(X_test)


# In[68]:


r2_score(y_test, y_pred)


# In[69]:


mean_squared_error(y_test, y_pred)


# In[70]:


mean_squared_error(y_test, y_pred, squared=False)


# In[71]:


mean_absolute_error(y_test, y_pred)


# In[73]:





# In[ ]:




