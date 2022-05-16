#!/usr/bin/env python
# coding: utf-8

# # 1st SK-learn model
# 
# #### First try including sk-learn model and EDA

# ## Load packages

# In[1]:


import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt


# ## import dataset

# In[2]:


df = pd.read_csv("car_prices.csv", on_bad_lines="skip")


# In[3]:


df.head(2)


# In[4]:


print(f"We have {len(df.index):,} observations and {len(df.columns)} columns in our dataset.")


# In[5]:


df.columns


# In[6]:


df.dtypes


# In[7]:


df.info()


# # Nicht benötigte Spalten entfernen
# 
# * trim
# * vin
# * mmr

# In[8]:


df = df.drop(["trim", "vin", "mmr"], axis=1)


# In[9]:


df.info()


# ## Rename Columns

# In[10]:


df = df.rename(columns={
"make" : "brand",
"body" : "type",
"transmission" : "drivetrain",
"odometer" : "miles"} 
    )


# In[11]:


df.info()


# In[12]:


df.dtypes


# # Fehlende Werte löschen

# In[13]:


# Fehlende Werte ermitteln
df.isna().sum()


# In[14]:


df = df.dropna()


# In[15]:


df.isna().sum()


# # Alles in lowercase umwandeln

# In[16]:


df["brand"].value_counts()


# In[17]:


df["brand"] = df["brand"].str.lower()
df["model"] = df["model"].str.lower()
df["type"] = df["type"].str.lower()


# In[18]:


df["brand"].head()


# # Kategorial oder Numerisch?
# 
# * year = categorial
# * brand = categorial
# * model = categorial
# * type = categorial
# * drivetrain = categorial
# * state = categorial
# * condition = categorial
# * miles = numeric
# * color = categorial
# * interior = categorial
# * seller = categorial
# * ratingprice = numeric
# * sellingprice = numeric
# * saledate = categorial

# In[19]:


# In kategorische Variablen umwandeln:

for cat in ["year", "brand", "model", "type", "drivetrain", "state", "condition", "color", "interior", "seller", "saledate"]:
    df[cat] = df[cat].astype("category")


# In[20]:


df.dtypes


# In[21]:


df.describe(include="category").T


# In[22]:


df.describe()


# # Variable lists
# 
# Furthermore, we prepare our data for the following processes of data splitting and building of data pipelines.

# In[23]:


# list of all numerical data
list_num = df.select_dtypes(include=[np.number]).columns.tolist()

# list of all categorical data
list_cat = df.select_dtypes(include=['category']).columns.tolist()

print(list_num, list_cat)


# In[24]:


# define outcome variable as y_label
y_label = 'sellingprice'

# select features
features = df.drop(columns=[y_label]).columns.tolist()

# create feature data for data splitting
X = df[features]

# list of numeric features
feat_num = X.select_dtypes(include=[np.number]).columns.tolist()

# list of categorical features
feat_cat = X.select_dtypes(include=['category']).columns.tolist() 

# create response for data splitting
y = df[y_label]


# In[25]:


print(feat_num)


# # Train and test split

# In[26]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Data exploration set
# 
# We make a copy of the training data since we don’t want to alter our data during data exploration. We will use this data for our exploratory data analysis.

# In[27]:


df_train = pd.DataFrame(X_train.copy())
df_train = df_train.join(pd.DataFrame(y_train))


# # Analyze data 
# ## Categorical data

# In[28]:


df_train.describe(include="category").T 


# In[29]:


for i in list_cat:
    print(i, "\n", df_train[i].value_counts())


# In[30]:


for i in list_cat:

    TOP_10 = df[i].value_counts().iloc[:10].index

    g = sns.catplot(y=i, 
            kind="count", 
            palette="ch:.25", 
            data=df,
            order = TOP_10)    
    
    plt.title(i)
    plt.show();


# In[31]:


# Numercial gruped by categorical
# median
for i in list_cat:
    print(df_train.groupby(i).median().round(2).T)


# ## Numerical data

# In[32]:


# summary of numerical attributes
df_train.describe().round(2).T


# In[33]:


# histograms
df_train.hist(figsize=(20, 15));


# # Relationships
# ## Correlation with response
# 
# Detect the relationship between each predictor and the response:
# 

# In[34]:


#sns.pairplot(data=df_train, y_vars=y_label, x_vars=features);


# In[35]:


# pairplot with one categorical variable
#sns.pairplot(data=df_train, y_vars=y_label, x_vars=features);


# In[36]:


# inspect correlation
#corr = df_train.corr()
#corr_matrix[y_label].sort_values(ascending=False)

print(df_train.corr())
sns.heatmap(df_train.corr())


# In[37]:


# Data exploration


sns.set_theme(style="ticks", color_codes=True)


# In[38]:


sns.pairplot(df_train);


# In[39]:


sns.histplot(data=df_train, x="miles")


# In[40]:


# sns.histplot(data=df, x="ratingprice")


# In[41]:


sns.histplot(data=df_train, x="sellingprice")


# In[42]:


#Kategorisch

sns.countplot(x="brand", data=df_train)


# In[43]:


sns.countplot(x="year", data=df_train)


# # Fehlende Daten

# In[44]:


# show missing values (missing values - if present - will be displayed in yellow )
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis');


# In[45]:


# absolute number of missing values
print(df_train.isnull().sum())


# In[46]:


# percentage of missing values
print(df_train.isnull().sum() * 100 / len(df))


# # Data Pipeline

# In[47]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder


# In[48]:


# build numeric pipeline
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
    ])

num_pipeline
    


# In[49]:



# build categorical pipeline
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
cat_pipeline


# In[50]:



# create full pipeline
full_pipeline = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, feat_num),
        ('cat', cat_pipeline, feat_cat)])


full_pipeline


# # Model
# ## Model Training

# In[51]:


from sklearn.linear_model import LinearRegression


# Use pipeline with linear regression model
lm_pipe = Pipeline(steps=[
            ('full_pipeline', full_pipeline),
            ('lm', LinearRegression())
                         ])


# In[52]:


from sklearn import set_config
# Show pipeline as diagram
set_config(display="diagram")

# Fit model
lm_pipe.fit(X_train, y_train)

# Obtain model coefficients
lm_pipe.named_steps['lm'].coef_


# # Model Evaluation

# In[138]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# obtain predictions for training data
y_pred = lm_pipe.predict(X_train)

# R squared
r2_score(y_train, y_pred) 

# MSE
mean_squared_error(y_train, y_pred)

# RMSE
mean_squared_error(y_train, y_pred, squared=False)

# MAE
mean_absolute_error(y_train, y_pred)


# In[139]:


sns.residplot(x=y_pred, y=y_train, scatter_kws={"s": 80});


# # Model Tuning

# In[145]:


import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Pipeline
pipe = Pipeline(steps=[
    ("scaler", StandardScaler()), 
    ("pca", PCA()), 
    ("logistic", LogisticRegression(max_iter=10000, tol=0.1))])

# Data
X_digits, y_digits = datasets.load_digits(return_X_y=True)

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    "pca__n_components": [5, 15, 30, 45, 60],
    "logistic__C": np.logspace(-4, 4, 4),
}

# Gridsearch
search = GridSearchCV(pipe, param_grid, n_jobs=2)
search.fit(X_digits, y_digits)

# Show results
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)


# # Evaluate best model

# In[ ]:





# # Evaluate on test set

# In[149]:


y_pred = lm_pipe.predict(X_test)

print('MSE:', mean_squared_error(y_test, y_pred))

print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))


# In[ ]:




