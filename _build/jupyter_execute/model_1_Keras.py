#!/usr/bin/env python
# coding: utf-8

# # Load Packages

# In[1]:


import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

tf.__version__


# In[3]:


# import dataset 
raw_dataset = pd.read_csv("car_prices.csv", on_bad_lines="skip")


# In[4]:


df = raw_dataset.copy()


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


# drop column with too many missing values
df = df.drop(['transmission'], axis=1)


# In[8]:


# drop remaining row with one missing value
df = df.dropna()


# In[9]:


# Drop irrelevant features
df = df.drop(['trim', 'vin', 'mmr', 'saledate'], axis=1)


# In[10]:


# rename columns
df = df.rename(columns={
"make" : "brand",
"body" : "type",
"odometer" : "miles"} 
    )


# In[11]:


# transform into lowercase
df["brand"] = df["brand"].str.lower()
df["model"] = df["model"].str.lower()
df["type"] = df["type"].str.lower()


# ## Define label

# In[12]:


y_label = 'sellingprice'


# ## Data format

# In[13]:


# Make a dictionary with int64 featureumns as keys and np.int32 as values
int_32 = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
# Change all columns from dictionary
df = df.astype(int_32)

# Make a dictionary with float64 columns as keys and np.float32 as values
float_32 = dict.fromkeys(df.select_dtypes(np.float64).columns, np.float32)
df = df.astype(float_32)


# In[14]:


int_32


# In[15]:


# Convert to categorical

# make a list of all categorical variables
cat_convert = ["brand", "model", "type", "state", "color", "interior", "seller"]

# convert variables
for i in cat_convert:
    df[i] = df[i].astype("string")


# In[16]:


# Convert to category
df['year'] = df['year'].astype("category")
df['condition'] = df['condition'].astype("category")


# In[17]:


# Make list of all numerical data (except label)
list_num = df.drop(columns=[y_label]).select_dtypes(include=[np.number]).columns.tolist()

# Make list of all categorical data which is stored as integers (except label)
list_cat_int = df.drop(columns=[y_label]).select_dtypes(include=['category']).columns.tolist()

# Make list of all categorical data which is stored as string (except label)
list_cat_string = df.drop(columns=[y_label]).select_dtypes(include=['string']).columns.tolist()


# In[18]:


list_num


# In[19]:


list_cat_int


# In[20]:


df.info()


# In[21]:


df.head()


# In[29]:


#df["seller"].unique()


# ## Data Splitting

# In[21]:


# Make validation data
df_val = df.sample(frac=0.2, random_state=1337)

# Create training data
df_train = df.drop(df_val.index)


# In[22]:


print(
    "Using %d samples for training and %d for validation"
    % (len(df_train), len(df_val))
)


# ## Transform to Tensors

# In[23]:


# Define a function to create our tensors

def dataframe_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    labels = df.pop(y_label) #y_label rausziehen und löschen
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels)) #ds für tensoren
    if shuffle:
        ds = ds.shuffle(buffer_size=10000) #len(dataframe)
    ds = ds.batch(batch_size)
    df = ds.prefetch(batch_size)
    return ds


# In[24]:


batch_size = 32

ds_train = dataframe_to_dataset(df_train, shuffle=True, batch_size=batch_size)
ds_val = dataframe_to_dataset(df_val, shuffle=True, batch_size=batch_size)


# In[25]:


ds_train


# # Feature preprocessing
# ### Numerical preprocessing function

# In[26]:


# Define numerical preprocessing function
def get_normalization_layer(name, dataset):
    
    # Create a Normalization layer for our feature
    normalizer = layers.Normalization(axis=None)

    # Prepare a dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    return normalizer


# ### Categorical preprocessing function

# In[27]:


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  
  # Create a layer that turns strings into integer indices.
  if dtype == 'string':
    index = layers.StringLookup(max_tokens=max_tokens)
  # Otherwise, create a layer that turns integer values into integer indices.
  else:
    index = layers.IntegerLookup(max_tokens=max_tokens) #, output_mode='multi_hot'

  # Prepare a `tf.data.Dataset` that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)

  # Encode the integer indices.
  encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

  # Apply multi-hot encoding to the indices. The lambda function captures the
  # layer, so you can use them, or include them in the Keras Functional model later.
  return lambda feature: encoder(index(feature))


# ### Data preprocessing

# In[28]:


all_inputs = []
encoded_features = []


# ### Numercial preprocessing

# In[29]:


# Numerical features
for feature in list_num:
  numeric_feature = tf.keras.Input(shape=(1,), name=feature)
  normalization_layer = get_normalization_layer(feature, ds_train)
  encoded_numeric_feature = normalization_layer(numeric_feature)
  all_inputs.append(numeric_feature)
  encoded_features.append(encoded_numeric_feature)


# In[30]:


encoded_features


# ### Categorical preprocessing

# In[31]:


for feature in list_cat_int:
  categorical_feature = tf.keras.Input(shape=(1,), name=feature, dtype='int32')
  encoding_layer = get_category_encoding_layer(name=feature,
                                               dataset=ds_train,
                                               dtype='int32',
                                               max_tokens=None)
  encoded_categorical_feature = encoding_layer(categorical_feature)
  all_inputs.append(categorical_feature)
  encoded_features.append(encoded_categorical_feature)


# In[32]:


for feature in list_cat_string:
  categorical_feature = tf.keras.Input(shape=(1,), name=feature, dtype='string')
  encoding_layer = get_category_encoding_layer(name=feature,
                                               dataset=ds_train,
                                               dtype='string',
                                               max_tokens=None)
  encoded_categorical_feature = encoding_layer(categorical_feature)
  all_inputs.append(categorical_feature)
  encoded_features.append(encoded_categorical_feature)


# In[33]:


#Merge
all_features = layers.concatenate(encoded_features)


# In[34]:


all_features


# In[35]:


# First layer
x = layers.Dense(32, activation="relu")(all_features)

# Dropout to prevent overvitting - soll sich auf die wichtigsten konzentrieren
x = layers.Dropout(0.5)(x)

# Output layer
output = layers.Dense(1)(x) #sigmoid nur für Classifikation // bei regression keine activation

# Group all layers 
model = tf.keras.Model(all_inputs, output)


# In[36]:


model.compile(optimizer="adam", 
              loss ="mse", 
              metrics=["mean_absolute_error"])
              
              #regression Metrics verwenden!!!!


# In[37]:


tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


# ## Training

# In[40]:


model.fit(ds_train, epochs=50, validation_data=ds_val)
#Anzahl der Epochen: sobald val_accuracy nicht mehr gesteigert werden kann
#4 Epochen sind genug


# In[41]:


#im "echten" die testdaten nehmen
loss, accuracy = model.evaluate(ds_val)

print("MAE", round(accuracy, 2))


# ## Perform inference

# In[42]:


model.save('my_car_model-mean-absolute')


# In[43]:


reloaded_model = tf.keras.models.load_model('my_car_model-mean-absolute')


# In[44]:


df.head()


# In[49]:


sample = {
    "year": 2015,
    "brand": "kia",
    "model": "sorento",
    "type": "suv",
    "state": "ca",
    "condition": 5.0,
    "miles": 9393.0,
    "color": "white",
    "interior": "black",
    "seller": "kia motors america, inc",
}


# In[50]:


input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}


# In[51]:


predictions = reloaded_model.predict(input_dict)


# In[52]:


predictions

