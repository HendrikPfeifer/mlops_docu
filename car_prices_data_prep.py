
# load packages
import pandas as pd

# import dataset 
raw_dataset = pd.read_csv("car_prices.csv", on_bad_lines="skip")

df = raw_dataset.copy()

# drop column with too many missing values
df = df.drop(['transmission'], axis=1)

# drop remaining row with one missing value
df = df.dropna()

# Drop irrelevant features
df = df.drop(['trim', 'vin', 'mmr'], axis=1)

# rename columns
df = df.rename(columns={
"make" : "brand",
"body" : "type",
"odometer" : "miles"} 
    )

# transform into lowercase
df["brand"] = df["brand"].str.lower()
df["model"] = df["model"].str.lower()
df["type"] = df["type"].str.lower()

# convert data types
for cat in ["year", "brand", "model", "type", "state", "condition", "color", "interior", "seller", "saledate"]:
    df[cat] = df[cat].astype("category")

