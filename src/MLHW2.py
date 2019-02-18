import pandas as pd
from pandas.plotting import scatter_purchaseCounts
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df_test = pd.read_csv("C:\\Users\\sky1\\PycharmProjects\\MLClass\\hw02\\Bike_test.csv")
df_train = pd.read_csv(
    "C:\\Users\\sky1\\PycharmProjects\\MLClass\\hw02\\Bike_train.csv"
)
# create another train test split since we have no
train, test = train_test_split(df_train, test_size=0.5, shuffle=True)

regr = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=100)

# rf = regr.fit(df_train[df_train.columns[~df_train.columns.isin(['count'])]], df_train['count'])
rf = regr.fit(train.columns[~train.columns.isin(["count"])], train["count"])
