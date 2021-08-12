import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
import os


#STOCK TO FETCH
symbol = 'TSLA'

# ***********************************************************************************************
# FETCH STOCK DATA (IF NEEDED)

csv_flag = False
print(os.path.isfile(f'./{symbol}.csv'))

if os.path.isfile(f'./{symbol}.csv'):
    pass
else:
    csv_flag = True
    print(f"No {symbol}.csv found...")

print(csv_flag)

while csv_flag:
    #INITIALIZE DATA API OBJECT
    symbol_get = yf.Ticker(symbol)

    #GET HISTORICAL DATA
    stock_hist = symbol_get.history(period="5d") 

    #DISPLAY HEAD IN GUI
    print(stock_hist.head())

    #WRITE TO CSV
    stock_hist.to_csv(f'{symbol}.csv', index = True)

    print(f"{symbol}.csv created...")

    csv_flag = False



# ***********************************************************************************************
# 
# Path of the file to read
# stock_file_path = ''

stock_data = pd.read_csv(f'{symbol}.csv')
print(stock_data.head())

#CREATE TARGET OBJECT AND CALL Y / CREATE X FROM FEATURES
y = stock_data.Open
# features = [f for f in (input().split()) if f in stock_data.columns]
features = ['High', 'Low', 'Close', 'Volume']
print(features)

X = stock_data[features]

# SPLIT
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# SPECIFY MODEL
stock_model = RandomForestRegressor(random_state=1)

# FIT MODEL
stock_model.fit(train_X, train_y)

# ***********************************************************************************************

# Make validation predictions and calculate mean absolute error
val_predictions = stock_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
stock_model = RandomForestRegressor(random_state=1)
stock_model.fit(train_X, train_y)
val_predictions = stock_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))


