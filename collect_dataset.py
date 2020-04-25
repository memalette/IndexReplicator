from pandas_datareader import data
import pandas as pd 
import matplotlib.pyplot as plt

tickers = ['^GSPC', 'AAPL', 'MSFT', 'GILD', 'UNP', 'UTX', 'HPQ', 'CSCO', 'SLB', 'AMGN', 
			'JPY=X', 'GOLD', 'BBVA', 'T', 'A', 'F']

start_date = '2000-01-01'
end_date = '2020-01-01'

close_px = []
for tick in tickers: 
	tick_data = data.DataReader(tick, 'yahoo', start_date, end_date)
	close_px.append(tick_data['Close'])


data = pd.concat(close_px, axis=1)
data.columns = tickers

data = data.iloc[:-2,:]
data = data.dropna()


# get returns
returns = data.pct_change()
returns = returns.dropna()
returns.to_csv('returns.csv')