

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import pandas_datareader.data as web
import matplotlib as mpl
import numpy as np
import yfinance as yahoo_finance

# Make function for calls to Yahoo Finance
def get_adj_close(ticker, start, end):
    '''
    A function that takes ticker symbols, starting period, ending period
    as arguments and returns with a Pandas DataFrame of the Adjusted Close Prices
    for the tickers from Yahoo Finance
    '''
    start = start
    end = end
    info = web.DataReader(ticker, data_source='yahoo', start=start, end=end)['Adj Close']
    return pd.DataFrame(info)


# Get Adjusted Closing Prices for Facebook, Tesla and Amazon between 2016-2017
fb = get_adj_close('fb', '1/2/2016', '31/12/2017')
tesla = get_adj_close('tsla', '1/2/2016', '31/12/2017')
amazon = get_adj_close('amzn', '1/2/2016', '31/12/2017')

# Calculate 30 Day Moving Average, Std Deviation, Upper Band and Lower Band
for item in (fb,):
    item['30 Day MA'] = item['Adj Close'].rolling(window=20).mean()

    # set .std(ddof=0) for population std instead of sample
    item['30 Day STD'] = item['Adj Close'].rolling(window=20).std()

    item['Upper Band'] = item['30 Day MA'] + (item['30 Day STD'] * 2)
    item['Lower Band'] = item['30 Day MA'] - (item['30 Day STD'] * 2)

# Simple 30 Day Bollinger Band for Facebook (2016-2017)
fb[['Adj Close', '30 Day MA', 'Upper Band', 'Lower Band']].plot(figsize=(12, 6))
plt.title('30 Day Bollinger Band for Facebook')
plt.ylabel('Price (USD)')
plt.show()

#stochastic value returner

def  stochastic(currentprice, high, low): #the high and low are the highest and lowest points in the past 15 periods
    upper = currentprice - low
    lower = high - low
    K = upper / lower * 100
    print("Overbought is 80 and oversold is 20              ")
    print(K)

stochastic(1, 2, 3)

# ___variables___
ticker = 'FB'

start_time = datetime.datetime(2017, 10, 1)
#end_time = datetime.datetime(2019, 1, 20)
end_time = datetime.datetime.now().date().isoformat() # today

connected = False
while not connected:
    try:
        ticker_df = web.get_data_yahoo(ticker, start=start_time, end=end_time)
        connected = True
        print('connected to yahoo')
    except Exception as e:
        print("type error: " + str(e))
        time.sleep( 5 )
        pass

# use numerical integer index instead of date
ticker_df = ticker_df.reset_index()
print(ticker_df.head(5))

df = ticker_df


def computeRSI(data, time_window):
    diff = data.diff(1).dropna()  # diff in one field(one day)

    # this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[diff > 0]

    # down change is equal to negative difference, otherwise equal to zero
    down_chg[diff < 0] = diff[diff < 0]


    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg = up_chg.ewm(com=time_window - 1, min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window - 1, min_periods=time_window).mean()

    rs = abs(up_chg_avg / down_chg_avg)
    rsi = 100 - 100 / (1 + rs)
    return rsi

df['RSI'] = computeRSI(df['Adj Close'], 14)




# plot corresponding RSI values and significant levels
plt.figure(figsize=(15,5))
plt.title('RSI chart')
plt.plot(df['Date'], df['RSI'])

plt.axhline(0, linestyle='--', alpha=0.1)
plt.axhline(20, linestyle='--', alpha=0.5)
plt.axhline(30, linestyle='--')

plt.axhline(70, linestyle='--')
plt.axhline(80, linestyle='--', alpha=0.5)
plt.axhline(100, linestyle='--', alpha=0.1)
plt.show()

#rate of change

#function to get daily close
def get_stock(stock, start, end):
    return web.DataReader(stock, 'yahoo',start, end)['Close']

#function for ROC
def ROC(df1, n):
    M = df1.diff(n - 1)
    N = df1.shift(n - 1)
    ROC = pd.Series(((M / N) * 100), name = 'ROC_' + str(n))
    return ROC

df1 = pd.DataFrame(get_stock('FB', '1/1/2016', '12,31,2016'))
df1['ROC'] = ROC(df1['Close'], 12)
df1.tail()

df1.plot(y=['Close'])
df1.plot(y=['ROC'])
plt.show()

#pivot points, 70% ACCURATE

yf.pdr_override()
start = datetime.datetime(2020,1,1)
now = datetime.datetime.now()

stock = 'FB'

while stock != 'quit':
    df = web.get_data_yahoo(stock, start, now)
    df['High'].plot(label='high')

    pivots = []
    dates = []
    counter = 0
    lastPivot = 0
    Range = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    daterange = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in df.index:
        currentMax = max(Range, default=0)
        value = round(df["High"][i], 2)

        Range = Range[1:9]
        Range.append(value)
        daterange = daterange[1:9]
        daterange.append(i)

        if currentMax == max(Range, default=0):
            counter += 1
        else:
            counter = 0
        if counter == 5:
            lastPivot = currentMax
            dateloc = Range.index(lastPivot)
            lastDate = daterange[dateloc]
            pivots.append(lastPivot)
            dates.append(lastDate)
    print()
    # print(str(pivots))
    # print(str(dates))
    timeD = datetime.timedelta(days=30)

    for index in range(len(pivots)):
        print(str(pivots[index]) + " :" + str(dates[index]))

        plt.plot_date([dates[index], dates[index] + timeD],
                      [pivots[index], pivots[index]], linestyle='-', linewidth=2, marker=',')

    plt.show()
