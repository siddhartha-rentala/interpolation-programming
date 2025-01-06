import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def moving_average_crossover(n_short, n_long, data) -> pd.DataFrame:
    """ Calculates the moving average crossover strategy for time unit in the given time series.
    Args:
        n_short (float): The short moving average window.
        n_long (float): The long moving average window.
        data (pd.DataFrame): The time series to calculate the strategy for.
            - Index: Date
            - Columns: {'Price': Adjusted Closing Price of the asset}
    Returns:
        data (pd.DataFrame): Matching time series as input data with signal
            - Index: Date | Matches input data
            - Columns: {'MA_signal': Signal for the moving average crossover strategy. 1 = buy, -1 = sell, 0 = no signal}
    """
    data['SMA_short'] = data['Adj Close'].rolling(window=n_short, min_periods=1).mean()
    data['SMA_long'] = data['Adj Close'].rolling(window=n_long, min_periods=1).mean()

    data['MA_signal'] = 0
    
    data['MA_signal'] = np.where(data['SMA_short'] > data['SMA_long'], 1, 0)

    # Sell Signal (short MA crosses > long MA)
    data['MA_signal'] = np.where(data['SMA_short'] < data['SMA_long'], -1, data['MA_signal'])
    return data

def MACD(n1, n2, n3, data) -> pd.DataFrame:
    """ Calculates the MACD (Moving Average Convergence Divergence) strategy for a given time series.

    Args:
        n1 (int): The short-term exponential moving average window.
        n2 (int): The long-term exponential moving average window.
        n3 (int): The signal line exponential moving average window.
        data (pd.DataFrame): The time series to calculate the strategy for.
            - Index: Date
            - Columns: {'Adj Close': Adjusted Closing Price of the asset}  
    Returns:
        pd.DataFrame: The original time series with an additional column for the MACD signal.
            - Index: Date | Matches input data
            - Columns: {'MACD_signal': MACD strategy signal (1 = buy, -1 = sell, 0 = no signal)}
    """

    data['EMA_short'] = data['Adj Close'].ewm(span=n1, adjust=False).mean() # span is the window basically, adjust = false means we find the weight
    # without adjusting for the initial values 
    data['EMA_long'] = data['Adj Close'].ewm(span=n2, adjust=False).mean()
    
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    
    data['Signal_Line'] = data['MACD'].ewm(span=n3, adjust=False).mean()
    
    data['MACD_signal'] = 0     
    data['MACD_signal'] = np.where(data['MACD'] > data['Signal_Line'], 1, 0)  # Buy signal
    data['MACD_signal'] = np.where(data['MACD'] < data['Signal_Line'], -1, data['MACD_signal'])  # Sell signal
    
    return data[['Adj Close', 'MACD', 'Signal_Line', 'MACD_signal']]
    

def RSI(n, data) -> pd.DataFrame:
    """ Calculates the RSI (Relative Strength Index) strategy for a given time series.
    
    Args:
        n (int): The window size for the RSI calculation.
        data (pd.DataFrame): The time series to calculate the strategy for.
            - Index: Date
            - Columns: {'Adj Close': Adjusted Closing price of the asset}
    Returns:
        pd.DataFrame: The original time series with an additional column for the RSI signal.
            - Columns: {'RSI_signal': RSI strategy signal (1 = buy, -1 = sell, 0 = no signal)}
    """

    # Daily price change
    delta = data['Adj Close'].diff()
    
    gain = np.where(delta > 0, delta, 0)  # If the price increased
    loss = np.where(delta < 0, -delta, 0)  # If the price decreased (negative, hence -delta)

    avg_gain = pd.Series(gain).rolling(window=n, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=n, min_periods=1).mean()
    
    rs = avg_gain / avg_loss # relative strength

    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi

    data['RSI_signal'] = 0  
    data['RSI_signal'] = np.where(data['RSI'] < 30, 1, 0)  # Buy signal when RSI < 30
    data['RSI_signal'] = np.where(data['RSI'] > 70, -1, data['RSI_signal'])  # Sell signal when RSI > 70
    
    return data[['Adj Close', 'RSI', 'RSI_signal']]



test_data = yf.download('AAPL', start='2020-01-01', end='2021-01-01')
test_results = pd.DataFrame({'MA': {-1: 5, 0: 243, 1: 5}, 
                             'MACD': {-1: 11, 0: 231, 1: 11}, 
                             'RSI': {-1: 24, 0: 226, 1: 3}})

print(test_data)
print("Expected results:")
print(test_results)

MA_count = moving_average_crossover(5, 20, test_data.copy()).value_counts()
MACD_count = MACD(12, 26, 9, test_data.copy()).value_counts()
RSI_count = RSI(14, test_data.copy()).value_counts()

results = pd.DataFrame({'MA': MA_count, 'MACD': MACD_count, 'RSI': RSI_count}).sort_index()

print("\nActual results:")
print(results)

start_date = '2019-10-01'
end_date = '2024-10-01'

SP_F = yf.download('ES=F', start=start_date, end=end_date)
ZB_F = yf.download('ZB=F', start=start_date, end=end_date)
GC_F = yf.download('GC=F', start=start_date, end=end_date)
SI_F = yf.download('SI=F', start=start_date, end=end_date)
CL_F = yf.download('CL=F', start=start_date, end=end_date)
NG_F = yf.download('NG=F', start=start_date, end=end_date)



# Code Here
import pandas as pd
import numpy as np
def backtest_futures(futures_data, signal_col, initial_cash=100000):
    cash = initial_cash 
    position = 0  # this is the existing position , -1 short , 1 long - remember the sh
    trade_book = []

    for i in range(1, len(futures_data)):
        signal = futures_data[signal_col].iloc[i]
        prev_price = futures_data['Adj Close'].iloc[i - 1]
        current_price = futures_data['Adj Close'].iloc[i]

        if signal == 1:  
            if position == 0:  
                position = 1  
                trade_book.append({'Action': 'Buy', 'Price': current_price, 'Position': position})
            elif position == -1:  
                cash += prev_price - current_price
                position = 1
                trade_book.append({'Action': 'Close Short and Buy', 'Price': current_price, 'Position': position})
        
        elif signal == -1:  
            if position == 0:  
                position = -1  
                trade_book.append({'Action': 'Sell', 'Price': current_price, 'Position': position})
            elif position == 1:  
                cash += current_price - prev_price
                position = -1
                trade_book.append({'Action': 'Close Long and Sell', 'Price': current_price, 'Position': position})

    if position == 1:  
        cash += futures_data['Adj Close'].iloc[-1] - futures_data['Adj Close'].iloc[-2]
        trade_book.append({'Action': 'Close Long', 'Price': futures_data['Adj Close'].iloc[-1], 'Position': 0})
    elif position == -1:  
        cash += futures_data['Adj Close'].iloc[-2] - futures_data['Adj Close'].iloc[-1]
        trade_book.append({'Action': 'Close Short', 'Price': futures_data['Adj Close'].iloc[-1], 'Position': 0})

    
    total_trades = len(trade_book)
    profits = [entry['Price'] for entry in trade_book if entry['Action'].startswith('Close')]
    profit_volatility = np.std(profits) if profits else 0
    avg_profit = np.mean(profits) if profits else 0
    total_return = (cash - initial_cash) / initial_cash * 100

    return {
        'Return %': total_return,
        'Profit and Loss Volatility': profit_volatility,
        'Average Profit and Loss': avg_profit
    }

futures_list = [SP_F, ZB_F, GC_F, SI_F, CL_F, NG_F]
futures_names = ['ES=F', 'ZB=F', 'GC=F', 'SI=F', 'CL=F', 'NG=F']
signals = ['MA_signal', 'MACD_signal', 'RSI_signal']

results = {}

for future_data in futures_list:
    # MAC
    ma_result = moving_average_crossover(5, 20, future_data)
    future_data['MA_signal'] = ma_result['MA_signal']  
    
    # MACD
    macd_result = MACD(12, 26, 9, future_data)
    future_data['MACD_signal'] = macd_result['MACD_signal']  
    
    # RSI
    rsi_result = RSI(14, future_data)
    future_data['RSI_signal'] = rsi_result['RSI_signal']  

for future_data, future_name in zip(futures_list, futures_names):
    for signal in signals:
        performance = backtest_futures(future_data, signal)
        results[f'{future_name} - {signal}'] = performance

for pair, metrics in results.items():
    print(f"{pair}: {metrics}")




def calc_metrics(results):
    summary = {}

    for pair, metrics in results.items():
        total_return = metrics['Return %'] / 100  
        trading_days = metrics['Trading Days']  
        
        ar = (1 + total_return) ** (252 / trading_days) - 1
        ratio_ar_to_vol = ar / metrics['Profit and Loss Volatility']

        summary[pair] = {
            'Return %': metrics['Return %'],
            'P&L Volatility': metrics['Profit and Loss Volatility'],
            'Average P&L': metrics['Average Profit and Loss'],
            'Annualized Return': ar * 100,  
            'AR/Volatility Ratio': ratio_ar_to_vol
        }

    
    df_summary = pd.DataFrame(summary).T  #gpt suggested this for better data formatting
    return df_summary

df_summary = calc_metrics(results)
print(df_summary)


def top_3_pairs(df_summary):
    df_sorted = df_summary.sort_values(by='Annualized Return', ascending=False)
    top_3_pairs = df_sorted.head(3)
    
    return top_3_pairs
top_3_pairs = top_3_pairs(df_summary)
print("Top 3 indicators and their corresponding futues based on AR:")
print(top_3_pairs)



top_3_pairs.sort_values(by='P&L Volatility', ascending=True)



