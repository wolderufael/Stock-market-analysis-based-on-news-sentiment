import yfinance as yf
import talib as ta
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

class StockAnalyzer:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
    
    def retrieve_stock_data(self):
        return yf.download(self.ticker, start=self.start_date, end=self.end_date)

    def filter_stock_data(self,data):
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        data['Date'] = pd.to_datetime(data['Date'])
        
        filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
        return filtered_data
    
    def calculate_moving_average(self, data, window_size):
        return ta.SMA(data, timeperiod=window_size)

    def calculate_technical_indicators(self, data):
        # Calculate various technical indicators
        data.loc[:, 'SMA'] = self.calculate_moving_average(data['Close'], 20)
        data.loc[:,'RSI'] = ta.RSI(data['Close'], timeperiod=14)
        data.loc[:,'EMA'] = ta.EMA(data['Close'], timeperiod=20)
        macd, macd_signal, _ = ta.MACD(data['Close'])
        data.loc[:,'MACD'] = macd
        data.loc[:,'MACD_Signal'] = macd_signal
        # Add more indicators as needed
        return data
   
    def plot_technical_indicators(self,data):
        fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle('Stock Analysis with Technical Indicators', fontsize=16)

        # Plot stock price and SMA
        axs[0].plot(data.index, data['Close'], label='Close', color='blue')
        axs[0].plot(data.index, data['SMA'], label='SMA', color='orange')
        axs[0].set_title('Stock Price and SMA')
        axs[0].legend(loc='upper left')

        # Plot RSI
        axs[1].plot(data.index, data['RSI'], label='RSI', color='green')
        axs[1].axhline(70, color='red', linestyle='--', label='Overbought')
        axs[1].axhline(30, color='blue', linestyle='--', label='Oversold')
        axs[1].set_title('Relative Strength Index (RSI)')
        axs[1].legend(loc='upper left')

        # Plot EMA
        axs[2].plot(data.index, data['Close'], label='Close', color='blue')
        axs[2].plot(data.index, data['EMA'], label='EMA', color='red')
        axs[2].set_title('Stock Price and EMA')
        axs[2].legend(loc='upper left')

        # Plot MACD
        axs[3].plot(data.index, data['MACD'], label='MACD', color='purple')
        axs[3].plot(data.index, data['MACD_Signal'], label='MACD Signal', color='orange')
        axs[3].set_title('Moving Average Convergence Divergence (MACD)')
        axs[3].legend(loc='upper left')

        # Formatting
        for ax in axs:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.grid(True)

        plt.xticks(rotation=90)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


    
    def calculate_portfolio_weights(self, tickers, start_date, end_date):
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        mu = expected_returns.mean_historical_return(data)
        cov = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, cov)
        weights = ef.max_sharpe()
        weights = dict(zip(tickers, weights.values()))
        return weights

    def calculate_portfolio_performance(self, tickers, start_date, end_date):
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        mu = expected_returns.mean_historical_return(data)
        cov = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, cov)
        weights = ef.max_sharpe()
        portfolio_return, portfolio_volatility, sharpe_ratio = ef.portfolio_performance()
        return portfolio_return, portfolio_volatility, sharpe_ratio


