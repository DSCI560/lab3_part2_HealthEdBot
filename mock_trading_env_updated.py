import mysql.connector
import yfinance as yf
import numpy as np
import warnings
import pandas as pd
import gc
from datetime import datetime, timedelta

# Filter out the specific FutureWarning related to TimedeltaIndex construction
warnings.filterwarnings("ignore", message="The 'unit' keyword in TimedeltaIndex construction is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning)


# Database configuration and connection setup
def connect_to_database():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='Ldy990912!',
        database='stock_portfolio'
    )
    return conn


def fetch_stock_data(stock_symbol, start_date, end_date):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(start=start_date, end=end_date)
    data.reset_index(inplace=True)
    # miss_val = input("Please choose a type among = (Forward Filling: ffill, Backward Filling: bfill, Interpolate: linear) ---> ")
    miss_val = 'ffill'
    if miss_val in ['ffill', 'bfill']:
        data.fillna(method = miss_val, inplace = True)
    else:
        data.interpolate(method = miss_val, inplace = True)

    # Data Time format for Attribute:Date
    if data['Date'].dtype == 'datetime64[ns]':
        # print('The date attribute is already in datetime format')
        1
    else:
        # print("The 'date' attribute is not in datetime format.")
        data['Date'] = pd.to_datetime(data['Date'])
        # print("Converted to datetime format")

    # Add Daily Returns as a new attributes
    data['daily_returns'] = data['Close']-data['Open']
    data['daily_returns'].fillna(0, inplace =True)
    return data

class MockTradingEnvironment:
    def __init__(self, initial_fund):
        self.initial_fund = initial_fund
        self.cash = initial_fund
        self.portfolio = {}
        self.conn = connect_to_database()
        self.cursor = self.conn.cursor()
        # self.load_initial_portfolio()
    
    # Initialize or load portfolio from the database
    def load_initial_portfolio(self, portfolio_name):
        try:
            # Check if the specified portfolio exists and get its ID
            self.cursor.execute("SELECT portfolio_id FROM portfolio WHERE name = %s", (portfolio_name,))
            portfolio = self.cursor.fetchone()

            if portfolio is None:
                print(f"Portfolio '{portfolio_name}' does not exist.")
                return

            portfolio_id = portfolio[0]

            # Fetch all stocks associated with the identified portfolio
            self.cursor.execute("SELECT symbol FROM stocks WHERE portfolio_id = %s", (portfolio_id,))
            stocks = self.cursor.fetchall()

            # Initialize the portfolio dictionary with fetched stocks, assuming initial shares and average_cost
            # These values could be adjusted based on your actual table structure and data
            for stock in stocks:
                symbol = stock[0]
                self.portfolio[symbol] = {'shares': 0, 'average_cost': 0.0}  # Placeholder values

            print(f"Loaded stocks for portfolio '{portfolio_name}': {[stock[0] for stock in stocks]}")
        except mysql.connector.Error as e:
            print(f"Database error: {e}")
        # No need to close the connection here, as it may be used by other methods


    # buy logic
    def buy_stock(self, symbol, quantity, date):
        # Fetch current stock data
        df = fetch_stock_data(symbol, '2023-01-01', date)
       
        # calculate moving averages
        df['Short_MA'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['Long_MA'] = df['Close'].rolling(window=50, min_periods=1).mean()
        
        # calculate RSI
        delta = df['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        RS = gain / loss
        df['RSI'] = 100 - (100 / (1 + RS))
        
        # Check for buy signal based on the latest data
        if df['Short_MA'].iloc[-1] > df['Long_MA'].iloc[-1] and df['RSI'].iloc[-1] < 70:
            # Fetch current stock price
            current_price = df['Close'].iloc[-1]
            # Calculate the total cost of the purchase
            total_cost = current_price * quantity
            if self.cash >= total_cost:
                # Update portfolio holdings
                if symbol in self.portfolio:
                    self.portfolio[symbol]['shares'] += quantity
                    # Update average cost
                    self.portfolio[symbol]['average_cost'] = ((self.portfolio[symbol]['average_cost'] * (self.portfolio[symbol]['shares'] - quantity)) + total_cost) / self.portfolio[symbol]['shares']
                else:
                    self.portfolio[symbol] = {'shares': quantity, 'average_cost': current_price}
                # Deduct total cost from cash
                self.cash -= total_cost
                # Log the trade
                print(f"Bought {quantity} shares of {symbol} at {current_price} per share.")
            else:
                print(f"Insufficient funds to buy {quantity} shares of {symbol}.")
        else:
            print(f"No buy signal for {symbol} based on SMA and RSI indicators.")
        del df
        gc.collect()
    
    
  
    # sell logic
    def sell_stock(self, symbol, quantity, date):
        # Fetch current stock data for the last 100 days to ensure enough data for SMA and RSI calculations
        df = fetch_stock_data(symbol, '2023-01-01', date)[-100:]
        
        # Calculate moving averages and RSI
        df['Short_MA'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['Long_MA'] = df['Close'].rolling(window=50, min_periods=1).mean()
        # calculate RSI
        delta = df['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        RS = gain / loss
        df['RSI'] = 100 - (100 / (1 + RS))
        
        # Check for sell signal based on the latest data
        if df['Short_MA'].iloc[-1] < df['Long_MA'].iloc[-1] or df['RSI'].iloc[-1] > 70:
            if symbol in self.portfolio and self.portfolio[symbol]['shares'] >= quantity:
                # Fetch current stock price
                current_price = df['Close'].iloc[-1]
                # Calculate the total revenue from the sale
                total_revenue = current_price * quantity
                # Update portfolio holdings
                self.portfolio[symbol]['shares'] -= quantity
                '''if self.portfolio[symbol]['shares'] == 0:
                    del self.portfolio[symbol]  # Remove the stock from the portfolio if no shares are left'''
                # Add total revenue to cash
                self.cash += total_revenue
                # Log the trade
                print(f"Sold {quantity} shares of {symbol} at {current_price} per share.")
            else:
                print(f"Not enough shares of {symbol} to sell or the stock is not in the portfolio.")
        else:
            print(f"No sell signal for {symbol} based on SMA and RSI indicators.")
        del df
        gc.collect()



    def update_portfolio_value(self, date):
        # Fetch current prices and update portfolio value
        total_value = 0
        for symbol, holdings in self.portfolio.items():
            current_data = fetch_stock_data(symbol, '2023-01-01', date)
            current_price = current_data['Close'].iloc[-1]
            stock_value = holdings['shares'] * current_price
            total_value += stock_value
            print(f"{symbol}: {holdings['shares']} shares at {current_price} per share. Total: {stock_value}")
        total_value += self.cash  # Adding cash holdings to total portfolio value
        print(f"Total Portfolio Value: {total_value}")

    
    def calculate_performance_metrics(self, days, date):
        # Assuming initial portfolio value is the initial fund
        initial_portfolio_value = self.initial_fund
        current_portfolio_value = sum(holdings['shares'] * fetch_stock_data(symbol, '2023-01-01', date)['Close'].iloc[-1] for symbol, holdings in self.portfolio.items()) + self.cash
        total_return = (current_portfolio_value - initial_portfolio_value) / initial_portfolio_value
        
        s = []
        for symbol, holdings in self.portfolio.items():
            begin_price = fetch_stock_data(symbol, '2023-01-01', date)['Close'].iloc[0]
            end_price = fetch_stock_data(symbol, '2023-01-01', date)['Close'].iloc[-1]
            avg_price = (begin_price + end_price) / 2
            daily_return = (end_price - avg_price) / avg_price
            s.append(daily_return)
            
        daily_returns = np.array(s)  # Example daily returns
        # Sharpe Ratio calculation (simplified with assumed risk-free rate and daily returns standard deviation)
        risk_free_rate = 0.01  # Example risk-free rate; adjust based on current rates

        # Assuming fetch_stock_data and other necessary imports and functions are defined
        daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calculate the excess returns by subtracting the daily risk-free rate from the daily returns
        excess_returns = daily_returns - daily_risk_free_rate
        
        # Calculate the average of the excess returns
        average_excess_return = np.mean(excess_returns)
        
        # Calculate the standard deviation of the daily returns
        std_dev = np.std(daily_returns)
    
        # Calculate the Sharpe Ratio
        sharpe_ratio = average_excess_return / std_dev
    
        # Annualize the Sharpe Ratio
        annualized_sharpe_ratio = sharpe_ratio * np.sqrt(252)

        print(f"Total Return: {total_return * 100:.2f}%")
        print(f"Annualized Sharpe Ratio: {annualized_sharpe_ratio:.2f}")



    def execute_trades(self, date):
        i = 0
        for symbol in self.portfolio.keys():
            # Fetch current stock data for analysis
            # df = fetch_stock_data(symbol, '2023-01-01', '2023-12-31')[-100:]  # Fetch the last 100 days of data
            df = fetch_stock_data(symbol, '2023-01-01', date)[-100:]  # Fetch the last 100 days of data
            # Calculate moving averages
            df['Short_MA'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['Long_MA'] = df['Close'].rolling(window=50, min_periods=1).mean()

            # Calculate RSI
            delta = df['Close'].diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            RS = gain / loss
            df['RSI'] = 100 - (100 / (1 + RS))


            # (FREE TO CHANGE) Example: attempt to buy a fixed quantity (e.g., 10 shares)
            shares = 10
           
            # Check for buy signal based on SMA and RSI
            if df['Short_MA'].iloc[-1] > df['Long_MA'].iloc[-1] and df['RSI'].iloc[-1] < 70:
                # Buy signal: Short MA is above Long MA and RSI is below 70
                # Determine the quantity to buy based on your strategy and available cash
                # Example: attempt to buy a fixed quantity (e.g., 10 shares)
                quantity_to_buy = min(shares, self.cash // df['Close'].iloc[-1])  # Ensure we have enough cash
                if quantity_to_buy > 0:
                    self.buy_stock(symbol, quantity_to_buy, date)
                    print(" ")
                    i = 1

            # Check for sell signal based on SMA and RSI
            elif df['Short_MA'].iloc[-1] < df['Long_MA'].iloc[-1] or df['RSI'].iloc[-1] > 70:
                # Sell signal: Short MA is below Long MA or RSI is above 70
                # Determine the quantity to sell based on your strategy and holdings
                # Example: sell a fixed quantity if available (e.g., 10 shares)
                if self.portfolio[symbol]['shares'] >= shares:
                    self.sell_stock(symbol, shares, date)
                    print(" ")
                    i = 1
            # del df
            # gc.collect()
        return i


# Main execution logic
initial_fund = 10000  # Example initial fund
trading_env = MockTradingEnvironment(initial_fund)
trading_env.load_initial_portfolio("tech")

# Simulate trading period
trading_days = 75  # Example trading days in a year
date_str = '2023-10-30'
date_obj = datetime.strptime(date_str, "%Y-%m-%d")

try:
    for day in range(trading_days):
        date_obj += timedelta(days=1)
        new_date = date_obj.strftime("%Y-%m-%d")
        print(new_date)
        if trading_env.execute_trades(new_date):
            trading_env.update_portfolio_value(new_date)

        print('------------------')
        # Additional logging or monitoring here
    trading_env.calculate_performance_metrics(trading_days, new_date)
    print("----------Final profit----------")
    trading_env.update_portfolio_value(new_date)
except Exception as e:
    print(f"An error occurred during the simulation: {e}")
finally:
    trading_env.conn.close()  # Ensure the database connection is closed
