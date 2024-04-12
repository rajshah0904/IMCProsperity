import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List
import string


price_day1 = pd.read_csv('prices_round_1_day_-1.csv', sep = ';')
price_day2 = pd.read_csv('prices_round_1_day_-2.csv', sep = ';')
price_day0 = pd.read_csv('prices_round_1_day_0.csv', sep = ';')

class Trader:
    def __init__(self, spread: float = 1.0):
        self.spread = spread
        self.position = {}  # Current position for each product
        self.cpnl = {}  # Cumulative profit and loss for each product
        self.volume_traded = {}  # Volume traded for each product

    def compute_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return np.nan  # Handle cases where order book is empty

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2
    
    def starfruit_lstm(self):
        df = pd.concat([price_day2, price_day1, price_day0], ignore_index=True)

        df['spread'] = df['ask_price_1'] - df['bid_price_1']
        df['mid_price_change'] = df['mid_price'].diff()
        df['volume_imbalance'] = (df['bid_volume_1'] - df['ask_volume_1']) / (df['bid_volume_1'] + df['ask_volume_1'])
        df['sma_5'] = df['mid_price'].rolling(window=5).mean()

        def calculate_rsi(data, window=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)

            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        def calculate_macd(data, slow=26, fast=12, signal=9):
            exp1 = data.ewm(span=fast, adjust=False).mean()
            exp2 = data.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return macd, signal_line

        df['rsi'] = calculate_rsi(df['mid_price'])

        df['macd'], df['signal_line'] = calculate_macd(df['mid_price'])

        df_starfruit = df.loc[df['product'] == 'STARFRUIT']

        max_timestamps = df_starfruit.groupby('day')['timestamp'].max().cumsum()

        # Shift the maximum timestamps series to start from zero
        shifted_max_timestamps = max_timestamps.shift(1).fillna(0)

        # Create a dictionary to map each day to its starting offset
        offsets = shifted_max_timestamps.to_dict()

        # Map the offsets back to the original data and adjust timestamps
        df_starfruit['adj_timestamp'] = df_starfruit.apply(lambda row: row['timestamp'] + offsets[row['day']], axis=1)

        selected_features = ['adj_timestamp', 'spread', 'mid_price_change', 'volume_imbalance', 'sma_5', 'rsi', 'macd']

        starfruit_feat = df_starfruit[selected_features]

        starfruit_feat.set_index('adj_timestamp', inplace=True)

        # Normalize the features
        n_features = len(selected_features)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(starfruit_feat.values)

        # Split data into training and test sets
        train_size = int(len(scaled_features) * 0.8)
        test_size = len(scaled_features) - train_size
        train, test = scaled_features[0:train_size,:], scaled_features[train_size:len(scaled_features),:]

        # Convert an array of values into a dataset matrix
        def create_dataset(dataset, look_back=1):
            X, Y = [], []
            for i in range(len(dataset) - look_back):
                a = dataset[i:(i + look_back), :]
                X.append(a)
                Y.append(dataset[i + look_back, 1])  # Assuming the mid_price_change is the target variable
            return np.array(X), np.array(Y)

        # Reshape into X=t and Y=t+1
        look_back = 60  
        X_train, y_train = create_dataset(train, look_back)
        X_test, y_test = create_dataset(test, look_back)

        # Reshape input to be [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], look_back, n_features))
        X_test = np.reshape(X_test, (X_test.shape[0], look_back, n_features))

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(look_back, n_features)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(1))
        model.compile(optimizer= 'adam', loss='mean_squared_error')

        # Fit the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=7)
        model.fit(X_train, y_train, epochs=90, batch_size=64, verbose=2, validation_data=(X_test, y_test))

        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        train_mse = mean_squared_error(y_train, train_predict)
        test_mse = mean_squared_error(y_test, test_predict)

        print(f'Training MSE: {train_mse}')
        print(f'Testing MSE: {test_mse}')
        
        return model

    def market_making(self, product: str, order_depth: OrderDepth) -> List[Order]:
        orders = []
        mid_price = self.compute_mid_price(order_depth)

        if np.isnan(mid_price):  # Skip if mid_price could not be calculated
            return orders

        bid_price = mid_price - self.spread / 2
        ask_price = mid_price + self.spread / 2

        # Assuming you want to place a buy and sell order for each product
        buy_order = Order(product=product, price=bid_price, quantity=1)  # Buy 1 unit
        sell_order = Order(product=product, price=ask_price, quantity=-1)  # Sell 1 unit

        orders.append(buy_order)
        orders.append(sell_order)
        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {product: [] for product in state.order_depths.keys()}
        timestamp = state.timestamp

        # Update positions based on state
        for product, order_depth in state.order_depths.items():
            self.position[product] = state.position.get(product, 0)
            orders = self.market_making(product, order_depth)
            result[product] += orders

            # Update volume traded and calculate P&L
            if product in state.own_trades:
                for trade in state.own_trades[product]:
                    self.volume_traded[product] = self.volume_traded.get(product, 0) + abs(trade.quantity)
                    if trade.buyer == "SUBMISSION":
                        self.cpnl[product] = self.cpnl.get(product, 0) - trade.quantity * trade.price
                    elif trade.seller == "SUBMISSION":
                        self.cpnl[product] = self.cpnl.get(product, 0) + trade.quantity * trade.price

        # Calculate total P&L and other performance metrics
        totpnl = sum(self.cpnl.values())
        print(f"Timestamp {timestamp}, Total P&L: {totpnl}")

        # Here you could add more detailed logging, performance analysis, or risk management features

        return result





