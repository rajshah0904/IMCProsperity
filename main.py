
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List
import string
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd

price_day_0= 'prices_round_1_day_0.csv'
price_day_1= 'prices_round_1_day_-1.csv'
price_day_2= 'prices_round_1_day_-2.csv'

df_day_1= pd.read_csv(price_day_1, sep= ';')
df_day_1

day_1_amethysts = df_day_1[df_day_1['product'] == 'AMETHYSTS']
day_1_amethysts

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
