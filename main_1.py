from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import numpy as np
import pandas as pd

class Trader:
    def __init__(self):
        # Dictionary to store historical prices and timestamps for each product
        self.price_history: Dict[str, List[float]] = {}
        self.time_history: Dict[str, List[int]] = {}
        # Initial setup for market making spread
        self.spread = 2.0  

    def compute_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return np.nan  # Handle cases where order book is empty

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def market_making(self, product: str, order_depth: OrderDepth) -> List[Order]:
        orders = []
        mid_price = self.compute_mid_price(order_depth)

        if np.isnan(mid_price):
            return orders

        bid_price = 998#int(mid_price - self.spread / 2)
        ask_price = 1002#int(mid_price + self.spread / 2)
        orders.append(Order(product, bid_price, 1))  # Buy 1 unit
        orders.append(Order(product, ask_price, -1))  # Sell 1 unit
        return orders

    def run(self, state: TradingState):
        print("traderData: ", state.traderData)
        print("Observations: ", state.observations)
        result = {}

        for product, order_depth in state.order_depths.items():
            if product == "AMETHYSTS":
                orders = self.market_making(product, order_depth)
            else:
                # Linear regression strategy for other products
                orders = []
                current_price = (min(order_depth.sell_orders.keys()) + max(order_depth.buy_orders.keys())) / 2
                if product not in self.price_history:
                    self.price_history[product] = []
                    self.time_history[product] = []
                self.price_history[product].append(current_price)
                self.time_history[product].append(state.timestamp)

                if len(self.price_history[product]) > 2:
                    X = np.array(self.time_history[product])
                    y = np.array(self.price_history[product])
                    A = np.vstack([X, np.ones(len(X))]).T
                    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
                    next_time = state.timestamp + 1
                    predicted_price = m * next_time + c
                    acceptable_price = predicted_price
                else:
                    acceptable_price = current_price  # Fallback to current price if not enough data

                print(f"Product: {product}, Acceptable price: {acceptable_price}")

                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    if best_ask < acceptable_price:
                        orders.append(Order(product, best_ask, -order_depth.sell_orders[best_ask]))

                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    if best_bid > acceptable_price:
                        orders.append(Order(product, best_bid, order_depth.buy_orders[best_bid]))

            result[product] = orders

        traderData = "Updated state after trading"
        conversions = 0  # Modify as needed based on strategy

        return result, conversions, traderData
