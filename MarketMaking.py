from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import numpy as np
import pandas as pd

class Trader:
    def __init__(self):
        self.price_history: Dict[str, List[float]] = {}
        self.time_history: Dict[str, List[int]] = {}
        self.spread = 3.0
        self.open_positions: Dict[str, float] = {}  # Stores the mid price when a position is opened
        self.order_count = 0  # Counter for orders to manage batches

    def compute_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return np.nan

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def market_making(self, product: str, order_depth: OrderDepth) -> List[Order]:
        orders = []
        mid_price = self.compute_mid_price(order_depth)

        if np.isnan(mid_price):
            return orders

        bid_price = int(mid_price - self.spread / 2)
        ask_price = int(mid_price + self.spread / 2)

        # Check for existing position and apply stop loss
        if product in self.open_positions:
            entry_price = self.open_positions[product]
            if mid_price <= entry_price - 5:  # STOP LOSS
                orders.append(Order(product, bid_price, -1))  # Sell to close position
                del self.open_positions[product]
                return orders

        # Standard market making with stop loss management
        orders.append(Order(product, bid_price, 1))  # Buy 1 unit at bid price
        orders.append(Order(product, ask_price, -1))  # Sell 1 unit at ask price
        self.open_positions[product] = mid_price  # Record the mid price at position opening

        return orders

    def run(self, state: TradingState):
        print("traderData: ", state.traderData)
        print("Observations: ", state.observations)
        result = {}
        orders_executed = 0

        for product, order_depth in state.order_depths.items():
            orders = self.market_making(product, order_depth)
            result[product] = orders
            orders_executed += len(orders)

            # Check if the batch limit has been reached
            if orders_executed >= 20:
                orders_executed = 0  # Reset the order count

        traderData = "Updated state after trading"
        conversions = 0

        return result, conversions, traderData
