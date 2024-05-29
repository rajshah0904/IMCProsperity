from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import numpy as np
import pandas as pd

class Trader:
    def __init__(self):
        self.price_history: Dict[str, List[float]] = {}
        self.time_history: Dict[str, List[int]] = {}
        # Historical data storage for ORCHIDS
        self.orchids_price_history: List[float] = []
        self.orchids_time_history: List[int] = []
        #Market making spread
        self.spread = 3.0
        
        self.open_positions: Dict[str, float] = {}  # Stores the mid price when a position is opened
        self.order_count = 0  # Counter for orders to manage batches
        self.excluded_product = 'COCONUT_COUPON'  # Product to be excluded from trading
        self.orchids_coefficients = np.array([-0.22134912, 0.2266288, -0.5625779, 0.25243944, -0.21968077])
        self.orchids_intercept = 7.329249382019043
        
 

    def compute_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return np.nan

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def market_making(self, product: str, order_depth: OrderDepth) -> List[Order]:
        orders = []
        mid_price = self.compute_mid_price(order_depth)

        if np.isnan(mid_price) or product == self.excluded_product:
            return orders

        bid_price = int(mid_price - self.spread / 2)
        ask_price = int(mid_price + self.spread / 2)

        if product in self.open_positions:
            entry_price = self.open_positions[product]
            if mid_price <= entry_price - 5: 
                orders.append(Order(product, bid_price, -1))  # Sell to close position
                del self.open_positions[product]
                return orders

        # Market making with stop loss
        orders.append(Order(product, bid_price, 1))  # Buy 1 unit at bid price
        orders.append(Order(product, ask_price, -1))  # Sell 1 unit at ask price
        self.open_positions[product] = mid_price  # Record the mid price at position opening

        return orders
    
    def process_orchids(self, order_depth: OrderDepth, state: TradingState):
        orders = []
        mid_price = self.compute_mid_price(order_depth)
        if mid_price is not None:
            self.orchids_price_history.append(mid_price)
            self.orchids_time_history.append(state.timestamp)
        
        if len(self.orchids_price_history) > 5:
            features = np.array([
                self.orchids_price_history[-1], 
                self.orchids_time_history[-1], 
                self.orchids_time_history[-1]**2, 
                np.mean(self.orchids_price_history), 
                np.var(self.orchids_price_history)
            ])
            predicted_price = np.dot(features, self.orchids_coefficients) + self.orchids_intercept
        else:
            predicted_price = mid_price if mid_price is not None else 0
        
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask < predicted_price:
                quantity = min(1, 100 - state.position.get('ORCHIDS', 0))
                orders.append(Order('ORCHIDS', best_ask, quantity))
        
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid > predicted_price:
                quantity = min(1, state.position.get('ORCHIDS', 0) + 100)
                orders.append(Order('ORCHIDS', best_bid, -quantity))
        
        return orders
    
    def run(self, state: TradingState):
        print("traderData: ", state.traderData)
        print("Observations: ", state.observations)
        result = {}
        orders_executed = 0
        
        for product, order_depth in state.order_depths.items():
            if product == 'ORCHIDS':
                orders = self.process_orchids(order_depth, state)
            elif product == 'STARFRUIT' or product =='AMETHYSTS' or product =='COCONUT':
                orders = self.market_making(product, order_depth)
                result[product] = orders
                orders_executed += len(orders)
                # Check if the batch limit has been reached
                if orders_executed >= 20 and (product =='STARFRUIT' or product =='AMETHYSTS'):
                    orders_executed = 0  # Reset the order count

            traderData = "Updated state after trading"
            conversions = -state.position.get(product, 0)

        return result, conversions, traderData

