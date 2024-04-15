from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import numpy as np

class Trader:
    def __init__(self):
        # Dictionary to store historical prices and timestamps for each product
        self.price_history: Dict[str, List[float]] = {}
        self.time_history: Dict[str, List[int]] = {}

    def run(self, state: TradingState):
        print("traderData: ", state.traderData)
        print("Observations: ", state.observations)
        result = {}

        # Process each product in the market data
        for product, order_depth in state.order_depths.items():
            orders = []

            # Update historical data
            current_price = (min(order_depth.sell_orders.keys()) + max(order_depth.buy_orders.keys())) / 2
            if product not in self.price_history:
                self.price_history[product] = []
                self.time_history[product] = []
            self.price_history[product].append(current_price)
            self.time_history[product].append(state.timestamp)

            # Perform linear regression if sufficient data is available
            if len(self.price_history[product]) > 2:
                # Using numpy for linear regression
                X = np.array(self.time_history[product])
                y = np.array(self.price_history[product])
                A = np.vstack([X, np.ones(len(X))]).T
                m, c = np.linalg.lstsq(A, y, rcond=None)[0]

                # Predict the next price
                next_time = state.timestamp + 1
                predicted_price = m * next_time + c
                acceptable_price = predicted_price
            else:
                acceptable_price = current_price  # fallback to current price if not enough data

            print(f"Product: {product}, Acceptable price: {acceptable_price}")

            # Determine buy or sell orders based on the predicted acceptable price
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_amount = order_depth.sell_orders[best_ask]
                if best_ask < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))

            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_amount = order_depth.buy_orders[best_bid]
                if best_bid > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))

            result[product] = orders

        # Manage trader state and possible conversions (if applicable)
        traderData = "Updated state after trading"
        conversions = 0  # Modify as needed based on strategy

        return result, conversions, traderData
