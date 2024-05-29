from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import numpy as np

class Trader:
    def __init__(self):
        self.position_limits = 60  # Maximum number of baskets that can be long or short
        self.batch_limit = 20  # Maximum number of orders per batch
        self.historical_diffs = []  # List to store historical price differences
        self.spread_multiplier = 0.5  # Multiplier to define significant price difference
        self.action_threshold = 1  # Threshold to determine action quantity
        self.base_quantity = 4  # Base quantity for less significant trades
        self.increased_quantity = 8  # Increased quantity for more significant trades

    def run(self, state: TradingState):
        print("Trader data:", state.traderData)
        print("Observations:", str(state.observations))

        result = {}
        orders_executed = 0

        def calculate_avg_price(order_depth):
            bids = [price for price, qty in order_depth.buy_orders.items() if qty > 0]
            asks = [price for price, qty in order_depth.sell_orders.items() if qty > 0]
            avg_bid = np.mean(bids) if bids else 0
            avg_ask = np.mean(asks) if asks else 0
            return (avg_bid + avg_ask) / 2 if (avg_bid and avg_ask) else 0

        basket_prices = {product: calculate_avg_price(state.order_depths.get(product, OrderDepth()))
                         for product in ['CHOCOLATE', 'STRAWBERRIES', 'ROSES']}
        basket_theoretical_price = sum([basket_prices['STRAWBERRIES'] * 6, basket_prices['CHOCOLATE'] * 4, basket_prices['ROSES']]) + 370
        basket_actual_price = state.observations.plainValueObservations.get('GIFT_BASKET', 0)

        print("Prices - ", basket_prices)
        print(f"Theoretical Basket Price: {basket_theoretical_price}, Actual Basket Price: {basket_actual_price}")

        price_diff = basket_actual_price - basket_theoretical_price
        self.historical_diffs.append(price_diff)
        std_dev = np.std(self.historical_diffs) if len(self.historical_diffs) > 1 else 1

        print(f"Price Difference: {price_diff}, Standard Deviation: {std_dev}")

        if abs(price_diff) > self.spread_multiplier * std_dev:
            qty = self.increased_quantity if abs(price_diff) > self.action_threshold * std_dev else self.base_quantity
            action = 'BUY' if price_diff > 0 else 'SELL'
        else:
            qty = 0
            action = None

        current_position = state.position.get('GIFT_BASKET', 0)
        if action:
            adjusted_qty = min(qty, self.position_limits - abs(current_position)) if action == 'BUY' else min(qty, abs(current_position) + self.position_limits)
            order_price = basket_theoretical_price if action == 'BUY' else basket_actual_price
            order = Order(symbol='GIFT_BASKET', price=order_price, quantity=adjusted_qty if action == 'BUY' else -adjusted_qty)
            result['GIFT_BASKET'] = [order]
            orders_executed += 1

            print(f"{action} {adjusted_qty}x GIFT_BASKET at {order_price}")

        # Check if the batch limit has been reached
        if orders_executed >= self.batch_limit:
            orders_executed = 0  # Reset the order count if necessary

        traderData = "Updated trader state"
        conversions = 0

        return result, conversions, traderData
