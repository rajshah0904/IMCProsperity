from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import numpy as np

class Trader:
    def __init__(self, coconut_regression_params, coupon_premium):
        # Regression parameters for predicting coconut prices
        self.coconut_regression_params = coconut_regression_params
        self.coupon_premium = 637.63
        self.position_limits = {'COCONUT': 100, 'COUPON': 100}  # Example position limits

    def predict_coconut_price(self, features: List[float]) -> float:
        # Predict the price based on the provided features and the regression parameters
        return np.dot(features, self.coconut_regression_params['coefficients']) + self.coconut_regression_params['intercept']

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        traderData = state.traderData
        orders_executed = 0
        
        # Get current coconut prices and historical prices to predict the future price
        current_coconut_price = self.get_mid_price(state.order_depths['COCONUT'])
        historical_coconut_prices = self.get_historical_prices(state, 'COCONUT')
        
        # Predict future coconut price using regression
        predicted_coconut_price = self.predict_coconut_price(historical_coconut_prices)
        
        # Determine if we should buy or sell coconut coupons
        if predicted_coconut_price - current_coconut_price > self.coupon_premium:
            # The predicted price increase is greater than the cost of the coupon plus the premium
            # It's a buy signal for the coupons
            quantity_to_buy = self.position_limits['COUPON'] - state.position.get('COUPON', 0)
            if quantity_to_buy > 0:
                result['COUPON'] = [Order('COUPON', current_coconut_price, quantity_to_buy)]
                orders_executed += 1
        elif current_coconut_price - predicted_coconut_price > self.coupon_premium:
            # The predicted price decrease is greater than the cost of the coupon minus the premium
            # It's a sell signal for the coupons
            quantity_to_sell = state.position.get('COUPON', 0)
            if quantity_to_sell > 0:
                result['COUPON'] = [Order('COUPON', current_coconut_price, -quantity_to_sell)]
                orders_executed += 1
        
        # Check if the batch limit has been reached
        if orders_executed >= self.batch_limit:
            orders_executed = 0  # Reset the order count

        traderData = "Updated state after trading"
        conversions = 0

        return result, conversions, traderData

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        # Calculate mid price from order depth
        bids = [price for price, qty in order_depth.buy_orders.items()]
        asks = [price for price, qty in order_depth.sell_orders.items()]
        if not bids or not asks:
            return 0
        return (max(bids) + min(asks)) / 2

    def get_historical_prices(self, state: TradingState, product: str) -> List[float]:
        # Extract historical prices for a product from the trading state
        trades = state.market_trades.get(product, [])
        return [trade.price for trade in trades]