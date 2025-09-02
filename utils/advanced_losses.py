#!/usr/bin/env python3
"""
Advanced Loss Functions for Stock Price Prediction
Implements multi-objective losses for maximum accuracy
"""

import tensorflow as tf
import numpy as np

class DirectionalAccuracyLoss(tf.keras.losses.Loss):
    """
    Loss function that optimizes for directional accuracy
    Penalizes wrong direction predictions more than magnitude errors
    """
    
    def __init__(self, direction_weight=1.0, magnitude_weight=0.5, name='directional_accuracy_loss'):
        super(DirectionalAccuracyLoss, self).__init__(name=name)
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        
    def call(self, y_true, y_pred):
        """
        Calculate directional accuracy loss
        
        Args:
            y_true: True values (batch_size, forecast_horizon)
            y_pred: Predicted values (batch_size, forecast_horizon)
        """
        # Calculate price changes (directions)
        true_direction = tf.sign(y_true[:, 1:] - y_true[:, :-1]) if y_true.shape[1] > 1 else tf.sign(y_true)
        pred_direction = tf.sign(y_pred[:, 1:] - y_pred[:, :-1]) if y_pred.shape[1] > 1 else tf.sign(y_pred)
        
        # Directional loss (0 if same sign, 1 if different sign)
        direction_loss = tf.reduce_mean(
            tf.cast(true_direction * pred_direction <= 0, tf.float32)
        )
        
        # Magnitude loss (Huber loss for robustness)
        magnitude_loss = tf.keras.losses.huber(y_true, y_pred)
        
        # Combined loss
        total_loss = (self.direction_weight * direction_loss + 
                     self.magnitude_weight * magnitude_loss)
        
        return total_loss

class SharpeRatioLoss(tf.keras.losses.Loss):
    """
    Loss function that optimizes for Sharpe ratio
    Maximizes risk-adjusted returns
    """
    
    def __init__(self, risk_free_rate=0.02, name='sharpe_ratio_loss'):
        super(SharpeRatioLoss, self).__init__(name=name)
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        
    def call(self, y_true, y_pred):
        """Calculate negative Sharpe ratio as loss"""
        # Calculate returns
        true_returns = (y_true[:, 1:] - y_true[:, :-1]) / y_true[:, :-1]
        pred_returns = (y_pred[:, 1:] - y_pred[:, :-1]) / y_pred[:, :-1]
        
        # Calculate trading signals (1 if predicted up, -1 if predicted down)
        signals = tf.sign(pred_returns)
        
        # Calculate strategy returns
        strategy_returns = signals * true_returns
        
        # Calculate excess returns
        excess_returns = strategy_returns - self.risk_free_rate
        
        # Calculate Sharpe ratio
        mean_excess_return = tf.reduce_mean(excess_returns)
        std_returns = tf.math.reduce_std(excess_returns)
        
        # Avoid division by zero
        sharpe_ratio = tf.cond(
            std_returns > 1e-8,
            lambda: mean_excess_return / std_returns,
            lambda: tf.constant(0.0)
        )
        
        # Return negative Sharpe ratio as loss (we want to maximize Sharpe)
        return -sharpe_ratio

class MultiObjectiveLoss(tf.keras.losses.Loss):
    """
    Multi-objective loss combining price accuracy, directional accuracy, and risk metrics
    """
    
    def __init__(self, 
                 price_weight=0.4,
                 direction_weight=0.3, 
                 sharpe_weight=0.2,
                 volatility_weight=0.1,
                 name='multi_objective_loss'):
        super(MultiObjectiveLoss, self).__init__(name=name)
        self.price_weight = price_weight
        self.direction_weight = direction_weight
        self.sharpe_weight = sharpe_weight
        self.volatility_weight = volatility_weight
        
        # Individual loss components
        self.directional_loss = DirectionalAccuracyLoss()
        self.sharpe_loss = SharpeRatioLoss()
        
    def call(self, y_true, y_pred):
        """Calculate multi-objective loss"""
        # Price accuracy loss (Huber loss for robustness)
        price_loss = tf.keras.losses.huber(y_true, y_pred)
        
        # Directional accuracy loss
        direction_loss = self.directional_loss(y_true, y_pred)
        
        # Sharpe ratio loss
        sharpe_loss = self.sharpe_loss(y_true, y_pred)
        
        # Volatility prediction loss (penalize high volatility predictions)
        pred_volatility = tf.math.reduce_std(y_pred, axis=-1)
        true_volatility = tf.math.reduce_std(y_true, axis=-1)
        volatility_loss = tf.keras.losses.mse(true_volatility, pred_volatility)
        
        # Combine all losses
        total_loss = (self.price_weight * price_loss +
                     self.direction_weight * direction_loss +
                     self.sharpe_weight * sharpe_loss +
                     self.volatility_weight * volatility_loss)
        
        return total_loss

class TradingProfitLoss(tf.keras.losses.Loss):
    """
    Loss function that directly optimizes for trading profitability
    """
    
    def __init__(self, transaction_cost=0.001, name='trading_profit_loss'):
        super(TradingProfitLoss, self).__init__(name=name)
        self.transaction_cost = transaction_cost
        
    def call(self, y_true, y_pred):
        """Calculate trading profit loss"""
        # Calculate actual returns
        true_returns = (y_true[:, 1:] - y_true[:, :-1]) / y_true[:, :-1]
        
        # Calculate predicted direction (buy/sell signals)
        pred_changes = y_pred[:, 1:] - y_pred[:, :-1]
        signals = tf.sign(pred_changes)
        
        # Calculate position changes (when we enter/exit positions)
        position_changes = tf.abs(signals[:, 1:] - signals[:, :-1])
        
        # Calculate gross profits
        gross_profits = signals[:, :-1] * true_returns[:, 1:]
        
        # Calculate transaction costs
        transaction_costs = position_changes * self.transaction_cost
        
        # Net profits
        net_profits = gross_profits - transaction_costs
        
        # Return negative profit as loss (maximize profit)
        return -tf.reduce_mean(net_profits)

class QuantileLoss(tf.keras.losses.Loss):
    """
    Quantile loss for uncertainty quantification
    Useful for risk management in trading
    """
    
    def __init__(self, quantiles=[0.1, 0.5, 0.9], name='quantile_loss'):
        super(QuantileLoss, self).__init__(name=name)
        self.quantiles = quantiles
        
    def call(self, y_true, y_pred):
        """
        Calculate quantile loss for multiple quantiles
        y_pred should have shape (batch_size, num_quantiles)
        """
        losses = []
        
        for i, q in enumerate(self.quantiles):
            # Extract predictions for this quantile
            y_pred_q = y_pred[:, i:i+1] if len(y_pred.shape) > 1 else y_pred
            
            # Calculate quantile loss
            error = y_true - y_pred_q
            loss = tf.maximum(q * error, (q - 1) * error)
            losses.append(tf.reduce_mean(loss))
        
        return tf.reduce_mean(losses)

class RankingLoss(tf.keras.losses.Loss):
    """
    Ranking loss for relative performance prediction
    Useful for portfolio selection and stock ranking
    """
    
    def __init__(self, margin=0.1, name='ranking_loss'):
        super(RankingLoss, self).__init__(name=name)
        self.margin = margin
        
    def call(self, y_true, y_pred):
        """
        Calculate ranking loss
        Ensures that stocks with higher true returns get higher predicted returns
        """
        batch_size = tf.shape(y_true)[0]
        
        # Create pairwise differences
        y_true_expanded = tf.expand_dims(y_true, axis=1)  # (batch, 1, features)
        y_pred_expanded = tf.expand_dims(y_pred, axis=1)
        
        y_true_diff = y_true_expanded - tf.transpose(y_true_expanded, [1, 0, 2])  # (batch, batch, features)
        y_pred_diff = y_pred_expanded - tf.transpose(y_pred_expanded, [1, 0, 2])
        
        # Only consider pairs where true difference is significant
        mask = tf.abs(y_true_diff) > self.margin
        
        # Calculate hinge loss for ranking
        ranking_loss = tf.maximum(0.0, self.margin - y_pred_diff * tf.sign(y_true_diff))
        
        # Apply mask and average
        masked_loss = tf.where(mask, ranking_loss, 0.0)
        
        return tf.reduce_mean(masked_loss)

class AdaptiveLoss(tf.keras.losses.Loss):
    """
    Adaptive loss that changes weights based on market conditions
    """
    
    def __init__(self, base_loss='huber', name='adaptive_loss'):
        super(AdaptiveLoss, self).__init__(name=name)
        self.base_loss = base_loss
        self.volatility_threshold = 0.02  # 2% volatility threshold
        
    def call(self, y_true, y_pred):
        """Adapt loss based on market volatility"""
        # Calculate market volatility
        returns = (y_true[:, 1:] - y_true[:, :-1]) / y_true[:, :-1]
        volatility = tf.math.reduce_std(returns, axis=-1, keepdims=True)
        
        # Base loss
        if self.base_loss == 'huber':
            base_loss = tf.keras.losses.huber(y_true, y_pred)
        else:
            base_loss = tf.keras.losses.mse(y_true, y_pred)
        
        # Adaptive weight based on volatility
        # Higher volatility -> more emphasis on directional accuracy
        # Lower volatility -> more emphasis on magnitude accuracy
        volatility_weight = tf.clip_by_value(volatility / self.volatility_threshold, 0.1, 2.0)
        
        # Calculate directional loss
        direction_loss = DirectionalAccuracyLoss()(y_true, y_pred)
        
        # Combine losses with adaptive weighting
        adaptive_loss = (2.0 - volatility_weight) * base_loss + volatility_weight * direction_loss
        
        return adaptive_loss

def get_advanced_loss(loss_type='multi_objective', **kwargs):
    """
    Factory function to get advanced loss functions
    
    Args:
        loss_type (str): Type of loss function
        **kwargs: Additional parameters for the loss function
        
    Returns:
        tf.keras.losses.Loss: The requested loss function
    """
    
    loss_functions = {
        'directional': DirectionalAccuracyLoss,
        'sharpe': SharpeRatioLoss,
        'multi_objective': MultiObjectiveLoss,
        'trading_profit': TradingProfitLoss,
        'quantile': QuantileLoss,
        'ranking': RankingLoss,
        'adaptive': AdaptiveLoss
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_type](**kwargs)

# Custom metrics for evaluation
class DirectionalAccuracyMetric(tf.keras.metrics.Metric):
    """Metric to track directional accuracy during training"""
    
    def __init__(self, name='directional_accuracy', **kwargs):
        super(DirectionalAccuracyMetric, self).__init__(name=name, **kwargs)
        self.correct_directions = self.add_weight(name='correct_directions', initializer='zeros')
        self.total_predictions = self.add_weight(name='total_predictions', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculate directions
        true_direction = tf.sign(y_true[:, 1:] - y_true[:, :-1]) if y_true.shape[1] > 1 else tf.sign(y_true)
        pred_direction = tf.sign(y_pred[:, 1:] - y_pred[:, :-1]) if y_pred.shape[1] > 1 else tf.sign(y_pred)
        
        # Count correct predictions
        correct = tf.cast(true_direction * pred_direction > 0, tf.float32)
        
        self.correct_directions.assign_add(tf.reduce_sum(correct))
        self.total_predictions.assign_add(tf.cast(tf.size(correct), tf.float32))
        
    def result(self):
        return self.correct_directions / (self.total_predictions + 1e-7)
    
    def reset_state(self):
        self.correct_directions.assign(0.0)
        self.total_predictions.assign(0.0)

class SharpeRatioMetric(tf.keras.metrics.Metric):
    """Metric to track Sharpe ratio during training"""
    
    def __init__(self, risk_free_rate=0.02, name='sharpe_ratio', **kwargs):
        super(SharpeRatioMetric, self).__init__(name=name, **kwargs)
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        self.returns_sum = self.add_weight(name='returns_sum', initializer='zeros')
        self.returns_squared_sum = self.add_weight(name='returns_squared_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculate returns
        true_returns = (y_true[:, 1:] - y_true[:, :-1]) / y_true[:, :-1]
        pred_returns = (y_pred[:, 1:] - y_pred[:, :-1]) / y_pred[:, :-1]
        
        # Calculate trading signals and strategy returns
        signals = tf.sign(pred_returns)
        strategy_returns = signals * true_returns - self.risk_free_rate
        
        # Update running statistics
        self.returns_sum.assign_add(tf.reduce_sum(strategy_returns))
        self.returns_squared_sum.assign_add(tf.reduce_sum(tf.square(strategy_returns)))
        self.count.assign_add(tf.cast(tf.size(strategy_returns), tf.float32))
        
    def result(self):
        mean_return = self.returns_sum / (self.count + 1e-7)
        variance = (self.returns_squared_sum / (self.count + 1e-7)) - tf.square(mean_return)
        std_return = tf.sqrt(tf.maximum(variance, 1e-7))
        
        return mean_return / (std_return + 1e-7)
    
    def reset_state(self):
        self.returns_sum.assign(0.0)
        self.returns_squared_sum.assign(0.0)
        self.count.assign(0.0)

def get_advanced_metrics():
    """Get list of advanced metrics for model compilation"""
    return [
        DirectionalAccuracyMetric(),
        SharpeRatioMetric(),
        'mae',
        'mse',
        'mape'
    ]

if __name__ == "__main__":
    # Test the loss functions
    print("🧪 Testing advanced loss functions...")
    
    # Create dummy data
    y_true = tf.random.normal((32, 5))  # 32 samples, 5-day forecast
    y_pred = tf.random.normal((32, 5))
    
    # Test each loss function
    losses = {
        'Directional Accuracy': DirectionalAccuracyLoss(),
        'Sharpe Ratio': SharpeRatioLoss(),
        'Multi-Objective': MultiObjectiveLoss(),
        'Trading Profit': TradingProfitLoss(),
        'Quantile': QuantileLoss(),
        'Ranking': RankingLoss(),
        'Adaptive': AdaptiveLoss()
    }
    
    print("\nLoss function test results:")
    print("-" * 40)
    
    for name, loss_fn in losses.items():
        try:
            loss_value = loss_fn(y_true, y_pred)
            print(f"{name}: {loss_value.numpy():.6f}")
        except Exception as e:
            print(f"{name}: Error - {str(e)}")
    
    # Test metrics
    print("\nMetrics test results:")
    print("-" * 40)
    
    dir_metric = DirectionalAccuracyMetric()
    dir_metric.update_state(y_true, y_pred)
    print(f"Directional Accuracy: {dir_metric.result().numpy():.4f}")
    
    sharpe_metric = SharpeRatioMetric()
    sharpe_metric.update_state(y_true, y_pred)
    print(f"Sharpe Ratio: {sharpe_metric.result().numpy():.4f}")
    
    print("\n✅ All tests completed!")