"""
ADAPTIVE TIER SYSTEM
Performance-based strategy degradation across all brokers (Binance, Alpaca, Kraken Pro)
When Tier 1 strategies fail, automatically activate Tier 2, then Tier 3 fallbacks
"""

import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

# Import unified risk configuration
from core.risk_config import (
    TP_PCT, SL_PCT, CONFIDENCE_MIN, MAX_POSITION_PCT, DAILY_TRADES_MAX
)

logger = logging.getLogger(__name__)

class PerformanceTier(Enum):
    """Strategy performance tiers - degradation system"""
    TIER_1_AGGRESSIVE = "tier_1_aggressive"      # Best performance - max risk/reward
    TIER_2_CONSERVATIVE = "tier_2_conservative"  # Moderate performance - reduced risk  
    TIER_3_SURVIVAL = "tier_3_survival"          # Poor performance - minimal risk
    TIER_4_EMERGENCY = "tier_4_emergency"        # Critical - emergency protocols

@dataclass
class TierConfiguration:
    """Tier-specific trading parameters"""
    tier: PerformanceTier
    confidence_threshold: float     # Minimum confidence required
    position_size_multiplier: float # Position sizing factor
    tp_multiplier: float           # Take profit adjustment
    sl_multiplier: float           # Stop loss adjustment  
    max_daily_trades: int          # Trade frequency limit
    strategy_types: List[str]      # Allowed strategy types
    description: str               # Tier description

@dataclass  
class BrokerPerformanceMetrics:
    """Track broker-specific performance"""
    broker_name: str
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    avg_confidence: float = 0.0
    recent_win_rate: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    last_profitable_trade: float = 0
    consecutive_losses: int = 0
    performance_score: float = 0.5
    current_tier: PerformanceTier = PerformanceTier.TIER_1_AGGRESSIVE

class AdaptiveTierSystem:
    """Manages performance-based tier degradation across all brokers"""
    
    def __init__(self):
        self.tiers = self._initialize_tiers()
        self.broker_performance: Dict[str, BrokerPerformanceMetrics] = {}
        self.performance_history_path = Path("tmp/tier_performance_history.json")
        self.tier_change_cooldown = 300  # 5-minute cooldown between tier changes
        self.last_tier_change = {}
        
        # Initialize broker tracking
        for broker in ["binance", "alpaca", "kraken_pro"]:
            self.broker_performance[broker] = BrokerPerformanceMetrics(broker_name=broker)
        
        # Load historical performance
        self._load_performance_history()
        
        logger.info("🎯 Adaptive Tier System initialized")
        logger.info(f"   Brokers tracked: {list(self.broker_performance.keys())}")
        
    def _initialize_tiers(self) -> Dict[PerformanceTier, TierConfiguration]:
        """Initialize performance tier configurations"""
        return {
            PerformanceTier.TIER_1_AGGRESSIVE: TierConfiguration(
                tier=PerformanceTier.TIER_1_AGGRESSIVE,
                confidence_threshold=CONFIDENCE_MIN,        # 75% standard
                position_size_multiplier=1.0,              # Full position sizes
                tp_multiplier=1.0,                          # Full 7.0% TP
                sl_multiplier=1.0,                          # Full 0.8% SL
                max_daily_trades=min(DAILY_TRADES_MAX or 100, 100),  # 100 trades/day max
                strategy_types=["momentum", "breakout", "scalping", "arbitrage", "high_frequency"],
                description="Optimal performance - full aggressive parameters"
            ),
            PerformanceTier.TIER_2_CONSERVATIVE: TierConfiguration(
                tier=PerformanceTier.TIER_2_CONSERVATIVE,
                confidence_threshold=0.80,                  # Higher confidence required
                position_size_multiplier=0.75,             # Reduced position sizes
                tp_multiplier=0.85,                         # Reduced TP to 5.95%
                sl_multiplier=1.25,                         # Tighter SL to 1.0%
                max_daily_trades=50,                        # Reduced trade frequency
                strategy_types=["momentum", "mean_reversion", "trend_following"],
                description="Conservative mode - reduced risk parameters"
            ),
            PerformanceTier.TIER_3_SURVIVAL: TierConfiguration(
                tier=PerformanceTier.TIER_3_SURVIVAL,
                confidence_threshold=0.85,                  # Very high confidence required
                position_size_multiplier=0.50,             # Half position sizes
                tp_multiplier=0.70,                         # Reduced TP to 4.9%
                sl_multiplier=1.5,                          # Tighter SL to 1.2%
                max_daily_trades=20,                        # Minimal trading
                strategy_types=["mean_reversion", "support_resistance"],
                description="Survival mode - capital preservation focus"
            ),
            PerformanceTier.TIER_4_EMERGENCY: TierConfiguration(
                tier=PerformanceTier.TIER_4_EMERGENCY,
                confidence_threshold=0.90,                  # Extreme confidence required
                position_size_multiplier=0.25,             # Minimal position sizes
                tp_multiplier=0.50,                         # Very conservative TP 3.5%
                sl_multiplier=2.0,                          # Very tight SL 1.6%
                max_daily_trades=5,                         # Emergency trading only
                strategy_types=["mean_reversion"],          # Only safest strategy
                description="Emergency mode - stop-loss minimization"
            )
        }
    
    def evaluate_broker_performance(self, broker_name: str) -> PerformanceTier:
        """Evaluate broker performance and determine appropriate tier"""
        if broker_name not in self.broker_performance:
            logger.warning(f"⚠️ Unknown broker: {broker_name}")
            return PerformanceTier.TIER_1_AGGRESSIVE
            
        metrics = self.broker_performance[broker_name]
        
        # Calculate performance score (0.0 to 1.0)
        score = 0.0
        
        # Win rate component (40% weight)
        if metrics.total_trades > 0:
            win_rate = metrics.winning_trades / metrics.total_trades
            score += win_rate * 0.4
        else:
            score += 0.5 * 0.4  # Neutral if no trades
        
        # PnL component (30% weight)  
        if metrics.total_pnl > 0:
            score += min(1.0, metrics.total_pnl / 10000) * 0.3  # Normalize to $10K
        elif metrics.total_pnl < 0:
            score -= min(1.0, abs(metrics.total_pnl) / 5000) * 0.3  # Penalty for losses
            
        # Drawdown component (20% weight)
        if metrics.max_drawdown > 0:
            drawdown_penalty = min(1.0, metrics.max_drawdown / 0.20)  # 20% max expected
            score += (1.0 - drawdown_penalty) * 0.2
        else:
            score += 0.2
            
        # Consecutive losses penalty (10% weight)
        if metrics.consecutive_losses > 0:
            loss_penalty = min(1.0, metrics.consecutive_losses / 10)  # 10 losses = full penalty
            score += (1.0 - loss_penalty) * 0.1
        else:
            score += 0.1
        
        # Update performance score
        metrics.performance_score = max(0.0, min(1.0, score))
        
        # Determine tier based on performance score
        if score >= 0.75:
            return PerformanceTier.TIER_1_AGGRESSIVE
        elif score >= 0.50:
            return PerformanceTier.TIER_2_CONSERVATIVE  
        elif score >= 0.25:
            return PerformanceTier.TIER_3_SURVIVAL
        else:
            return PerformanceTier.TIER_4_EMERGENCY
    
    def update_trade_result(self, broker_name: str, profit: float, confidence: float, 
                          was_winner: bool) -> Optional[PerformanceTier]:
        """Update broker performance with trade result"""
        if broker_name not in self.broker_performance:
            logger.warning(f"⚠️ Unknown broker for trade update: {broker_name}")
            return None
            
        metrics = self.broker_performance[broker_name]
        previous_tier = metrics.current_tier
        
        # Update metrics
        metrics.total_trades += 1
        if was_winner:
            metrics.winning_trades += 1
            metrics.consecutive_losses = 0
            metrics.last_profitable_trade = time.time()
        else:
            metrics.consecutive_losses += 1
            
        metrics.total_pnl += profit
        metrics.avg_confidence = ((metrics.avg_confidence * (metrics.total_trades - 1)) + confidence) / metrics.total_trades
        
        # Calculate recent win rate (last 20 trades)
        if metrics.total_trades >= 20:
            recent_trades = min(20, metrics.total_trades)
            recent_winners = metrics.winning_trades if metrics.total_trades == recent_trades else \
                           max(0, metrics.winning_trades - (metrics.total_trades - recent_trades))
            metrics.recent_win_rate = recent_winners / recent_trades
        else:
            metrics.recent_win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0.0
        
        # Update drawdown
        if profit < 0:
            metrics.current_drawdown += abs(profit)
            metrics.max_drawdown = max(metrics.max_drawdown, metrics.current_drawdown)
        else:
            metrics.current_drawdown = max(0, metrics.current_drawdown - profit)
        
        # Evaluate new tier
        new_tier = self.evaluate_broker_performance(broker_name)
        
        # Check cooldown before tier change
        current_time = time.time()
        last_change = self.last_tier_change.get(broker_name, 0)
        
        if current_time - last_change >= self.tier_change_cooldown or new_tier != metrics.current_tier:
            metrics.current_tier = new_tier
            
            if new_tier != previous_tier:
                self.last_tier_change[broker_name] = current_time
                logger.warning(f"🎯 TIER CHANGE: {broker_name.upper()} {previous_tier.value} → {new_tier.value}")
                logger.warning(f"   Performance Score: {metrics.performance_score:.3f}")
                logger.warning(f"   Win Rate: {metrics.recent_win_rate:.1%} | PnL: ${metrics.total_pnl:.2f}")
                
        # Save performance history
        self._save_performance_history()
        
        return new_tier if new_tier != previous_tier else None
    
    def get_tier_parameters(self, broker_name: str, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get tier-adjusted parameters for a trading signal"""
        if broker_name not in self.broker_performance:
            logger.warning(f"⚠️ Unknown broker for parameters: {broker_name}")
            return signal_data  # Return unchanged
            
        current_tier = self.broker_performance[broker_name].current_tier
        tier_config = self.tiers[current_tier]
        
        # Apply tier adjustments
        adjusted_params = signal_data.copy()
        
        # Adjust confidence threshold
        original_confidence = adjusted_params.get('confidence', CONFIDENCE_MIN)
        if original_confidence < tier_config.confidence_threshold:
            logger.info(f"🚫 Signal FILTERED by {broker_name} {current_tier.value}: "
                       f"confidence {original_confidence:.1%} < {tier_config.confidence_threshold:.1%}")
            return {}  # Return empty dict instead of None
        
        # Adjust position sizing
        if 'position_size' in adjusted_params:
            adjusted_params['position_size'] *= tier_config.position_size_multiplier
            
        # Adjust profit targets and stop losses (for Kraken Pro specifically)
        if broker_name == "kraken_pro":
            adjusted_params['tp_pct'] = TP_PCT * tier_config.tp_multiplier
            adjusted_params['sl_pct'] = SL_PCT * tier_config.sl_multiplier
            
        # Add tier metadata
        adjusted_params['tier_applied'] = current_tier.value
        adjusted_params['tier_description'] = tier_config.description
        adjusted_params['performance_score'] = self.broker_performance[broker_name].performance_score
        
        return adjusted_params
    
    def get_broker_status(self, broker_name: str) -> Dict[str, Any]:
        """Get comprehensive broker performance status"""
        if broker_name not in self.broker_performance:
            return {"error": f"Unknown broker: {broker_name}"}
            
        metrics = self.broker_performance[broker_name]
        tier_config = self.tiers[metrics.current_tier]
        
        return {
            'broker_name': broker_name,
            'current_tier': metrics.current_tier.value,
            'tier_description': tier_config.description,
            'performance_score': round(metrics.performance_score, 3),
            'total_trades': metrics.total_trades,
            'win_rate': round(metrics.recent_win_rate, 3) if metrics.total_trades > 0 else 0.0,
            'total_pnl': round(metrics.total_pnl, 2),
            'current_drawdown': round(metrics.current_drawdown, 2),
            'max_drawdown': round(metrics.max_drawdown, 2),
            'consecutive_losses': metrics.consecutive_losses,
            'avg_confidence': round(metrics.avg_confidence, 3),
            'tier_parameters': {
                'confidence_threshold': tier_config.confidence_threshold,
                'position_multiplier': tier_config.position_size_multiplier,
                'tp_multiplier': tier_config.tp_multiplier,
                'sl_multiplier': tier_config.sl_multiplier,
                'max_daily_trades': tier_config.max_daily_trades,
                'allowed_strategies': tier_config.strategy_types
            }
        }
    
    def get_all_brokers_status(self) -> Dict[str, Any]:
        """Get status for all tracked brokers"""
        return {
            broker: self.get_broker_status(broker) 
            for broker in self.broker_performance.keys()
        }
    
    def _save_performance_history(self):
        """Save performance metrics to disk"""
        try:
            history_data = {}
            for broker, metrics in self.broker_performance.items():
                history_data[broker] = asdict(metrics)
                
            self.performance_history_path.parent.mkdir(exist_ok=True)
            with open(self.performance_history_path, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"⚠️ Error saving tier performance history: {e}")
    
    def _load_performance_history(self):
        """Load performance metrics from disk"""
        try:
            if self.performance_history_path.exists():
                with open(self.performance_history_path, 'r') as f:
                    history_data = json.load(f)
                    
                for broker, data in history_data.items():
                    if broker in self.broker_performance:
                        # Restore metrics
                        metrics = self.broker_performance[broker]
                        metrics.total_trades = data.get('total_trades', 0)
                        metrics.winning_trades = data.get('winning_trades', 0)
                        metrics.total_pnl = data.get('total_pnl', 0.0)
                        metrics.avg_confidence = data.get('avg_confidence', 0.0)
                        metrics.recent_win_rate = data.get('recent_win_rate', 0.0)
                        metrics.current_drawdown = data.get('current_drawdown', 0.0)
                        metrics.max_drawdown = data.get('max_drawdown', 0.0)
                        metrics.consecutive_losses = data.get('consecutive_losses', 0)
                        metrics.performance_score = data.get('performance_score', 0.5)
                        
                        # Restore tier
                        tier_str = data.get('current_tier', 'TIER_1_AGGRESSIVE')
                        try:
                            metrics.current_tier = PerformanceTier(tier_str)
                        except ValueError:
                            metrics.current_tier = PerformanceTier.TIER_1_AGGRESSIVE
                        
                logger.info("✅ Performance history loaded from disk")
        except Exception as e:
            logger.warning(f"⚠️ Error loading performance history: {e}")

# Global tier system instance
tier_system = AdaptiveTierSystem()

def get_tier_system() -> AdaptiveTierSystem:
    """Get global tier system instance"""
    return tier_system