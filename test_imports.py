#!/usr/bin/env python3
"""
Test script to verify all imports work correctly after package installation.
"""

def test_imports():
    """Test all main imports from the src package."""
    try:
        print("Testing imports...")
        
        # Test core imports
        from src.utils.logger import setup_logging, TradingBotLogger
        print("✓ Logger imports successful")
        
        from src.data_pipeline.loader import DataLoader
        from src.data_pipeline.features import FeatureEngine
        from src.data_pipeline.preprocess import DataPreprocessor
        print("✓ Data pipeline imports successful")
        
        from src.models.gru_trainer import GRUTrainer
        from src.models.lgbm_trainer import LightGBMTrainer
        from src.models.ppo_trainer import PPOTrainer
        print("✓ Model imports successful")
        
        from src.rl_env.trading_env import TradingEnvironment
        print("✓ RL environment import successful")
        
        from src.backtesting.backtest import Backtester
        print("✓ Backtesting import successful")
        
        from src.notifier.telegram import TelegramNotifier
        print("✓ Notification import successful")
        
        print("\n🎉 All imports successful! Package installation is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)