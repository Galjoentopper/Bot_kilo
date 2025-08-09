# Crypto Trading Bot Fixes Plan

## Overview
This document outlines the issues identified in the crypto trading bot and the proposed fixes to resolve them. The main issues are:

1. PPO model loading failure due to incorrect file path handling
2. High number of NaN values in feature generation affecting model performance

## Issue 1: PPO Model Loading Failure

### Problem
The PPO model fails to load with the error:
```
Failed to load PPO model: Model file not found: ./models\ppo_model_20250807_132238.zip.zip
```

### Root Cause
In `src/models/ppo_trainer.py`, the `load_model` method at line 593 checks for `f"{filepath}.zip"` but the actual file path being passed already includes the `.zip` extension. This results in looking for a file with a double extension.

### Solution
Modify the `load_model` method to check if the filepath already has the `.zip` extension before appending it.

### Files to Modify
- `src/models/ppo_trainer.py`

### Code Changes
```python
@classmethod
def load_model(cls, filepath: str, config: Dict[str, Any]) -> 'PPOTrainer':
    """
    Load a trained model.
    
    Args:
        filepath: Path to the saved model
        config: Configuration dictionary
        
    Returns:
        Loaded PPOTrainer instance
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required for PPO training")
    
    # Check if filepath already has .zip extension
    if not filepath.endswith('.zip'):
        filepath_with_extension = f"{filepath}.zip"
    else:
        filepath_with_extension = filepath
    
    if not os.path.exists(filepath_with_extension):
        raise FileNotFoundError(f"Model file not found: {filepath_with_extension}")
    
    # Create trainer instance
    trainer = cls(config)
    
    # Load model
    if SB3_AVAILABLE:
        trainer.model = SB3_PPO.load(filepath)
    else:
        trainer.model = PPO()
    
    logger.info(f"PPO model loaded from {filepath}")
    return trainer
```

## Issue 2: High NaN Values in Feature Generation

### Problem
The logs show warnings about high numbers of NaN values in generated features:
```
WARNING - Found 1010 NaN values in generated features
WARNING - Found 901 NaN values in generated features
WARNING - Found 930 NaN values in generated features
WARNING - Found 1100 NaN values in generated features
WARNING - Found 1046 NaN values in generated features
```

### Root Cause
The feature generation process creates many NaN values during technical indicator calculations, particularly at the beginning of time series where rolling window calculations don't have enough data.

### Solution
Improve the NaN handling in the feature generation process by:

1. Using more robust forward/backward fill methods
2. Adding minimum data validation before calculating indicators
3. Implementing better edge case handling for technical indicators

### Files to Modify
- `src/data_pipeline/features.py`

### Code Changes
The changes should focus on improving the NaN handling in the `generate_all_features` method and the individual technical indicator calculation methods.

## Implementation Plan

### Phase 1: Fix PPO Model Loading
1. Modify `load_model` method in `src/models/ppo_trainer.py` to correctly handle file extensions
2. Test that PPO model loads correctly

### Phase 2: Improve Feature Generation
1. Review and improve NaN handling in `src/data_pipeline/features.py`
2. Add validation for minimum data requirements before calculating indicators
3. Test with sample data to verify reduced NaN values

### Phase 3: Integration Testing
1. Run the complete trading bot to verify all models load correctly
2. Verify that feature generation produces fewer NaN values
3. Test trading signals generation with improved features

## Testing Strategy

### Unit Tests
- Test PPO model loading with various file path formats
- Test feature generation with edge cases and minimal data

### Integration Tests
- Run complete trading bot cycle with all models
- Verify trading signals are generated correctly
- Check portfolio value calculations

## Expected Outcomes

1. PPO model loads successfully without double extension error
2. Reduced number of NaN values in feature generation
3. Improved model performance due to better quality features
4. More reliable trading signals