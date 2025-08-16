# Model Artifact Structure & Feature Persistence
## Enhanced Training Pipeline Implementation

This document describes the comprehensive model artifact structure and feature persistence system implemented across the Bot_kilo trading pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Feature Index Persistence](#feature-index-persistence)
4. [Model-Specific Implementations](#model-specific-implementations)
5. [BaseModelAdapter System](#basemodeladapter-system)
6. [Path Alignment](#path-alignment)
7. [Usage Examples](#usage-examples)

---

## Overview

### Key Improvements

✅ **Standardized Directory Structure**: All models follow `models/{model_type}/{symbol}/{run_id}/` pattern  
✅ **Feature Index Persistence**: All models save and restore feature names and selected feature indices  
✅ **Consistent Artifact Management**: Uniform interface across GRU, LightGBM, and PPO models  
✅ **Path Alignment**: Training saves and trading loads use identical directory structures  
✅ **Automatic Symlink Creation**: "Latest" symlinks for easy model discovery  
✅ **Enhanced Metadata**: Rich model metadata including creation time, symbol, and feature information  

### Problems Solved

- **Feature Shape Mismatches**: Models now persist exact feature information used during training
- **Path Inconsistencies**: Unified directory structure eliminates path mismatches
- **Missing Feature Context**: Complete feature name and index preservation
- **Manual Model Discovery**: Automatic "latest" symlink creation

---

## Directory Structure

### Standardized Layout

```
models/
├── gru/
│   └── {symbol}/
│       ├── latest -> 20240816_142301/  # Symlink to most recent
│       ├── 20240816_142301/
│       │   ├── metadata.json           # Enhanced model metadata
│       │   ├── model.pth              # GRU model weights
│       │   ├── preprocessor.pkl       # Feature preprocessor
│       │   └── config.json            # Model configuration
│       └── 20240815_093045/
├── lightgbm/
│   └── {symbol}/
│       ├── latest -> 20240816_143502/
│       ├── 20240816_143502/
│       │   ├── metadata.json
│       │   ├── model.pkl              # LightGBM model
│       │   └── feature_importance.parquet
│       └── 20240815_102314/
└── ppo/
    └── {symbol}/
        ├── latest -> 20240816_144023/
        ├── 20240816_144023/
        │   ├── metadata.json
        │   ├── model.zip              # PPO model
        │   ├── model_metadata.json   # PPO-specific metadata
        │   └── model_vecnormalize.pkl # VecNormalize stats
        └── 20240815_111247/
```

### Metadata Structure

Each model directory contains a comprehensive `metadata.json`:

```json
{
  "model_type": "gru",
  "run_id": "20240816_142301",
  "symbol": "BTCEUR",
  "created_at": "2024-08-16T14:23:01.234567",
  "config": { /* model-specific config */ },
  "is_fitted": true,
  "feature_names": ["feature_0", "feature_1", "..."],
  "selected_features": [0, 1, 5, 8, 12],  # Selected feature indices
  "feature_count": 114,
  "input_size_actual": 5,                  # Actual input size after selection
  "training_config": { /* training parameters */ }
}
```

---

## Feature Index Persistence

### The Problem

Previously, models would fail during inference due to:
- Feature shape mismatches between training and inference
- Missing feature selection information
- Inconsistent feature ordering
- No feature name preservation

### The Solution

All trainers now implement **comprehensive feature tracking**:

```python
class EnhancedTrainer:
    def __init__(self, config):
        # Feature tracking attributes
        self.feature_names = []           # Names of all features
        self.selected_features = None     # Indices of selected features
        self.feature_count = None         # Total feature count
        self.input_size = None           # Actual input size
    
    def train(self, X, y, feature_names=None, selected_features=None):
        # Store feature information
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self.selected_features = selected_features
        self.feature_count = len(self.feature_names)
        self.input_size = len(selected_features) if selected_features else len(self.feature_names)
```

### Feature Consistency Flow

1. **Training**: Features and selection indices stored in model
2. **Saving**: Feature metadata persisted to disk
3. **Loading**: Feature information restored from artifacts
4. **Inference**: Features automatically aligned and subsetted

---

## Model-Specific Implementations

### GRU Trainer Enhancements

**Enhanced Save Method**:
```python
def save_model(self, filepath: str, symbol: Optional[str] = None):
    save_data = {
        'model_state_dict': self.best_model_state or self.model.state_dict(),
        'model_config': { /* architecture config */ },
        # NEW: Feature persistence
        'feature_names': self.feature_names,
        'selected_features': self.selected_features,
        'feature_count': self.feature_count,
        'input_size_actual': self.input_size,
        'symbol': symbol,
        'created_at': datetime.now().isoformat(),
        'model_type': 'gru'
    }
    torch.save(save_data, filepath)
```

**Enhanced Load Method**:
```python
@classmethod
def load_model(cls, filepath: str, config: Dict[str, Any]) -> 'GRUTrainer':
    checkpoint = torch.load(filepath, map_location='cpu')
    trainer = cls(config)
    
    # Restore feature information
    trainer.feature_names = checkpoint.get('feature_names', [])
    trainer.selected_features = checkpoint.get('selected_features', None)
    trainer.feature_count = checkpoint.get('feature_count', None)
    trainer.input_size = checkpoint.get('input_size_actual', None)
    
    # Build and load model
    trainer.build_model(checkpoint['model_config']['input_size'])
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    
    return trainer
```

### LightGBM Trainer Enhancements

**Feature Selection During Training**:
```python
def train(self, X_train, y_train, X_val, y_val, 
          feature_names=None, selected_features=None):
    # Store feature information
    self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
    self.selected_features = selected_features
    
    # Apply feature selection if provided
    if selected_features is not None:
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.iloc[:, selected_features]
            self.feature_names = [self.feature_names[i] for i in selected_features]
        else:
            X_train = X_train[:, selected_features]
```

**Enhanced Metadata Persistence**:
```python
def save_model(self, filepath: str, symbol: Optional[str] = None):
    model_data = {
        'model': self.model,
        'feature_names': self.feature_names,
        'selected_features': self.selected_features,
        'feature_count': self.feature_count,
        'input_size_actual': self.input_size,
        'symbol': symbol,
        'created_at': datetime.now().isoformat(),
        'model_type': 'lightgbm',
        # ... other model data
    }
    joblib.dump(model_data, filepath)
```

### PPO Trainer Enhancements

**Observation Space Metadata**:
```python
def train(self, train_data, eval_data, feature_names=None):
    # Store environment observation space information
    if hasattr(train_env, 'observation_space'):
        self.observation_shape = train_env.observation_space.shape
        self.input_size = int(np.prod(self.observation_shape))
    
    # Store feature names (data columns)
    self.feature_names = feature_names or list(train_data.columns)
```

**Separate Metadata File**:
```python
def save_model(self, filepath: str, symbol: Optional[str] = None):
    # Save PPO model
    self.model.save(model_path)
    
    # Save additional metadata
    metadata = {
        'model_type': 'ppo',
        'feature_names': self.feature_names,
        'observation_shape': list(self.observation_shape) if self.observation_shape else None,
        'action_space_info': self.action_space_info,
        'symbol': symbol,
        'created_at': datetime.now().isoformat()
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
```

---

## BaseModelAdapter System

### Uniform Interface

The `BaseModelAdapter` provides a standardized interface across all model types:

```python
class BaseModelAdapter(ABC):
    def save(self, output_dir: str, run_id: Optional[str] = None, 
             symbol: Optional[str] = None) -> str:
        # Create standardized directory structure
        if symbol:
            model_dir = os.path.join(output_dir, self.model_type, symbol, run_id)
        else:
            model_dir = os.path.join(output_dir, self.model_type, run_id)
        
        # Save artifacts and metadata
        # Create "latest" symlink
        self._create_latest_symlink(model_dir, latest_path)
```

### Enhanced Metadata

All adapters save rich metadata:

```python
metadata = {
    'model_type': self.model_type,
    'run_id': run_id,
    'symbol': symbol,
    'created_at': datetime.now().isoformat(),
    'config': self.model_config,
    'is_fitted': self.is_fitted,
    'feature_names': getattr(self, 'feature_names', []),
    'selected_features': getattr(self, 'selected_features', None),
    'feature_count': getattr(self, 'feature_count', None)
}
```

### Automatic Symlink Creation

```python
def _create_latest_symlink(self, target_dir: str, symlink_path: str) -> None:
    """Create or update 'latest' symlink to point to the most recent model."""
    try:
        os.symlink(relative_target, symlink_path)
    except Exception:
        # Fallback: create pointer file on Windows
        with open(symlink_path + '_pointer.txt', 'w') as f:
            f.write(target_dir)
```

---

## Path Alignment

### Training Pipeline (scripts/trainer.py)

```python
# Creates: models/{model_type}/{symbol}/{run_id}/
model_dir = os.path.join(args.output_dir, model_type, symbol, run_id)
saved_path = final_adapter.save(os.path.join(args.output_dir, model_type, symbol), run_id=run_id)

# Creates symlink: models/{model_type}/{symbol}/latest -> {run_id}/
latest_path = os.path.join(args.output_dir, model_type, symbol, 'latest')
```

### Trading Pipeline (scripts/trader.py)

```python
def _find_latest_unified_artifact(self, model_type: str, symbol: str, filename: str):
    """Looks under: models/{model_type}/{symbol}/**/{filename}"""
    search_root = os.path.join(self.models_dir, model_type, symbol)
    pattern = os.path.join(search_root, '**', filename)
    files = glob.glob(pattern, recursive=True)
    return max(files, key=os.path.getmtime) if files else None
```

### Model Loading Hierarchy

1. **Walk-forward best artifacts**: `models/metadata/best_wf_{model}_{symbol}.{ext}`
2. **Latest unified artifacts**: `models/{model_type}/{symbol}/latest/{filename}`
3. **Timestamped artifacts**: `models/{model_type}/{symbol}/**/{filename}`

---

## Usage Examples

### Training with Feature Selection

```python
# Train GRU with specific features
feature_names = ['rsi_14', 'macd', 'bb_upper', 'volume_sma']
selected_features = [0, 2, 5, 8]  # Indices of important features

trainer = GRUTrainer(config)
results = trainer.train(
    X_train, y_train, X_val, y_val,
    feature_names=feature_names,
    selected_features=selected_features
)

# Save with symbol information
trainer.save_model('models/gru/BTCEUR/20240816_142301/model.pth', symbol='BTCEUR')
```

### Loading with Feature Restoration

```python
# Load model - features automatically restored
loaded_trainer = GRUTrainer.load_model(
    'models/gru/BTCEUR/20240816_142301/model.pth', 
    config
)

# Feature information available
print(f"Features: {len(loaded_trainer.feature_names)}")
print(f"Selected: {len(loaded_trainer.selected_features)}")
print(f"Input size: {loaded_trainer.input_size}")

# Prediction with automatic feature alignment
predictions = loaded_trainer.predict(new_X)
```

### Using BaseModelAdapter

```python
# Create adapter
adapter = GRUAdapter(config)

# Train
adapter.fit(X, y, train_idx, valid_idx)

# Save with standardized structure
saved_path = adapter.save(
    output_dir='./models',
    symbol='BTCEUR'
)
# Creates: ./models/gru/BTCEUR/{timestamp}/
# Creates: ./models/gru/BTCEUR/latest -> {timestamp}

# Load
adapter.load(saved_path)
```

### Trading Script Integration

```python
class UnifiedPaperTrader:
    def load_all_models(self):
        for symbol in self.symbols:
            # Uses standardized path discovery
            gru_path = (self._find_latest_best_wf('gru', symbol) or 
                       self._find_latest_unified_artifact('gru', symbol, 'model.pth'))
            
            if gru_path:
                # Loads with feature information intact
                self.models[symbol]['gru'] = GRUTrainer.load_model(gru_path, self.config)
```

---

## Migration Notes

### For Existing Models

- **Backward Compatibility**: Old models without feature metadata will still load
- **Feature Inference**: Missing feature names auto-generated as `feature_{i}`
- **Graceful Degradation**: Missing selected_features defaults to all features

### For New Deployments

- **Clean Start**: Use new standardized structure from beginning
- **Feature Planning**: Design feature selection strategy early
- **Symbol Organization**: Organize models by trading symbol

---

## Best Practices

### During Training

1. **Always provide feature_names**: Helps with debugging and model interpretability
2. **Document feature_selection**: Save selection rationale in logs
3. **Use consistent symbols**: Follow exchange naming conventions
4. **Enable artifact validation**: Check saved models load correctly

### During Deployment

1. **Monitor feature consistency**: Log feature count mismatches
2. **Use latest symlinks**: Simplifies model discovery
3. **Validate before trading**: Ensure models load and predict correctly
4. **Track model versions**: Keep deployment logs with model timestamps

---

## Troubleshooting

### Common Issues

**Feature Count Mismatch**:
```
Feature count mismatch: stored=114, model=50
```
- **Cause**: Different feature engineering between training and inference
- **Solution**: Ensure identical feature generation pipeline

**Missing Model Files**:
```
Model file not found: models/gru/BTCEUR/latest/model.pth
```
- **Cause**: Symlink creation failed or model not trained
- **Solution**: Check for `*_pointer.txt` files on Windows, retrain if needed

**Shape Mismatch During Prediction**:
```
Expected 5 features, got 114
```
- **Cause**: Feature selection not applied during inference
- **Solution**: Verify selected_features is loaded and applied correctly

### Validation Script

```python
def validate_model_artifacts(models_dir='./models'):
    """Validate all model artifacts are loadable and consistent."""
    for model_type in ['gru', 'lightgbm', 'ppo']:
        type_dir = os.path.join(models_dir, model_type)
        if os.path.exists(type_dir):
            for symbol in os.listdir(type_dir):
                symbol_dir = os.path.join(type_dir, symbol)
                latest_link = os.path.join(symbol_dir, 'latest')
                
                if os.path.exists(latest_link):
                    print(f"✓ {model_type}/{symbol}/latest exists")
                    # Try loading model
                    try:
                        if model_type == 'gru':
                            model_path = os.path.join(latest_link, 'model.pth')
                            GRUTrainer.load_model(model_path, config)
                        # ... similar for other types
                        print(f"✓ {model_type}/{symbol} loads successfully")
                    except Exception as e:
                        print(f"✗ {model_type}/{symbol} failed: {e}")
```

---

## Conclusion

The enhanced model artifact structure provides:

- **Reliability**: Consistent feature handling prevents runtime errors
- **Maintainability**: Standardized structure simplifies debugging
- **Scalability**: Easy addition of new models and symbols
- **Traceability**: Rich metadata enables model provenance tracking

This implementation ensures the trading pipeline can reliably train, save, and load models with complete feature consistency across the entire workflow.