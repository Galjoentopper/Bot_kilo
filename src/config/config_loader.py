#!/usr/bin/env python3
"""Lightweight ConfigLoader used by runtime scripts.

Bridges existing configuration assets:
- Primary runtime config: Auto-detects based on calling script type
  - Training scripts: `src/config/config_training.yaml`
  - Trading scripts: `src/config/config_trading.yaml`
  - Fallback: `config.yaml` (at project root) OR a supplied path
- Environment info (optional): `config/environment_info.yaml`

Loads YAML safely, merges configurations, and exposes
both a dict (`config`) and attribute-style access for convenience.

If the requested file does not exist, it falls back to an empty config dict
so that scripts can still start with sane defaults.
"""
from __future__ import annotations
import os
import yaml
import inspect
from pathlib import Path
from typing import Any, Dict, Optional

class ConfigLoader:
    """Simple configuration loader with minimal dependencies."""
    def __init__(self, config_path: Optional[str | os.PathLike[str]] = None, project_root: Optional[str | os.PathLike[str]] = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent.parent
        
        # Auto-detect config path if not provided
        if config_path is None:
            config_path = self._detect_config_path()
        
        self._raw_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    def _detect_config_path(self) -> str:
        """Auto-detect the appropriate config file based on calling script."""
        # Get the call stack to find the calling script
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the actual calling script (skip ConfigLoader frames)
            caller_frame = frame.f_back.f_back if frame.f_back else None
            if caller_frame:
                caller_file = caller_frame.f_code.co_filename
                caller_name = Path(caller_file).name.lower()
                
                # Check if it's a training script
                training_keywords = ['trainer', 'train', 'training']
                if any(keyword in caller_name for keyword in training_keywords):
                    return 'src/config/config_training.yaml'
                
                # Check if it's a trading script
                trading_keywords = ['trader', 'trade', 'trading']
                if any(keyword in caller_name for keyword in trading_keywords):
                    return 'src/config/config_trading.yaml'
        finally:
            del frame
        
        # Fallback to original default
        return 'config.yaml'
    
    # ------------------------------------------------------------------
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with path.open('r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                if not isinstance(data, dict):
                    return {}
                return data
        except Exception:
            return {}

    # ------------------------------------------------------------------
    def _load(self):
        # Base config: explicit path OR auto-detected path
        if self._raw_path.is_absolute():
            base_path = self._raw_path
        else:
            # Try relative to caller CWD first, then project root
            base_path = Path.cwd() / self._raw_path
            if not base_path.exists():
                base_path = self.project_root / self._raw_path
        
        base_cfg = self._load_yaml(base_path)
        
        # If we're using the old config.yaml fallback, still merge with trading config
        # Otherwise, the selected config file is already the primary one
        if str(self._raw_path) == 'config.yaml':
            # Legacy behavior: merge trading config as override
            trading_path = self.project_root / 'src' / 'config' / 'config_trading.yaml'
            trading_cfg = self._load_yaml(trading_path)
            merged = {**base_cfg, **trading_cfg}
        else:
            # New behavior: selected config is primary
            merged = base_cfg
        
        # Environment info (attach under meta if present)
        env_info_path = self.project_root / 'config' / 'environment_info.yaml'
        env_info = self._load_yaml(env_info_path)
        if env_info:
            merged.setdefault('meta', {})['environment'] = env_info

        self.config = merged

    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    # Attribute-style access convenience
    def __getattr__(self, item: str) -> Any:  # pragma: no cover - convenience
        try:
            return self.config[item]
        except KeyError as e:
            raise AttributeError(item) from e

__all__ = ["ConfigLoader"]
