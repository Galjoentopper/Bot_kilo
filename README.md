Bot Kilo — Crypto Trading Bot: Training and Model Transfer

This guide shows how to train models on a Linux machine and import them on your Windows machine. Keep training off Windows (it’s too slow) — only import there.

Quick overview
- Train on Linux using the provided .sh scripts
- A transfer bundle ZIP is created under models/exports/
- Copy the ZIP to Windows and import it

1) Linux (training machine)
Prerequisites
- Python 3.8+ and pip
- Linux shell with permission to execute scripts

Initial setup (run once)
1. Make scripts executable:
   chmod +x setup_training_environment.sh train_models.sh fetch_training_data.sh
2. Run setup (creates venv, installs dependencies, prepares folders):
   ./setup_training_environment.sh

(Optional) fetch or update training data
- Default (all symbols from config):
  ./fetch_training_data.sh
- Single symbol example:
  ./fetch_training_data.sh --symbol BTCEUR

Train and package models (creates transfer bundle automatically)
- Train with defaults (all configured models/symbols):
  ./train_models.sh
- Train specific symbols:
  ./train_models.sh --symbols BTCEUR ETHEUR ADAEUR
- Train only LightGBM:
  ./train_models.sh --models lightgbm
- Resume after interruption (checkpointing supported):
  ./train_models.sh --resume

Result
- When training finishes, a bundle appears at:
  models/exports/model_transfer_bundle_YYYYMMDD_HHMMSS.zip
- If your Linux machine is headless, any “Opening exports folder…” warning is safe to ignore.

2) Move bundle to Windows
- Copy the ZIP from models/exports/ on Linux to your Windows project directory (same repo folder).

3) Windows (import only — do not train here)
- Using the batch file:
  .\import_models.bat ".\model_transfer_bundle_YYYYMMDD_HHMMSS.zip"
- Or directly with Python:
  py scripts\import_models.py ".\model_transfer_bundle_YYYYMMDD_HHMMSS.zip"

Optional: Validate models (does not start trading)
- Quick validation examples:
  py scripts\validate_models.py --models-dir models --quick
  py scripts\validate_models.py --type lightgbm --models-dir models --quick

Notes
- The trainer automatically handles packaging; you don’t need extra steps.
- Do not train on the Windows machine. Only import there.