#!/usr/bin/env python3
"""
Debug script to test Binance symbol validation
"""

import requests
import json

def test_binance_symbols():
    """Test Binance exchangeInfo API and check EUR pairs"""
    base_url = "https://api.binance.com/api/v3"
    eur_pairs = ['BTCEUR', 'ETHEUR', 'ADAEUR', 'DOTEUR', 'LINKEUR']
    
    print("Testing Binance exchangeInfo API...")
    
    try:
        response = requests.get(f"{base_url}/exchangeInfo", timeout=10)
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"API Error: {response.text}")
            return
        
        data = response.json()
        all_symbols = [s['symbol'] for s in data.get('symbols', [])]
        
        print(f"Total symbols available: {len(all_symbols)}")
        
        # Check EUR pairs specifically
        print("\nChecking EUR pairs:")
        for pair in eur_pairs:
            if pair in all_symbols:
                print(f"✓ {pair} - VALID")
            else:
                print(f"✗ {pair} - INVALID")
        
        # Find all EUR pairs
        eur_symbols = [s for s in all_symbols if s.endswith('EUR')]
        print(f"\nAll available EUR pairs ({len(eur_symbols)}):")
        for symbol in sorted(eur_symbols)[:20]:  # Show first 20
            print(f"  {symbol}")
        if len(eur_symbols) > 20:
            print(f"  ... and {len(eur_symbols) - 20} more")
        
        # Check if symbols are active
        print("\nChecking symbol status:")
        for pair in eur_pairs:
            symbol_info = next((s for s in data['symbols'] if s['symbol'] == pair), None)
            if symbol_info:
                status = symbol_info.get('status', 'UNKNOWN')
                print(f"  {pair}: {status}")
            else:
                print(f"  {pair}: NOT FOUND")
                
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_binance_symbols()