import ccxt
import asyncio
import pandas as pd

async def test_ccxt():
    try:
        # Initialize exchange
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            },
            'timeout': 30000
        })
        
        # Load markets
        exchange.load_markets()
        
        # Test with a single symbol
        symbol = 'BTC/USDT'  # Use a standard symbol format
        print(f"Testing with symbol: {symbol}")
        
        # Try to fetch data
        ohlcv = exchange.fetch_ohlcv(symbol, '15m', limit=10)
        print(f"Successfully fetched {len(ohlcv)} data points")
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(df.head())
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ccxt())