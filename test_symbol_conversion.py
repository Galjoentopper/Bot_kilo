import ccxt
import asyncio

async def test_symbol_conversion():
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
        
        # Test symbol conversion
        symbols = ["BTCEUR", "ETHEUR", "ADAEUR", "SOLEUR", "XRPEUR"]
        
        for symbol in symbols:
            # Convert symbol format from BTCEUR to BTC/EUR
            formatted_symbol = symbol.replace('EUR', '/EUR')
            print(f"Original: {symbol} -> Converted: {formatted_symbol}")
            
            # Check if the symbol exists in the exchange
            if formatted_symbol in exchange.markets:
                print(f"  Symbol {formatted_symbol} exists in exchange")
                # Try to fetch data
                try:
                    ohlcv = exchange.fetch_ohlcv(formatted_symbol, '15m', limit=5)
                    print(f"  Successfully fetched {len(ohlcv)} data points")
                except Exception as e:
                    print(f"  Error fetching data: {e}")
            else:
                print(f"  Symbol {formatted_symbol} does not exist in exchange")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_symbol_conversion())