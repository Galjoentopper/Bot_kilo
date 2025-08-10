import yfinance as yf
import pandas as pd

# Test getting data for BTC-EUR
df = yf.download('BTC-EUR', period="60d", interval="15m")
print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Index type:", type(df.index))
print("First few rows:")
print(df.head())

# Check if we can access the columns directly
if not df.empty:
    print("\nColumn access test:")
    try:
        print("High column:", df['High'].head())
        print("Low column:", df['Low'].head())
        print("Close column:", df['Close'].head())
    except Exception as e:
        print("Error accessing columns:", e)
        
    # Test creating new columns
    print("\nCreating new columns:")
    try:
        df['price_range'] = df['High'] - df['Low']
        print("Price range column created successfully")
        print("Price range values:", df['price_range'].head())
        
        # Test division
        df['price_range_pct'] = df['price_range'] / (df['Close'] + 1e-10)
        print("Price range pct column created successfully")
        print("Price range pct values:", df['price_range_pct'].head())
    except Exception as e:
        print("Error creating new columns:", e)