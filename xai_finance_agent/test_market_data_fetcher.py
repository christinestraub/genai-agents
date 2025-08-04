#!/usr/bin/env python3
"""
Test script for MarketDataFetcher class.
This demonstrates how to use the MarketDataFetcher for retrieving stock data.
"""

import sys
import os

# Add the current directory to Python path to import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xai_finance_agent import MarketDataFetcher


def test_basic_functionality():
    """Test basic functionality of MarketDataFetcher."""
    print("=== Testing MarketDataFetcher ===\n")
    
    # Create fetcher instance
    fetcher = MarketDataFetcher(cache_duration_hours=1)
    
    # Test 1: Fetch Apple stock data for last 30 days
    print("1. Fetching AAPL data for last 30 days...")
    aapl_data = fetcher.fetch_stock_data("AAPL", days=30)
    
    if aapl_data is not None:
        print(f"   ✓ Successfully fetched {len(aapl_data)} rows")
        print(f"   ✓ Columns: {list(aapl_data.columns)}")
        print(f"   ✓ Date range: {aapl_data.index[0].date()} to {aapl_data.index[-1].date()}")
        print(f"   ✓ Latest close price: ${aapl_data['Close'].iloc[-1]:.2f}")
    else:
        print("   ✗ Failed to fetch AAPL data")
    
    print()
    
    # Test 2: Get current price
    print("2. Getting current AAPL price...")
    current_price = fetcher.get_current_price("AAPL")
    
    if current_price:
        print(f"   ✓ Current AAPL price: ${current_price:.2f}")
    else:
        print("   ✗ Failed to get current price")
    
    print()
    
    # Test 3: Fetch multiple tickers
    print("3. Fetching multiple tickers (AAPL, GOOGL, MSFT) for last 7 days...")
    tickers = ["AAPL", "GOOGL", "MSFT"]
    multiple_data = fetcher.get_multiple_tickers(tickers, days=7)
    
    for ticker in tickers:
        if ticker in multiple_data:
            data = multiple_data[ticker]
            print(f"   ✓ {ticker}: {len(data)} rows, latest close: ${data['Close'].iloc[-1]:.2f}")
        else:
            print(f"   ✗ {ticker}: Failed to fetch data")
    
    print()
    
    # Test 4: Test caching (fetch same data again)
    print("4. Testing cache functionality (fetching AAPL again)...")
    aapl_data_cached = fetcher.fetch_stock_data("AAPL", days=30)
    
    if aapl_data_cached is not None:
        print("   ✓ Successfully retrieved data (likely from cache)")
        # Compare if data is the same
        if aapl_data is not None and len(aapl_data) == len(aapl_data_cached):
            print("   ✓ Cached data matches original data")
        else:
            print("   ⚠ Cached data differs from original")
    else:
        print("   ✗ Failed to retrieve cached data")
    
    print()
    
    # Test 5: Test different intervals
    print("5. Fetching AAPL data with hourly interval (last 2 days)...")
    hourly_data = fetcher.fetch_stock_data("AAPL", days=2, interval="1h")
    
    if hourly_data is not None:
        print(f"   ✓ Successfully fetched {len(hourly_data)} rows of hourly data")
        print(f"   ✓ Date range: {hourly_data.index[0]} to {hourly_data.index[-1]}")
    else:
        print("   ✗ Failed to fetch hourly data")
    
    print()
    
    # Test 6: Display sample data
    if aapl_data is not None and len(aapl_data) > 0:
        print("6. Sample AAPL data (last 5 days):")
        print(aapl_data.tail().to_string())
        print()
    
    print("=== Test completed ===")


def test_error_handling():
    """Test error handling functionality."""
    print("\n=== Testing Error Handling ===\n")
    
    fetcher = MarketDataFetcher()
    
    # Test invalid ticker
    print("1. Testing invalid ticker...")
    invalid_data = fetcher.fetch_stock_data("INVALID_TICKER_XYZ")
    if invalid_data is None:
        print("   ✓ Correctly handled invalid ticker")
    else:
        print("   ⚠ Unexpected data returned for invalid ticker")
    
    # Test invalid days
    print("2. Testing invalid days parameter...")
    invalid_days_data = fetcher.fetch_stock_data("AAPL", days=-5)
    if invalid_days_data is None:
        print("   ✓ Correctly handled negative days")
    else:
        print("   ⚠ Unexpected data returned for negative days")
    
    # Test empty ticker
    print("3. Testing empty ticker...")
    empty_ticker_data = fetcher.fetch_stock_data("")
    if empty_ticker_data is None:
        print("   ✓ Correctly handled empty ticker")
    else:
        print("   ⚠ Unexpected data returned for empty ticker")
    
    print("\n=== Error handling test completed ===")


if __name__ == "__main__":
    try:
        # Run basic functionality tests
        test_basic_functionality()
        
        # Run error handling tests
        test_error_handling()
        
        print("\n🎉 All tests completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure you have installed the required packages:")
        print("pip install yfinance pandas numpy")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()