import os
import yfinance as yf
import pandas as pd
import polars as pl
from tqdm import tqdm
from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the date range for data download
START_DATE = "2021-04-25"
END_DATE = "2023-08-15"

def download_one_crypto(symbol: str) -> Union[None, pl.DataFrame]:
    """
    Downloads historical data for a single cryptocurrency symbol using yfinance.

    Parameters:
        symbol (str): The cryptocurrency ticker symbol (e.g., 'BTC-USD').

    Returns:
        pl.DataFrame or None: A Polars DataFrame with the data and a 'symbol' column, or None if failed.
    """
    try:
        # Download data using yfinance
        data = yf.download(
            tickers=[symbol],
            start=START_DATE,
            end=END_DATE,
            progress=False,
            interval="1d",  # Daily data
            group_by='ticker',
            auto_adjust=False
        )
        
        # Check if data is empty
        if data.empty:
            return None
        
        # Reset index to bring the date into a column
        data.reset_index(inplace=True)
        
        # If multiple tickers are downloaded, select the relevant one
        if isinstance(data.columns, pd.MultiIndex):
            data = data[symbol].reset_index()
        
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return None
    
    # Convert to Polars DataFrame
    try:
        pl_df = pl.from_pandas(data)
    except Exception as e:
        print(f"Error converting {symbol} to Polars DataFrame: {e}")
        return None
    
    # Add the symbol as a column
    pl_df = pl_df.with_columns(
        pl.lit(symbol).alias("symbol")
    )
    
    return pl_df

def main():
    """
    Main function to orchestrate the download of cryptocurrency data.
    """
    # Path to the CSV file containing cryptocurrency symbols
    crypto_csv_path = os.path.join("data", "02_intermediate", "parsed_crypto.csv")
    
    # Read the list of cryptocurrency symbols
    try:
        crypto_data = pl.read_csv(
            crypto_csv_path,
            try_parse_dates=True
        ).drop_nulls()
    except Exception as e:
        print(f"Error reading CSV file {crypto_csv_path}: {e}")
        return
    
    # Extract unique symbols (ensure they are in the correct format for yfinance)
    unique_symbols = crypto_data.select(pl.col("symbol").unique())["symbol"].to_list()
    
    # Initialize lists to store results
    downloaded_dfs = []
    failed_downloaded_symbols = []
    counter = 0
    
    # Define the number of concurrent threads (adjust based on your system and API limits)
    max_workers = 5
    
    # Start concurrent downloading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        futures = {
            executor.submit(download_one_crypto, symbol): symbol
            for symbol in unique_symbols
        }
        
        # Iterate over completed futures with a progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading Cryptos"):
            symbol = futures[future]
            try:
                df = future.result()
                if df is not None:
                    downloaded_dfs.append(df)
                else:
                    failed_downloaded_symbols.append(symbol)
                    counter += 1
                    if counter % 10 == 0:
                        print(f"Failed to download {counter} symbols")
            except Exception as e:
                print(f"Unhandled exception for {symbol}: {e}")
                failed_downloaded_symbols.append(symbol)
                counter += 1
                if counter % 10 == 0:
                    print(f"Failed to download {counter} symbols")
    
    # Concatenate all successfully downloaded DataFrames
    if downloaded_dfs:
        try:
            combined_df = pl.concat(downloaded_dfs)
            # Ensure the output directory exists
            output_dir = os.path.join("data", "02_intermediate")
            os.makedirs(output_dir, exist_ok=True)
            # Save the combined data as a Parquet file
            combined_df.write_parquet(os.path.join(output_dir, "crypto_data.parquet"))
            print(f"Successfully saved crypto data to {os.path.join(output_dir, 'crypto_data.parquet')}")
        except Exception as e:
            print(f"Error saving combined DataFrame: {e}")
    else:
        print("No data downloaded successfully.")
    
    # Save the list of failed symbols to a text file
    if failed_downloaded_symbols:
        try:
            with open(os.path.join(output_dir, "failed_crypto_symbols.txt"), "w") as f:
                f.write("\n".join(failed_downloaded_symbols))
            print(f"List of failed symbols saved to {os.path.join(output_dir, 'failed_crypto_symbols.txt')}")
        except Exception as e:
            print(f"Error saving failed symbols: {e}")
    else:
        print("All symbols downloaded successfully.")

if __name__ == "__main__":
    main()
