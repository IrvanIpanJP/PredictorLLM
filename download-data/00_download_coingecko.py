import os
import math
import polars as pl
import requests
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Union, Optional

# If you'd like to use the official pycoingecko client instead:
# from pycoingecko import CoinGeckoAPI
# cg = CoinGeckoAPI()

# --- USER PARAMETERS ---
START_DATE_STR = "2021-04-25"
END_DATE_STR   = "2023-08-15"
VS_CURRENCY    = "usd"  # vs_currency to fetch
MAX_WORKERS    = 5      # concurrency

# Convert date strings to Unix timestamps
start_dt = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
end_dt   = datetime.strptime(END_DATE_STR, "%Y-%m-%d")
START_TS = int(datetime.timestamp(start_dt))
END_TS   = int(datetime.timestamp(end_dt))

def coingecko_market_chart_range(
    coin_id: str, 
    vs_currency: str,
    from_ts: int,
    to_ts: int
) -> Optional[dict]:
    """
    Query the CoinGecko API for daily data between two Unix timestamps.
    
    coin_id    : e.g. 'bitcoin', 'ethereum'
    vs_currency: e.g. 'usd'
    from_ts    : Start time in seconds (Unix timestamp)
    to_ts      : End time in seconds (Unix timestamp)

    Returns:
        JSON data (dict) with fields: 'prices', 'market_caps', 'total_volumes'
        or None if something fails.
    """
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
        f"?vs_currency={vs_currency}&from={from_ts}&to={to_ts}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"[ERROR] CoinGecko request failed for '{coin_id}': {e}")
        return None


def process_coingecko_response(
    coin_id: str,
    raw_data: dict
) -> Optional[pl.DataFrame]:
    """
    Transform the raw CoinGecko response into a Polars DataFrame with columns:
    ['date', 'price', 'market_cap', 'total_volume', 'coin_id'].

    raw_data: A dictionary returned by coingecko_market_chart_range
              with keys 'prices', 'market_caps', 'total_volumes'.
    """
    try:
        # Each of these is a list of [timestamp, value] pairs
        prices      = raw_data.get("prices", [])
        market_caps = raw_data.get("market_caps", [])
        volumes     = raw_data.get("total_volumes", [])

        if not prices:
            print(f"[WARN] No price data returned for {coin_id}.")
            return None

        # We expect them all to have the same length, but let's be cautious
        n_prices = len(prices)
        n_mcaps  = len(market_caps)
        n_vols   = len(volumes)

        # Convert each list into columns
        # prices[i] = [ts_millis, price_float]
        df_prices = pl.DataFrame(
            {
                "timestamp": [p[0] for p in prices],
                "price": [p[1] for p in prices],
            }
        )

        # Some coins may not have market_cap or volumes for every row
        if n_mcaps == n_prices:
            df_mcaps = pl.DataFrame(
                {
                    "timestamp": [m[0] for m in market_caps],
                    "market_cap": [m[1] for m in market_caps],
                }
            )
            df_prices = df_prices.join(df_mcaps, on="timestamp", how="left")

        if n_vols == n_prices:
            df_vols = pl.DataFrame(
                {
                    "timestamp": [v[0] for v in volumes],
                    "total_volume": [v[1] for v in volumes],
                }
            )
            df_prices = df_prices.join(df_vols, on="timestamp", how="left")

        # Convert timestamp (ms) to datetime
        # CoinGecko returns ms since epoch, so we must do ts/1000
        df_prices = df_prices.with_columns(
            (pl.col("timestamp") / 1000).cast(pl.Int64).alias("ts_s")
        ).drop("timestamp")

        df_prices = df_prices.with_columns(
            pl.col("ts_s").apply(lambda x: datetime.utcfromtimestamp(x)).alias("date_utc")
        ).drop("ts_s")

        # Add coin_id column
        df_prices = df_prices.with_columns(
            pl.lit(coin_id).alias("coin_id")
        )
        return df_prices
    except Exception as e:
        print(f"[ERROR] Processing data for {coin_id}: {e}")
        return None


def download_one_crypto(coin_id: str) -> Union[pl.DataFrame, None]:
    """
    Downloads historical data for a single crypto (coin_id) from CoinGecko 
    using a daily range approach.

    Returns:
        pl.DataFrame with columns [date_utc, price, market_cap, total_volume, coin_id]
        or None if it fails.
    """
    raw = coingecko_market_chart_range(
        coin_id=coin_id,
        vs_currency=VS_CURRENCY,
        from_ts=START_TS,
        to_ts=END_TS
    )
    if raw is None:
        return None
    return process_coingecko_response(coin_id, raw)


def main():
    """
    Main function to orchestrate the download of cryptocurrency data from CoinGecko.
    """
    # CSV file with a column 'symbol' or 'coin_id' containing valid CoinGecko IDs
    crypto_csv_path = os.path.join("data", "02_intermediate", "parsed_crypto.csv")

    # Read the list of coin IDs from CSV
    try:
        crypto_data = pl.read_csv(crypto_csv_path).drop_nulls()
    except Exception as e:
        print(f"[ERROR] reading CSV file {crypto_csv_path}: {e}")
        return

    # IMPORTANT: Your CSV must have the correct CoinGecko IDs, not just "BTC-USD"
    # For instance, "bitcoin", "ethereum", "cardano", etc.
    # Adjust the column name if needed:
    if "coin_id" not in crypto_data.columns:
        print("[ERROR] CSV missing 'coin_id' column (CoinGecko IDs). Aborting.")
        return

    unique_coin_ids = crypto_data.select(pl.col("coin_id").unique())["coin_id"].to_list()

    downloaded_dfs = []
    failed_coins = []
    counter = 0

    # Create output directory
    output_dir = os.path.join("data", "02_intermediate")
    os.makedirs(output_dir, exist_ok=True)

    # Use concurrency, but be mindful of CoinGecko rate limits (use 1-5 workers max)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_one_crypto, cid): cid
            for cid in unique_coin_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            coin_id = futures[future]
            try:
                df = future.result()
                if df is not None and df.height > 0:
                    downloaded_dfs.append(df)
                else:
                    failed_coins.append(coin_id)
                    counter += 1
                    if counter % 10 == 0:
                        print(f"[WARN] Failed to download data for {counter} coins so far")
            except Exception as e:
                print(f"[ERROR] Unhandled exception for {coin_id}: {e}")
                failed_coins.append(coin_id)
                counter += 1
                if counter % 10 == 0:
                    print(f"[WARN] Failed to download data for {counter} coins so far")

    # Combine all successful data
    if downloaded_dfs:
        try:
            combined = pl.concat(downloaded_dfs, how="vertical")
            # Save as Parquet
            out_file = os.path.join(output_dir, "crypto_data_coingecko.parquet")
            combined.write_parquet(out_file)
            print(f"[INFO] Successfully saved combined data -> {out_file}")
        except Exception as e:
            print(f"[ERROR] Saving combined data: {e}")
    else:
        print("[INFO] No data downloaded successfully.")

    # Save failed coins to a text file
    if failed_coins:
        failed_path = os.path.join(output_dir, "failed_coins.txt")
        try:
            with open(failed_path, "w") as f:
                f.write("\n".join(failed_coins))
            print(f"[INFO] List of failed coins saved -> {failed_path}")
        except Exception as e:
            print(f"[ERROR] Saving failed coins list: {e}")
    else:
        print("[INFO] All coins downloaded successfully.")

if __name__ == "__main__":
    main()
