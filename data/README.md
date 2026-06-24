# Intraday Data

Place raw market data under `data/raw/`. That directory is git-ignored and should not be committed.

Expected CSV columns:

```text
timestamp,symbol,close,volume
```

Optional columns:

```text
open,high,low,bid,ask,mid
```

The calibration command filters one symbol, aggregates observations into `--bar-minutes` time buckets, and estimates empirical volume and volatility profiles for the existing simulator.

Download an intraday sample with yfinance:

```bash
python experiments/download_yfinance_intraday.py \
  --symbol AAPL \
  --interval 5m \
  --period 60d \
  --output data/raw/AAPL_5m_60d.csv
```

Then calibrate from the downloaded CSV:

```bash
python experiments/calibrate_from_intraday.py \
  --input data/raw/AAPL_5m_60d.csv \
  --symbol AAPL \
  --bar-minutes 5 \
  --target-order-participation 0.05 \
  --max-participation-rate 0.10 \
  --output-config configs/calibrated/AAPL_5min.yaml
```
