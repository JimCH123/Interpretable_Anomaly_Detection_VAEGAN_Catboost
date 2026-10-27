"""Step 6 - Online validation: serve the detector and stream the held-out raw records to it.

Starts the FastAPI server and, once it is healthy, posts each record of
data/streaming/test_data.csv (7 normal + 3 anomalous) to /predict. Waterfall plots
of detected anomalies are written to outputs/api_plots/.
"""
import argparse
import logging
import threading
import time

import pandas as pd
import requests
import uvicorn

from anomaly_detection import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stream")


def wait_until_ready(base_url, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=1).ok:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    return False


def stream_test_data(base_url, interval):
    """Send every held-out raw record to /predict, one every ``interval`` seconds."""
    if not wait_until_ready(base_url):
        logger.error("API did not become ready, streaming aborted.")
        return
    test_data = pd.read_csv(config.STREAMING_DATA)
    for index, row in test_data.iterrows():
        raw = row.drop([config.TARGET_COL, config.TIME_COL]).values
        line = ",".join("" if pd.isna(v) else str(v) for v in raw)
        label = "anomaly" if row[config.TARGET_COL] == config.ANOMALY_LABEL else "normal"
        try:
            response = requests.post(f"{base_url}/predict", json={"line": line}, timeout=30)
            logger.info(f"Record {index} (true label: {label}) -> {response.json()}")
        except Exception as e:
            logger.error(f"Send error: {e}")
            break
        time.sleep(interval)
    logger.info(f"All {len(test_data)} records sent. The API keeps running; press Ctrl+C to stop.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--interval", type=float, default=3.0, help="seconds between streamed records")
    parser.add_argument("--no-stream", action="store_true", help="only serve the API")
    args = parser.parse_args()

    if not args.no_stream:
        base_url = f"http://127.0.0.1:{args.port}"
        threading.Thread(target=stream_test_data, args=(base_url, args.interval), daemon=True).start()
    uvicorn.run("anomaly_detection.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
