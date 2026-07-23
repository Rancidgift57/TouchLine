import os
import time
import urllib.request
import urllib.error
from datetime import datetime

# URL to ping — checks environment variable 'RENDER_API_URL' first, otherwise uses fallback
TARGET_URL = os.getenv("RENDER_API_URL", "https://touchlinee.onrender.com/health")


def ping_api(url: str) -> None:
    """Sends a GET request to the target URL and logs the response."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Pinging: {url}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Render-KeepAlive-Pinger/1.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            status_code = response.getcode()
            print(f"[{timestamp}] Success! Status Code: {status_code}")
    except urllib.error.HTTPError as e:
        print(f"[{timestamp}] HTTP Error: {e.code} - {e.reason}")
    except urllib.error.URLError as e:
        print(f"[{timestamp}] Connection Error: {e.reason}")
    except Exception as e:
        print(f"[{timestamp}] Unexpected Error: {e}")


if __name__ == "__main__":
    ping_api(TARGET_URL)
