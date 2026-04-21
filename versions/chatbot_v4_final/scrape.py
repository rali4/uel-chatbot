import os
import re
import time
import requests
from bs4 import BeautifulSoup

URLS = [
    "https://www.uel.ac.uk/",
]

OUT_DIR = "data/pages"
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Connection": "keep-alive",
}

def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def fetch_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
    if r.status_code == 403:
        raise RuntimeError(f"403 Forbidden for {url}. Site is blocking automated requests.")
    r.raise_for_status()
    return r.text

for i, url in enumerate(URLS, 1):
    print(f"[{i}/{len(URLS)}] Downloading: {url}")

    try:
        html = fetch_html(url)
    except Exception as e:
        print(f"FAILED: {e}")
        continue

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = clean_text(soup.get_text(" "))

    filename = os.path.join(OUT_DIR, f"page_{i}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"SOURCE_URL: {url}\n\n{text}")

    time.sleep(1)  # be polite

print("Done. Check data/pages/")