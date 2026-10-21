"""
Downloader for the BEND data. It replaces BEND/scripts/download_bend.py and is run in
the same way, from the BEND folder:
    python scripts/download_bend.py

BEND's own script failed for us because the links in the file listing contain
'&current_dir=...', and the HTML parser reads '&curren' as the entity for the currency
sign (U+00A4). That breaks the folder links, the listing comes back empty, and the
script finishes as if it had succeeded. This version undoes that in every link, retries
failed requests, streams large files to disk, and stops with an error if it finds
nothing to download.
"""

import os
import sys
import time
import requests
from bs4 import BeautifulSoup

BASE = "https://sid.erda.dk/"
ROOT = "https://sid.erda.dk/cgi-sid/ls.py?share_id=f6hdp1zTzh&current_dir=.&flags=f"


def fix_href(href: str) -> str:
    """Turn the currency sign back into '&curren' in a link, which restores '&current_dir'."""
    if href is None:
        return None
    href = href.replace("\u00a4t_dir", "&current_dir")
    href = href.replace("\u00a4", "&curren")
    return href


def make_soup(content):
    # html.parser keeps '&current_dir' more reliably than lxml here
    try:
        return BeautifulSoup(content, "html.parser")
    except Exception:
        return BeautifulSoup(content, "lxml")


def get_links(url, tries=4):
    last_err = None
    for attempt in range(1, tries + 1):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            soup = make_soup(resp.content)
            links = []
            links.extend(soup.find_all("a", {"class": ["leftpad directoryicon"]}))
            links.extend(soup.find_all("a", {"title": "open"}))
            return links
        except Exception as e:
            last_err = e
            print(f"  [retry {attempt}/{tries}] listing failed: {e}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"Could not list {url} after {tries} tries: {last_err}")


def download_file(url, destination, tries=4):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    for attempt in range(1, tries + 1):
        try:
            with requests.get(url, allow_redirects=True, stream=True, timeout=600) as r:
                r.raise_for_status()
                tmp = destination + ".part"
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
                os.replace(tmp, destination)
            return
        except Exception as e:
            print(f"  [retry {attempt}/{tries}] download failed ({destination}): {e}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"Could not download {url} -> {destination}")


def crawl(url, destination="."):
    links = get_links(url)
    if not links:
        raise RuntimeError(
            f"No file or folder links found on:\n  {url}\n"
            "The listing came back empty (share layout or URL problem)."
        )
    n_files = 0
    for child in links:
        href = fix_href(child.get("href"))
        if href is None:
            continue
        if child.get("title") == "open":
            file_url = f"{BASE}{href}"
            child_path = href[27:]
            dest = f"{destination}/{child_path}"
            print(f"  file: {dest}")
            download_file(file_url, dest)
            n_files += 1
        else:
            dir_url = f"{BASE}cgi-sid/{href}"
            n_files += crawl(dir_url, destination)
    return n_files


if __name__ == "__main__":
    print("Downloading the BEND data ...")
    total = crawl(ROOT, destination=".")
    if total <= 1:
        print(f"\nERROR: only {total} file(s) downloaded; no dataset was found.")
        sys.exit(1)
    print(f"\nDownloaded {total} files.")
