import logging

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import os
import json
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config import URL_MAPPING, CITIES

try:
    from curl_cffi import requests as curl_requests
except Exception:  # Preferred fast client; falls back if unavailable.
    curl_requests = None

try:
    import cloudscraper
except Exception:  # Optional dependency; scraper falls back to requests.
    cloudscraper = None

try:
    from playwright.sync_api import sync_playwright
except Exception:  # Optional dependency for browser fallback.
    sync_playwright = None


def clean_numeric_value(text):
    """Parses and extracts floating-point numbers from messy strings securely."""
    if not text:
        return None
    cleaned = (
        text.replace(".", "")
        .replace(",", ".")
        .replace("€", "")
        .replace("EUR", "")
        .replace(" ", "")
        .replace("\xa0", "")
    )
    match = re.search(r"[\d\.]+", cleaned)
    if match:
        return float(match.group(0))
    return None


def parse_area_value(val_text, property_type):
    """Parses area and normalizes non-flat units such as land ar to square meters."""
    if not val_text:
        return None

    area_value = clean_numeric_value(val_text)
    if area_value is None:
        return None

    normalized = val_text.lower()
    if property_type == "land" and "ari" in normalized:
        return area_value * 100.0

    return area_value


def parse_single_ad(ad, property_type="flat"):
    """Extracts required data fields from a single property HTML element."""
    ad_id = ad.get("data-id")
    if not ad_id:
        return None

    total_price, area, rooms = None, None, None
    current_floor, total_floors = None, None
    city, municipality, neighborhood = "Unknown", "Unknown", "Unknown"
    subtype_defaults = {
        "flat": "Stan",
        "house": "Kuća",
        "land": "Zemljište",
        "garage": "Garaža",
    }
    property_subtype = subtype_defaults.get(property_type, "Unknown")

    central_feature = ad.find("div", class_="central-feature")
    if central_feature:
        total_price = clean_numeric_value(central_feature.get_text(strip=True))

    places_ul = ad.find("ul", class_="subtitle-places")
    if places_ul:
        parts = [
            li.text.strip().replace("\xa0", "").replace(",", "")
            for li in places_ul.find_all("li")
            if li.text.strip()
        ]
        if len(parts) > 0:
            city = parts[0]
        if len(parts) > 1:
            municipality = parts[1].replace("Opština ", "")
        if len(parts) > 2:
            neighborhood = parts[2]

    features_ul = ad.find("ul", class_="product-features")
    if features_ul:
        for li in features_ul.find_all("li"):
            legend = li.find("span", class_="legend")
            val = li.find("div", class_="value-wrapper")
            if legend and val:
                legend_text = legend.text.strip().lower()
                val_text = val.text.strip().replace("\xa0", " ")

                if "kvadratura" in legend_text or "površina" in legend_text:
                    area = parse_area_value(val_text, property_type)
                elif "soba" in legend_text:
                    rooms = val_text
                elif "tip nekretnine" in legend_text:
                    property_subtype = val_text.strip()
                elif "spratnost" in legend_text:
                    # Logika za spratnost (npr. "3 / 5" ili "Prizemlje")
                    if "/" in val_text:
                        floor_parts = val_text.split("/")
                        current_floor = floor_parts[0].strip()
                        total_floors = floor_parts[1].strip()
                    else:
                        current_floor = val_text.strip()

    advertiser = "Unknown"
    adv_span = ad.find("span", {"data-field-name": "oglasivac_nekretnine_s"})
    if adv_span and "data-field-value" in adv_span.attrs:
        advertiser = adv_span["data-field-value"].capitalize()

    # 5. Broj fotografija
    photo_count = 0
    img_span = ad.find("span", class_="pi-img-count-num")
    if img_span:
        match = re.search(r"\d+", img_span.text)
        if match:
            photo_count = int(match.group(0))

    return {
        "ID": ad_id,
        "City": city,
        "Municipality": municipality,
        "Neighborhood": neighborhood,
        "Property_Type": property_type,
        "Property_Subtype": property_subtype,
        "Area": area,
        "Rooms": rooms,
        "Current_Floor": current_floor,
        "Total_Floors": total_floors,
        "Advertiser_Type": advertiser,
        "Photo_Count": photo_count,
        "Total_Price_EUR": total_price,
    }


def _build_http_client(proxy_url=None):
    """Builds a fast HTTP client with browser impersonation when available."""
    if curl_requests is not None:
        client = curl_requests.Session(impersonate="chrome124")
    elif cloudscraper is not None:
        try:
            client = cloudscraper.create_scraper()
        except Exception:
            client = requests.Session()
    else:
        client = requests.Session()

    if proxy_url:
        client.proxies.update({"http": proxy_url, "https": proxy_url})
    return client


def scrape_page(url, headers, property_type="flat", proxy_url=None):
    """Executes an HTTP GET request and parses property cards with diagnostics."""
    parsed_ads = []
    error = None
    status_code = None
    try:
        client = _build_http_client(proxy_url=proxy_url)
        time.sleep(random.uniform(0.5, 1.5))
        response = client.get(url, headers=headers, timeout=20)
        status_code = response.status_code

        if status_code != 200:
            error = f"HTTP {status_code}"
            return {"ads": parsed_ads, "error": error, "status_code": status_code}

        soup = BeautifulSoup(response.text, "html.parser")
        ad_nodes = soup.find_all("div", class_="product-item")

        if not ad_nodes and "Just a moment" in response.text:
            error = "Blocked by anti-bot protection"

        for ad in ad_nodes:
            parsed = parse_single_ad(ad, property_type=property_type)
            if parsed:
                parsed_ads.append(parsed)

    except Exception as exc:
        error = str(exc)

    return {"ads": parsed_ads, "error": error, "status_code": status_code}


def scrape_page_browser(
    url,
    property_type="flat",
    proxy_url=None,
    cookie_header=None,
    playwright_headless=None,
):
    """Loads a listing page in a real browser and parses rendered HTML."""
    if sync_playwright is None:
        return {
            "ads": [],
            "error": "Playwright is not installed",
            "status_code": None,
        }

    parsed_ads = []
    error = None
    status_code = None

    def _ensure_windows_subprocess_loop_support():
        """Ensure Playwright can spawn subprocesses on Windows event loop."""
        if os.name != "nt":
            return

        policy_cls = getattr(asyncio, "WindowsProactorEventLoopPolicy", None)
        if policy_cls is None:
            return

        current_policy = asyncio.get_event_loop_policy()
        if not isinstance(current_policy, policy_cls):
            asyncio.set_event_loop_policy(policy_cls())

    try:
        _ensure_windows_subprocess_loop_support()

        # Headed mode is less likely to trigger anti-bot blocks on some sites.
        if playwright_headless is None:
            headless_env = (
                os.getenv("HALOOGLASI_PLAYWRIGHT_HEADLESS", "0").strip().lower()
            )
            use_headless = headless_env in {"1", "true", "yes"}
        else:
            use_headless = bool(playwright_headless)
        launch_kwargs = {"headless": use_headless}
        if proxy_url:
            launch_kwargs["proxy"] = {"server": proxy_url}

        with sync_playwright() as p:
            browser = p.chromium.launch(**launch_kwargs)
            context = browser.new_context(locale="sr-RS")

            if cookie_header:
                # Optional cookie injection from browser session.
                context.add_cookies(
                    [
                        {
                            "name": c.split("=", 1)[0].strip(),
                            "value": c.split("=", 1)[1].strip(),
                            "domain": ".halooglasi.com",
                            "path": "/",
                        }
                        for c in cookie_header.split(";")
                        if "=" in c
                    ]
                )

            page = context.new_page()
            response = page.goto(url, wait_until="domcontentloaded", timeout=45000)
            status_code = response.status if response else None

            # Let dynamic content settle and try finding product cards.
            page.wait_for_timeout(2500)
            html = page.content()
            browser.close()

        soup = BeautifulSoup(html, "html.parser")
        ad_nodes = soup.find_all("div", class_="product-item")
        if not ad_nodes:
            if "Just a moment" in html or "cf-challenge" in html:
                error = "Blocked by anti-bot protection (browser fallback)"
            else:
                error = "No listing cards found in rendered page"

        for ad in ad_nodes:
            parsed = parse_single_ad(ad, property_type=property_type)
            if parsed:
                parsed_ads.append(parsed)

    except NotImplementedError:
        error = (
            "Playwright browser fallback is not supported by the current asyncio loop. "
            "Try disabling browser fallback or run with HTTP cookie/proxy in Advanced settings."
        )
    except Exception as exc:
        error = str(exc)

    return {"ads": parsed_ads, "error": error, "status_code": status_code}


def log_metadata(file_id, property_type, cities, num_records, scrape_time):
    """Saves dataset context securely into a central JSON registry, avoiding messy filenames."""
    os.makedirs("data", exist_ok=True)
    meta_path = os.path.join("data", "metadata.json")

    # Load existing registry or create a new one
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Register the new dataset
    metadata[file_id] = {
        "property_type": property_type,
        "cities": cities,
        "records": num_records,
        "scraped_at": scrape_time,
    }

    # Save registry
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def scrape_halooglasi(
    property_type="flat",
    target_cities=None,
    num_pages=5,
    max_workers=5,
    progress_callback=None,
    proxy_url=None,
    cookie_header=None,
    use_browser_fallback=True,
    playwright_headless=None,
):
    """Coordinates a threaded scraping loop and generates uniquely tracked datasets."""
    if target_cities is None:
        target_cities = ["Beograd"]

    if proxy_url is None:
        proxy_url = os.getenv("HALOOGLASI_PROXY", "").strip() or None
    if cookie_header is None:
        cookie_header = os.getenv("HALOOGLASI_COOKIE", "").strip() or None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "sr-RS,sr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.halooglasi.com/",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
    }
    if cookie_header:
        headers["Cookie"] = cookie_header

    all_data, seen_ids = [], set()
    request_errors = []
    fallback_signal_detected = False

    urls = []
    for city in target_cities:
        base_url = f"https://www.halooglasi.com/nekretnine/{URL_MAPPING[property_type]}/{CITIES.get(city, 'beograd')}"
        urls.extend([f"{base_url}?page={page}" for page in range(1, num_pages + 1)])

    total_urls = len(urls)
    completed_urls = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(scrape_page, url, headers, property_type, proxy_url): url
            for url in urls
        }
        for future in as_completed(futures):
            source_url = futures[future]
            try:
                result = future.result()
                for ad in result["ads"]:
                    if ad["ID"] not in seen_ids:
                        seen_ids.add(ad["ID"])
                        all_data.append(ad)
                if result.get("error"):
                    request_errors.append(f"{source_url} -> {result['error']}")
                if result.get("status_code") == 403:
                    fallback_signal_detected = True
                if result.get("error") and "anti-bot" in result["error"].lower():
                    fallback_signal_detected = True
            except Exception as exc:
                request_errors.append(f"{source_url} -> {exc}")

            completed_urls += 1
            if progress_callback:
                progress_callback(completed_urls, total_urls, len(all_data))

    # Browser fallback only if the fast HTTP path is blocked.
    if len(all_data) == 0 and use_browser_fallback and fallback_signal_detected:
        logging.info(
            "No ads found with HTTP requests. Retrying with browser fallback..."
        )
        browser_errors = []
        for idx, url in enumerate(urls, start=1):
            result = scrape_page_browser(
                url,
                property_type=property_type,
                proxy_url=proxy_url,
                cookie_header=cookie_header,
                playwright_headless=playwright_headless,
            )
            for ad in result["ads"]:
                if ad["ID"] not in seen_ids:
                    seen_ids.add(ad["ID"])
                    all_data.append(ad)

            if result.get("error"):
                browser_errors.append(f"{url} -> {result['error']}")

            if progress_callback:
                progress_callback(idx, total_urls, len(all_data))

        request_errors.extend(browser_errors)

    if len(all_data) == 0:
        unique_errors = sorted(set(request_errors))
        details = (
            "\n".join(unique_errors[:5])
            if unique_errors
            else "No response diagnostics available."
        )
        raise RuntimeError(
            "Scraping returned 0 ads. Possible anti-bot block or changed site structure. "
            "Try fewer pages, one city, or retry later.\n"
            f"Diagnostics:\n{details}"
        )

    df_new = pd.DataFrame(all_data)
    scrape_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_new["Scrape_Date"] = scrape_time

    os.makedirs("data", exist_ok=True)

    # 1. Generate a clean, unique identifier using a Unix timestamp
    file_id = f"dataset_{int(time.time())}"
    filename = f"{file_id}.csv"
    filepath = os.path.join("data", filename)

    # 2. Save the CSV
    df_new.to_csv(filepath, index=False, encoding="utf-8")

    # 3. Log the complex metadata to our central JSON catalog
    log_metadata(file_id, property_type, target_cities, len(df_new), scrape_time)

    return df_new, filename
