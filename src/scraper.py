import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config import URL_MAPPING, CITIES


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


def parse_single_ad(ad):
    """Extracts required data fields from a single property HTML element."""
    ad_id = ad.get("data-id")
    if not ad_id:
        return None

    total_price, area, rooms = None, None, None
    current_floor, total_floors = None, None
    city, municipality, neighborhood = "Unknown", "Unknown", "Unknown"

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
                    area = clean_numeric_value(val_text)
                elif "soba" in legend_text:
                    rooms = val_text
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
        "Area": area,
        "Rooms": rooms,
        "Current_Floor": current_floor,
        "Total_Floors": total_floors,
        "Advertiser_Type": advertiser,
        "Photo_Count": photo_count,
        "Total_Price_EUR": total_price,
    }


def scrape_page(url, headers):
    """Executes an HTTP GET request to a target URL and parses the HTML."""
    parsed_ads = []
    try:
        time.sleep(random.uniform(0.5, 1.5))
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for ad in soup.find_all("div", class_="product-item"):
                parsed = parse_single_ad(ad)
                if parsed:
                    parsed_ads.append(parsed)
    except Exception:
        pass
    return parsed_ads


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
):
    """Coordinates a threaded scraping loop and generates uniquely tracked datasets."""
    if target_cities is None:
        target_cities = ["Beograd"]
    headers = {"User-Agent": "Mozilla/5.0"}
    all_data, seen_ids = [], set()

    urls = []
    for city in target_cities:
        base_url = f"https://www.halooglasi.com/nekretnine/{URL_MAPPING[property_type]}/{CITIES.get(city, 'beograd')}"
        urls.extend([f"{base_url}?page={page}" for page in range(1, num_pages + 1)])

    total_urls = len(urls)
    completed_urls = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scrape_page, url, headers): url for url in urls}
        for future in as_completed(futures):
            try:
                for ad in future.result():
                    if ad["ID"] not in seen_ids:
                        seen_ids.add(ad["ID"])
                        all_data.append(ad)
            except Exception:
                pass

            completed_urls += 1
            if progress_callback:
                progress_callback(completed_urls, total_urls, len(all_data))

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
