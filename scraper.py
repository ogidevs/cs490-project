import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

URL_MAPPING = {
    "flat": "prodaja-stanova",
    "house": "prodaja-kuca",
    "land": "prodaja-zemljista",
    "garage": "prodaja-garaza",
}

CITIES = {
    "Beograd": "beograd",
    "Novi Sad": "novi-sad",
    "Niš": "nis",
    "Kragujevac": "kragujevac",
    "Subotica": "subotica",
    "Zrenjanin": "zrenjanin",
    "Pančevo": "pancevo",
    "Čačak": "cacak",
    "Kruševac": "krusevac",
    "Sombor": "sombor",
}


def clean_numeric_value(text):
    """Cleans a string containing numbers and returns a float."""
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


def parse_price(ad):
    """Extracts the total property price from an ad."""
    total_price = None
    central_feature = ad.find("div", class_="central-feature")
    if central_feature:
        total_price = clean_numeric_value(central_feature.get_text(strip=True))
    return total_price


def parse_price_per_unit(ad):
    """Extracts the price per square meter (or unit) from an ad."""
    price_per_unit = None
    price_surface_div = ad.find("div", class_="price-by-surface")
    if price_surface_div:
        price_per_unit = clean_numeric_value(price_surface_div.get_text(strip=True))
    return price_per_unit


def parse_location(ad):
    """Extracts city, municipality, neighborhood, and street from an ad."""
    city, municipality, neighborhood, street = "Beograd", None, None, None
    places_ul = ad.find("ul", class_="subtitle-places")
    if places_ul:
        parts = [
            li.text.strip().replace("\xa0", "").replace(",", "")
            for li in places_ul.find_all("li")
        ]
        parts = [p for p in parts if p]
        if len(parts) > 0:
            city = parts[0]
        if len(parts) > 1:
            municipality = parts[1].replace("Opština ", "")
        if len(parts) > 2:
            neighborhood = parts[2]
        if len(parts) > 3:
            street = parts[3]
    return city, municipality, neighborhood, street


def parse_features(ad):
    """Extracts area, area unit, room count, and floor details from an ad."""
    area, area_unit, rooms, floor_str = None, None, None, None
    features_ul = ad.find("ul", class_="product-features")

    if features_ul:
        for li in features_ul.find_all("li"):
            legend_tag = li.find("span", class_="legend")
            if legend_tag:
                legend = legend_tag.text.strip().lower()
                legend_tag.extract()

                val_tag = li.find("div", class_="value-wrapper")
                if val_tag:
                    val_text = val_tag.text.strip().replace("\xa0", " ")

                    if "kvadratura" in legend or "površina" in legend:
                        area = clean_numeric_value(val_text)
                        if "ar" in val_text:
                            area_unit = "ar"
                        elif "ha" in val_text:
                            area_unit = "ha"
                        else:
                            area_unit = "m2"
                    elif "soba" in legend:
                        rooms = val_text
                    elif "spratnost" in legend:
                        floor_str = val_text

    current_floor, total_floors = floor_str, None
    if floor_str and "/" in floor_str:
        floor_parts = floor_str.split("/")
        current_floor = floor_parts[0].strip()
        if len(floor_parts) > 1:
            total_floors = floor_parts[1].strip()

    return area, area_unit, rooms, current_floor, total_floors


def parse_advertiser_and_meta(ad):
    """Extracts advertiser type, publish date, photo count, and description."""
    advertiser = None
    adv_span = ad.find("span", {"data-field-name": "oglasivac_nekretnine_s"})
    if adv_span and "data-field-value" in adv_span.attrs:
        advertiser = adv_span["data-field-value"].capitalize()

    pub_date = None
    date_span = ad.find("span", class_="publish-date")
    if date_span:
        pub_date = date_span.text.strip()

    photo_count = 0
    img_span = ad.find("span", class_="pi-img-count-num")
    if img_span:
        match = re.search(r"\d+", img_span.text)
        if match:
            photo_count = int(match.group(0))

    description = None
    desc_p = ad.find("p", class_="short-desc")
    if desc_p:
        description = desc_p.text.strip()

    return advertiser, pub_date, photo_count, description


def parse_single_ad(ad):
    """Coordinates parsing of a single ad's HTML element into a dictionary mapping."""
    ad_id = ad.get("data-id")
    if not ad_id:
        return None

    title_tag = ad.find("h3", class_="product-title")
    title = title_tag.text.strip() if title_tag else None

    link = None
    if title_tag and title_tag.a and "href" in title_tag.a.attrs:
        link = "https://www.halooglasi.com" + title_tag.a["href"]

    total_price = parse_price(ad)
    price_per_unit = parse_price_per_unit(ad)
    city, municipality, neighborhood, street = parse_location(ad)
    area, area_unit, rooms, current_floor, total_floors = parse_features(ad)
    advertiser, pub_date, photo_count, description = parse_advertiser_and_meta(ad)

    # Impute price per unit if missing
    if not price_per_unit and total_price and area and area > 0:
        price_per_unit = round(total_price / area, 2)

    return {
        "ID": ad_id,
        "Title": title,
        "Total_Price_EUR": total_price,
        "Area": area,
        "Area_Unit": area_unit,
        "Price_per_Unit_EUR": price_per_unit,
        "Rooms": rooms,
        "Current_Floor": current_floor,
        "Total_Floors": total_floors,
        "City": city,
        "Municipality": municipality,
        "Neighborhood": neighborhood,
        "Street": street,
        "Advertiser_Type": advertiser,
        "Publish_Date": pub_date,
        "Photo_Count": photo_count,
        "Description": description,
        "Link": link,
    }


def scrape_page(url, headers):
    """Fetches and parses a single URL page, returning a list of parsed ad dictionaries."""
    parsed_ads = []
    try:
        # Short random sleep prevents overwhelming the target server
        time.sleep(random.uniform(0.5, 1.5))
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            ads = soup.find_all("div", class_="product-item")

            for ad in ads:
                parsed_ad = parse_single_ad(ad)
                if parsed_ad:
                    parsed_ads.append(parsed_ad)
    except Exception as e:
        print(f"Error scraping {url}: {e}")

    return parsed_ads


def scrape_halooglasi(
    property_type="flat", city="Beograd", num_pages=50, max_workers=10
):
    """
    Coordinates concurrent scraping of multiple pages for a target city and property type.

    Args:
        property_type (str): Type of property (flat, house, land, garage).
        city (str): Target city name.
        num_pages (int): Number of pages to iterate over.
        max_workers (int): The number of concurrent threads to use.
    """
    if property_type not in URL_MAPPING:
        raise ValueError(
            f"Invalid property_type. Supported types: {list(URL_MAPPING.keys())}"
        )

    base_url = f"https://www.halooglasi.com/nekretnine/{URL_MAPPING[property_type]}/{CITIES.get(city, city)}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    all_data = []
    seen_ids = set()

    # Generate all targeted URLs
    urls = [f"{base_url}?page={page}" for page in range(1, num_pages + 1)]
    print(
        f"Starting parallel scraping of {len(urls)} pages with {max_workers} threads..."
    )

    # Execute requests in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(scrape_page, url, headers): url for url in urls}

        # Process results as they complete
        for i, future in enumerate(as_completed(futures), 1):
            ads_from_page = future.result()

            # Deduplicate entries by ID across threads
            for ad in ads_from_page:
                if ad["ID"] not in seen_ids:
                    seen_ids.add(ad["ID"])
                    all_data.append(ad)

            if i % 10 == 0 or i == num_pages:
                print(
                    f"Completed {i}/{num_pages} pages. Total ads gathered: {len(all_data)}"
                )

    return all_data


if __name__ == "__main__":
    target_type = "flat"
    # Using 'max_workers=10' parallelizes the job. Keep it between 5-15 to avoid getting blocked.
    data = scrape_halooglasi(
        property_type=target_type, city="Beograd", num_pages=400, max_workers=10
    )

    df = pd.DataFrame(data)
    df.to_csv(f"dataset_{target_type}s.csv", index=False, encoding="utf-8")
    print("\nSample Data:")
    print(df.head())
