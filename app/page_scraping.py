import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.scraper import scrape_halooglasi, CITIES, URL_MAPPING


def render_scraping_page(active_dataset_filename, active_data_path):
    """Renders the UI elements and logic for scraping new data from the web."""
    st.header("Web Scraping & Data Loading")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Scrape New Data")
        scrape_type = st.selectbox("Property Type", list(URL_MAPPING.keys()))
        selected_cities = st.multiselect(
            "Select Cities", list(CITIES.keys()), default=["Beograd", "Novi Sad"]
        )
        pages_per_city = st.slider("Pages per city", 1, 500, 5)

        with st.expander("Advanced (anti-bot)"):
            cookie_header = st.text_area(
                "Cookie header (optional)",
                value="",
                help=(
                    "Paste Cookie header from browser DevTools request to HaloOglasi "
                    "if requests are blocked by anti-bot protection."
                ),
            ).strip()
            proxy_url = st.text_input(
                "Proxy URL (optional)",
                value="",
                help="Format example: http://user:pass@host:port",
            ).strip()
            use_browser_fallback = st.checkbox(
                "Use browser fallback on 403/anti-bot",
                value=False,
                help="Retry with Playwright only if curl_cffi requests are blocked.",
            )
            use_headless_browser = st.checkbox(
                "Use headless browser fallback",
                value=False,
                help="Headed mode is usually more reliable against anti-bot, but opens browser windows.",
            )

        if st.button("Start Scraping"):
            if not selected_cities:
                st.error("Please select at least 1 city.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(completed, total, found_ads):
                    progress_fraction = completed / total if total > 0 else 0
                    progress_bar.progress(progress_fraction)
                    status_text.info(
                        f"Pages scraped: **{completed}/{total}** | Unique properties found: **{found_ads}**"
                    )

                try:
                    df, new_filename = scrape_halooglasi(
                        property_type=scrape_type,
                        target_cities=selected_cities,
                        num_pages=pages_per_city,
                        progress_callback=update_progress,
                        proxy_url=proxy_url or None,
                        cookie_header=cookie_header or None,
                        use_browser_fallback=use_browser_fallback,
                        playwright_headless=use_headless_browser,
                    )
                    progress_bar.empty()
                    status_text.success(
                        f"Scraping complete. Saved to `{new_filename}` ({len(df)} records)."
                    )
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Scraping failed: {e}")

    with col2:
        st.subheader("Dataset Preview")
        if active_dataset_filename and os.path.exists(active_data_path):
            df = pd.read_csv(active_data_path)
            scrape_date = "Unknown"
            if "Scrape_Date" in df.columns and not df.empty:
                scrape_date = df["Scrape_Date"].iloc[0]

            st.info(f"Scraped on: **{scrape_date}**")
            st.write(f"Total properties: {len(df)}")

            if df.empty:
                st.warning(
                    "Selected dataset is empty. Run scraping again to collect records."
                )
            else:
                st.dataframe(df.astype(object).tail(10))
        else:
            st.warning("No dataset selected. Scrape data to begin.")
