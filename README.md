# Real Estate Price Prediction System

## Project Overview

This project analyzes the real estate market and predicts property prices based on features such as area, rooms, floor position, advertiser type, photo count, municipality, and neighborhood. It leverages machine learning algorithms and web scraping to dynamically gather data.

## Dataset

The dataset is collected via web scraping from public real estate listings (e.g., HaloOglasi). It includes features like total price, price per square meter, area, municipality, neighborhood, floor level, number of rooms, advertiser type, and photo count.

## Setup Instructions

### 1. Environment Setup (using `uv`)

Ensure you have `uv` installed.

```bash
uv sync
```

### 2. Running the Application

To launch the Streamlit app:

```bash
uv run streamlit run app/streamlit_app.py
```

To run the Jupyter Notebook for exploratory analysis:

```bash
uv run jupyter lab
```

## Project Plan

1. Phase 1: Data scraping module development (Target: >1000 real estate ads).
2. Phase 2: Exploratory Data Analysis (EDA) and cleaning of outliers/missing data.
3. Phase 3: Feature engineering (floor ratios, room density, advertiser signals, categorical encoding, numerical scaling).
4. Phase 4: Model development (Linear Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM) and evaluation.
5. Phase 5: Streamlit Application development for dynamic prediction and visualization.

- **Data Collection:** Web scraping of real estate listings.
- **Exploratory Data Analysis (EDA):** Statistical analysis and data visualization.
- **Feature Engineering:** Cleaning missing values, converting string categories to numerical, and adding floor, room-density, and listing-quality signals.
- **Model Training:** Implementation of Linear Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM.
- **Deployment:** Building a Streamlit application for user interaction.

## Functional Requirements

- Scrape property data dynamically by selecting specific cities and property types.
- Persist collected data locally in CSV format.
- Visualize distributions and correlations of property features.
- Evaluate multiple ML algorithms using RMSE and R-Squared metrics.
- Provide a user-friendly UI (Streamlit) to input property attributes and receive a price prediction.

- The system must scrape up-to-date listings based on user-selected cities.
- The system must clean the scraped data, handle outliers/missing values, and engineer reusable features for prediction.
- The system must provide interactive visualizations of the dataset.
- The system must train ML models and allow the user to predict the price of a property based on manual inputs.

## Architecture

- Data Layer: Web scraper feeding into CSV files.
- Logic Layer: Modularized Python scripts (src/) handling preprocessing, training, and prediction.
- Presentation Layer: Streamlit App and Jupyter Notebooks.

- **data/**: Stores raw CSV files and serialized joblib models.
- **src/**: Contains core Python scripts for scraping, EDA, preprocessing, training, and predicting.
- **app/**: Contains the Streamlit frontend.
- **notebooks/**: Contains Jupyter notebooks for raw experimentation and Level 1 submissions.
