import logging
import pandas as pd
import pgeocode
import holidays
import requests
from typing import Dict, List, Any, Tuple
from .utility import  normalize_postal_code

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_holiday_features(
        postal_code_date_ranges: Dict[str, Dict[str, str]],
        country: str = "US"
) -> pd.DataFrame:
    """
    Generates state-aware holiday features for each given postal code and its date range.
    """
    if not postal_code_date_ranges:
        logger.warning("Input postal_code_date_ranges is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    logger.info(f"Generating holiday features for {len(postal_code_date_ranges)} postal codes.")
    geo = pgeocode.Nominatim(country)
    all_holidays_list = []

    for code, dates in postal_code_date_ranges.items():
        cleaned_code = normalize_postal_code(code)
        query_result = geo.query_postal_code(cleaned_code)

        state = query_result.state_code if query_result is not None else None
        if not isinstance(state, str) or not state:
            logger.warning(f"Could not determine state for postal code {code}. Skipping.")
            continue

        try:
            s_date = pd.to_datetime(dates["start_date"])
            e_date = pd.to_datetime(dates["end_date"])
        except Exception as e:
            logger.warning(f"Invalid date format for postal code {code}. Skipping. Error: {e}")
            continue

        date_range = pd.date_range(start=s_date, end=e_date, freq='D')

        # Get holiday provider for the specific state
        holiday_provider = getattr(holidays, country)(state=state, years=range(s_date.year, e_date.year + 2))

        holidays_df = pd.DataFrame({'date': date_range})
        holidays_df['holiday_name'] = holidays_df['date'].apply(lambda d: holiday_provider.get(d))
        holidays_df['is_holiday'] = holidays_df['holiday_name'].notna()

        # Ensure date column stays in datetime format
        holidays_df['date_only'] = pd.to_datetime(holidays_df['date'])

        holiday_dates = holidays_df.loc[holidays_df['is_holiday'], 'date_only']

        if not holiday_dates.empty:
            last_holiday_map = {d: d for d in holiday_dates}
            next_holiday_map = {d: d for d in holiday_dates}

            holidays_df['last_holiday_date'] = holidays_df['date_only'].map(last_holiday_map).ffill()
            holidays_df['last_holiday_date'] = pd.to_datetime(holidays_df['last_holiday_date'], errors='coerce')
            holidays_df['days_since_last_holiday'] = (
                        holidays_df['date_only'] - holidays_df['last_holiday_date']).dt.days

            holidays_df['next_holiday_date'] = holidays_df['date_only'].map(next_holiday_map).bfill()
            holidays_df['next_holiday_date'] = pd.to_datetime(holidays_df['next_holiday_date'], errors='coerce')
            holidays_df['days_until_next_holiday'] = (
                        holidays_df['next_holiday_date'] - holidays_df['date_only']).dt.days
        else:
            # If no holidays in the range, set default values
            holidays_df['days_since_last_holiday'] = 999
            holidays_df['days_until_next_holiday'] = 999

        # Fill NaNs for non-holiday dates
        holidays_df.fillna({
            'days_since_last_holiday': 999,
            'days_until_next_holiday': 999,
            'holiday_name': "No Holiday"
        }, inplace=True)

        holidays_df['postal_code'] = code
        holidays_df['state'] = state

        all_holidays_list.append(holidays_df)

    if not all_holidays_list:
        logger.warning("No holiday data could be generated.")
        return pd.DataFrame()

    final_holidays = pd.concat(all_holidays_list, ignore_index=True)

    # Select and return final columns
    final_features = final_holidays[[
        'date', 'postal_code', 'is_holiday', 'holiday_name',
        'days_since_last_holiday', 'days_until_next_holiday', 'state'
    ]].copy()

    logger.info(f"Finished creating holiday features. Total rows: {len(final_features)}")
    return final_features


def create_weather_features(
        postal_code_date_ranges: Dict[str, Dict[str, str]]
) -> pd.DataFrame:
    """
    Fetches historical weather data for a list of postal codes, each with its own date range.
    """
    if not postal_code_date_ranges:
        logger.warning("Input postal_code_date_ranges is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    logger.info(f"Generating weather features for {len(postal_code_date_ranges)} postal codes.")

    geo = pgeocode.Nominatim('us')
    all_weather_data = []

    for raw_code, dates in postal_code_date_ranges.items():
        code = normalize_postal_code(raw_code)

        try:
            start_date = pd.to_datetime(dates["start_date"]).strftime("%Y-%m-%d")
            end_date = pd.to_datetime(dates["end_date"]).strftime("%Y-%m-%d")
        except Exception as e:
            logger.warning(f"Invalid date range for postal code {raw_code}. Skipping. Error: {e}")
            continue

        location_info = geo.query_postal_code(code)
        state = location_info.state_code if location_info is not None else None
        if location_info is None or pd.isna(location_info.latitude) or pd.isna(location_info.longitude):
            logger.warning(f"Could not find coordinates for postal code: {raw_code}. Skipping.")
            continue

        lat, lon = location_info.latitude, location_info.longitude
        api_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum,wind_speed_10m_max",
            "timezone": "auto"
        }

        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'daily' not in data:
                logger.warning(f"No 'daily' weather data found for postal code {raw_code}.")
                continue

            weather_df = pd.DataFrame(data['daily'])
            if weather_df.empty:
                logger.warning(f"Empty weather data returned for postal code {raw_code}.")
                continue

            weather_df['postal_code'] = raw_code
            weather_df['state'] = state
            all_weather_data.append(weather_df)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch weather data for postal code {raw_code}: {e}")

    if not all_weather_data:
        logger.warning("No weather data was fetched.")
        return pd.DataFrame()

    final_weather_df = pd.concat(all_weather_data, ignore_index=True)
    final_weather_df.rename(columns={'time': 'date'}, inplace=True)
    final_weather_df['date'] = pd.to_datetime(final_weather_df['date'], errors='coerce')
    final_weather_df.dropna(subset=['date'], inplace=True)

    final_features = final_weather_df[[
        'date', 'postal_code', 'weather_code', 'temperature_2m_max', 'temperature_2m_min',
        'precipitation_sum', 'snowfall_sum', 'wind_speed_10m_max', 'state'
    ]].copy()

    logger.info(f"Finished creating weather features. Total rows: {len(final_features)}")
    return final_features