from ipyleaflet import Map, Marker
from pathlib import Path
from retry_requests import retry
import openmeteo_requests
import pandas as pd
import requests_cache


def download_weather_data(city):
    session = requests_cache.CachedSession(".cache", expire_after=3600)
    session = retry(session, retries=5, backoff_factor=0.1)
    openmeteo = openmeteo_requests.Client(session=session)

    # Make sure all required weather variables are listed here. The order of
    # variables in hourly or daily is important to assign them correctly below.
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "start_date": "2021-01-01",
        "end_date": "2025-05-31",
        "hourly": [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "cloud_cover",
            "soil_moisture_1_to_3cm",
            "relative_humidity_2m",
        ],
        "timezone": "GMT",  # Use GMT to ease temporal joins.
    }
    response = openmeteo.weather_api(url, params=params)[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()
    hourly_soil_moisture_1_to_3cm = hourly.Variables(4).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(5).ValuesAsNumpy()

    hourly_data = {
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }

    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["soil_moisture_1_to_3cm"] = hourly_soil_moisture_1_to_3cm
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    return pd.DataFrame(data=hourly_data)
