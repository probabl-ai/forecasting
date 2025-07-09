# %% [markdown]
# # Historical weather data download from open-meteo.com
#
# This notebook downloads historical weather data for 10 medium to large urban
# areas in France from the Open Meteo Historical Forecast API. The data is
# saved in a Parquet file in the `datasets` folder.
#
# Since calling the API is slow and can reach rate limits quite easily with
# free accounts, we use a cache to avoid downloading the same data multiple
# times.

# %%
# # Extra dependencies when running this notebook in JupyterLite/Pyodide.
# %pip install -q openmeteo-requests retry-requests requests-cache ipyleaflet

# %%
from pathlib import Path
from ipyleaflet import Map, Marker
import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

# %% [markdown]
#
# List of 10 medium to large urban areas with their GPS coordinates to cover
# most regions in France with a slight focus on most populated regions that
# are likely to drive electricity demand.
#
# The coordinates were suggested by Mistral's Le Chat. So better check on a map
# if they are correct.

# %%
cities = [
    {"name": "Paris", "latitude": 48.8566, "longitude": 2.3522},
    {"name": "Lyon", "latitude": 45.7640, "longitude": 4.8357},
    {"name": "Marseille", "latitude": 43.2965, "longitude": 5.3698},
    {"name": "Toulouse", "latitude": 43.6047, "longitude": 1.4442},
    {"name": "Lille", "latitude": 50.6292, "longitude": 3.0573},
    {"name": "Limoges", "latitude": 45.8336, "longitude": 1.2616},
    {"name": "Nantes", "latitude": 47.2184, "longitude": -1.5536},
    {"name": "Strasbourg", "latitude": 48.5734, "longitude": 7.7521},
    {"name": "Brest", "latitude": 48.3904, "longitude": -4.4861},
    {"name": "Bayonne", "latitude": 43.4833, "longitude": -1.4667},
]

map_center = [46.6034, 1.8883]  # Approximate center of France
m = Map(center=map_center, zoom=6)
for city in cities:
    marker = Marker(location=(city["latitude"], city["longitude"]), title=city["name"])
    m.add_layer(marker)
m

# %% [markdown]
#
# Download weather data for each city. The data is downloaded from the Open
# Meteo Historical Forecast API, which provides historical weather data for
# free (with rate limits).


# %%
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


# %% [markdown]
#
# Download weather data for each city and save it to a Parquet file in the
# `datasets` folder. The data is saved in a format that can be easily joined
# with the electricity load data. We use the Parquet format to save storage
# space and to speed up the loading time in the notebook. Parquet is also
# interesting because data types are not ambiguous contrary to CSV.
# %%
datasets_folder = Path("../datasets")
for city in cities:
    filepath = datasets_folder / f"weather_{city['name'].lower()}.parquet"
    if filepath.exists():
        print(f"Weather data for {city['name']} already exists at {filepath}.")
        continue

    print(f"Downloading weather data for {city['name']}...")
    df = download_weather_data(city)
    df.to_parquet(filepath, index=False)
    print(f"Weather data for {city['name']} saved to {filepath}.")

# %%
