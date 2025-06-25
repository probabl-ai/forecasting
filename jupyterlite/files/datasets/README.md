## Electricity load data (France)

The data is manually retrieved from the ENTSOE Open Data portal. The CSV file needs to be downloaded from an browser by clicking for the right fields in a form:

https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show?name=&defaultValue=false&viewType=TABLE&areaType=BZN&atch=false&dateTime.dateTime=12.10.2023+00:00|UTC|DAY&biddingZone.values=CTY|10YFR-RTE------C!BZN|10YFR-RTE------C&dateTime.timezone=UTC&dateTime.timezone_input=UTC#

- click the "Export Data" menu, then login / create and account;
- once logged-in, navigate back to the link above;
- make sure to select UTC time;
- tick the country of interest ("BZN|FR" in our case) in the left-hand panel;
- put a date within the year of interest (between 2020 and now);
- click the down arrow of the "Export Data" menu, and select "Total Load - Day
  Ahead / Actual (Year, CSV)".

## Weather data:

From https://open-meteo.com/en/docs/historical-forecast-api. The data is
fetched from the `fetch_weather_data.py` script.