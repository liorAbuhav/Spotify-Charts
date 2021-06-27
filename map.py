import pandas as pd
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
from geopy.geocoders import Nominatim
import numpy as np
import folium
from folium.plugins import MarkerCluster

def get_continent(col):
    try:
        cn_a2_code = country_name_to_country_alpha2(col)
    except:
        cn_a2_code = 'Unknown'
    try:
        cn_continent = country_alpha2_to_continent_code(cn_a2_code)
    except:
        cn_continent = 'Unknown'
    return cn_a2_code, cn_continent


def geolocate(country):
    try:
        # Geolocate the center of the country
        loc = geolocator.geocode(country)
        # And return latitude and longitude
        return loc.latitude, loc.longitude
    except:
        # Return missing value
        return np.nan


# get the countries
charts_df = pd.read_csv("charts.csv")
country_codes = charts_df.name.unique()

# init df
columns = ['codes', 'Country', 'Continent', 'Geolocate', 'Latitude', 'Longitude'];
df = pd.DataFrame(country_codes, columns=['CountryName'])
for col in columns:
    df[col] = None

# init geolocator
geolocator = Nominatim(user_agent="http")

# fill data to df
for index, row in df.iterrows():
    code, continent = get_continent(row['CountryName'])
    df.at[index, 'codes'] = (code, continent)
    df.at[index, 'Country'] = code
    df.at[index, 'Continent'] = continent

    latitude, longtitude = geolocate(row['CountryName'])
    df.at[index, 'Geolocate'] = (latitude, longtitude)
    df.at[index, 'Latitude'] = latitude
    df.at[index, 'Longitude'] = longtitude

# create map
world_map = folium.Map(tiles="cartodbpositron")
marker_cluster = MarkerCluster().add_to(world_map)
# fill map points
for i in range(len(df)):
        lat = df.iloc[i]['Latitude']
        long = df.iloc[i]['Longitude']
        radius = 5
        folium.CircleMarker(location=[lat, long], radius=radius, fill=True).add_to(marker_cluster)

# save map
world_map.save('map.html')
print('finished')
