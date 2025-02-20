import ee
from geopy.geocoders import Nominatim


ee.Authenticate()
ee.Initialize(project='ee-cethan12022')

def get_city_bounds(city_name, country_code=None):
    geolocator = Nominatim(user_agent="geo_bounds_finder")
    location = geolocator.geocode(city_name if not country_code else f"{city_name}, {country_code}", exactly_one=True)

    if location and "boundingbox" in location.raw:
        min_lat, max_lat, min_lon, max_lon = map(float, location.raw["boundingbox"])
        return {
            "min_latitude": min_lat,
            "max_latitude": max_lat,
            "min_longitude": min_lon,
            "max_longitude": max_lon
        }

    return None

# Example Usage
city_bounds = get_city_bounds("San Francisco", "US")
print(city_bounds)


#Example: California; function takes in coords as
#a list of four numbers in the order xMin, yMin, xMax, yMax.
AREA = ee.Geometry.Rectangle([-125, 32, -113, 42])

#NDVI
ndvi_dataset = ee.ImageCollection("MODIS/061/MOD13Q1") \
    .filterDate('2024-01-01', '2024-02-01') \
    .select('NDVI')
# Scale the NDVI values to be between 0 and 1 (divide by 10000)
scaled_ndvi_dataset = ndvi_dataset.map(lambda image: image.divide(10000))

#LST
lst_dataset = ee.ImageCollection("MODIS/061/MOD11A1") \
    .filterDate('2024-01-01', '2024-02-01') \
    .select('LST_Day_1km')

#BURNED_AREA
burned_area_dataset = ee.ImageCollection("MODIS/061/MOD14A1") \
    .filterDate('2024-01-01', '2024-02-01') \
    .select('FireMask')

# Get the mean NDVI for the time range over the AREA
ndvi_mean = scaled_ndvi_dataset.mean().reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=AREA,
    scale=250,
    maxPixels=1e13
)

'''
maxPixels is high to ensure full coverage if AREA is large
A higher scale(in meters) means lower resolution, but faster processing.
'''

lst_mean = lst_dataset.mean().reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=AREA,
    scale=1000,
    maxPixels=1e13
)

burned_area_mean = burned_area_dataset.mean().reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=AREA,
    scale=1000,  # 1 km resolution
    maxPixels=1e13
)

# fetches info
ndvi_mean_info = ndvi_mean.getInfo()
lst_mean_info = lst_mean.getInfo()
burned_area_mean_info = burned_area_mean.getInfo()

# Prepare the data to be written into a text file
output_data = f"""
NDVI:
{ndvi_mean_info}

LST:
{lst_mean_info}

Burned Area:
{burned_area_mean_info}
"""

# Write the data to a text file
output_file_path = './scriptOutput.txt'

with open(output_file_path, 'w') as file:
    file.write(output_data)

print(f"Data has been written to {output_file_path}")