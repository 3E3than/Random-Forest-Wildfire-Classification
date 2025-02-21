import ee
import joblib
import numpy as np
from geopy.geocoders import Nominatim
from datetime import date, timedelta

#function that gets the coordinates for city, countrycode
def get_city_bounds(city_name, country_code=None):
    geolocator = Nominatim(user_agent="geo_bounds_finder")
    location = geolocator.geocode(city_name if not country_code else f"{city_name}, {country_code}", exactly_one=True)

    if location and "boundingbox" in location.raw:
        min_lat, max_lat, min_lon, max_lon = map(float, location.raw["boundingbox"])
        return {
            "minlat": min_lat,
            "maxlat": max_lat,
            "minlong": min_lon,
            "maxlong": max_lon
        }

    return None


#ENTER CITY AND COUNTRY TO SEARCH HERE
city = "san francisco"
countryCode = "US"
city_bounds = get_city_bounds(city, countryCode)

#initialize and authenticate google earth engine
ee.Authenticate()
ee.Initialize(project='ee-cethan12022')

#get date and city
today_ = date.today() - timedelta(days = 1)
prevWeek_ = today_ - timedelta(days=14)
prevMonth_ = today_ - timedelta(days=45)
today = ee.Date(str(today_))
prevWeek = ee.Date(str(prevWeek_))
prevMonth = ee.Date(str(prevMonth_))

#Example: California; function takes in coords as
#a list of four numbers in the order mlong, mlat, maxlong, maxlat.
AREA = ee.Geometry.Rectangle([city_bounds['minlong'], city_bounds['minlat'], city_bounds['maxlong'], city_bounds['maxlat']])

#NDVI get monthly
ndvi_dataset = ee.ImageCollection("MODIS/061/MOD13Q1") \
    .filterDate(prevMonth, prevWeek) \
    .select('NDVI')
# Scale the NDVI values to be between 0 and 1 (divide by 10000)
scaled_ndvi_dataset = ndvi_dataset.map(lambda image: image.divide(10000))

#LST get weekly
lst_dataset = ee.ImageCollection("MODIS/061/MOD11A1") \
    .filterDate(prevWeek, today) \
    .select('LST_Day_1km')

#BURNED_AREA get weekly
burned_area_dataset = ee.ImageCollection("MODIS/061/MOD14A1") \
    .filterDate(prevWeek, today) \
    .select('FireMask')

'''
maxPixels is high to ensure full coverage if AREA is large
A higher scale(in meters) means lower resolution, but faster processing.
'''
ndvi_mean = scaled_ndvi_dataset.mean().reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=AREA,
    scale=250,
    maxPixels=1e13
)

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
finalNDVI = list(ndvi_mean.getInfo().values())[0]
finalLST = list(lst_mean.getInfo().values())[0]
finalBURNED = list(burned_area_mean.getInfo().values())[0]

model, scaler = joblib.load("models/random_forest_model.joblib"), joblib.load("models/scaler.joblib")
X = np.array([[finalNDVI, finalLST, finalBURNED]])
scaledX = scaler.transform(X)
prediction = model.predict(scaledX)[0]
probability = model.predict_proba(scaledX)[0].max()

# Prepare the data to be written into a text file
output_data = f"""
Location: {city}, {countryCode}
Coords: {city_bounds}
dates: {today_} - {prevWeek_}
NDVI:
{finalNDVI}
LST:
{finalLST}
Burned Area:
{finalBURNED}
Prediction: {prediction} with probability {probability}
"""

# Write the data to a text file
output_file_path = './scriptOutput.txt'

with open(output_file_path, 'w') as file:
    file.write(output_data)

print(f"Data has been written to {output_file_path}")