from scripts.getEEData import getData
from scripts.makePredictions import predictFire

# Define city and country
city = "San Francisco"
countryCode = "US"

# Fetch data from Earth Engine
data = getData(city, countryCode)

# Run prediction
prediction, probability = predictFire([data["NDVI"], data["LST"], data["BURNED_AREA"]])

# Format the output
output_text = f"""
Location: {city}, {countryCode}
NDVI: {data['NDVI']}
LST: {data['LST']}
Burned Area: {data['BURNED_AREA']}
Prediction: {'Fire Risk' if prediction == 1 else 'No Fire Risk'}
Confidence: {probability * 100:.2f}%
"""

# Save to output.txt
with open("output.txt", "w") as file:
    file.write(output_text)

print("Results have been saved to output.txt")