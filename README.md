# Wildfire Prediction using Earth Engine and Random Forest

## Overview
This project integrates Google Earth Engine and machine learning to analyze and predict wildfire occurrences. It processes remote sensing data, extracts key environmental indicators, and utilizes a pre trained random forest classifier model to predict the wildfire risk. The initial training dataset and data processing approaches were derived from Predictive modeling of wildfires: A new dataset and machine learning approach created by Hajar Mousannif and Hassan Al Moatassime.

## How to Use:
After cloning the repository and downloading the necessary dependencies under requirements.txt, simply head to the main.py file
to input the city and country code that you would like to search. Then, run the script:

```bash 
python3 main.py
```

You should see the results of the predictions under Output.txt