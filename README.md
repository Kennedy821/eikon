
<p align="center">
  <img src="https://github.com/Kennedy821/eikon/blob/main/slug_logo.png" alt="eikonai logo" width="500"/>
</p>

# EikonAI

On demand location intelligence.

Eikonai is a lightweight Python SDK for the EIKON Location APIs: compare places, find places that contain X, and monitor change over time—without the hassle of GIS data wrangling.

*   Context - "What does location X contain?"
*	Similarity — “How similar is location X and location Y?”
*	Change — “How much did this place change since then?”
*   Models - Some models to use on demand for object detection and other helpful tasks

Works nicely in data notebooks, Streamlit apps, and backend services.

## Install (dev)
```bash
pip install eikonsai
```


Python: 3.9+
OS: macOS, Linux, Windows

# Usage 

## Quick start

```bash
from eikonsai import utils, context, similarity, models

# set your api_key 
import os


# 1. Use the utils module to get your api_key
my_api_key = utils.get_api_key_from_credentials(email="myregisteredemail",
                                                password="mypassword")
os.environ["user_api_key"] = my_api_key

# 2. Get a location description for a given place 

location_1 = [51.531143, -0.159893] # regent's park
location_1_description = context.get_location_description(lat = location_1[0],
                                lon = location_1[1], 
                                resolution="low",
                                user_api_key=os.environ["user_api_key"]
                                )
print(location_1_description)


# 3. Comparing the visual similarity of two different locations

location_1 = [51.531143, -0.159893] # regent's park
location_2 = [51.433727, -0.214443] # wimbledon tennis club
resolution = "high"
visual_similarity = similarity.visual_similarity(location_1_lat_lon_list=location_1,
                             location_2_lat_lon_list=location_2, 
                             resolution=resolution,
                             user_api_key=os.environ["user_api_key"]
                            )
print(visual_similarity)

```

# Background

All qualifying education email addresses will be provided with free credits for each month for research purposes. To access these credits you must be enrolled on a UK recognised postgraduate degree programme [listed here](https://www.postgrad.com/).
