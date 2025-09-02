# this is the first version of the eikon application

import streamlit as st
import json
import requests
import plotly.express as px
from shapely import wkt
import geopandas as gp
import base64, io, requests
from PIL import Image
import pandas as pd

# define some functions 
def clear_inputs() -> None:
    """Reset the text box and forget the previous results."""
    st.session_state["spatial_resolution_for_search"] = ""      # resets <input>
    st.session_state["selected_london_borough"] = ""      # resets <input>
    st.session_state["user_search_prompt"] = ""      # resets <input>
    st.session_state["number_of_results"] = ""      # resets <input>

    # st.session_state["df"] = None       # hides the cards
    # st.session_state["mic_audio"] = None




# Initialize session state variables
if 'spatial_resolution_for_search' not in st.session_state:
    st.session_state.spatial_resolution_for_search = None

if 'selected_london_borough' not in st.session_state:
    st.session_state.selected_london_borough = None

if 'user_search_prompt' not in st.session_state:
    st.session_state.user_search_prompt = None

if 'number_of_results' not in st.session_state:
    st.session_state.number_of_results = None

if 'df_results' not in st.session_state:
    st.session_state.df_results = None

if 'init_gdf' not in st.session_state:
    st.session_state.init_gdf = None

# Set page config
st.set_page_config(
    page_title="EIKON",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ EIKON - alpha app")


# we're going to all a dropdown to select which model the user would like to use
# the options at the moment are gemma3 and llama3

spatial_resolution_for_search = st.selectbox("Select the level you want to search for things", ["","London - all", "London - boroughs"])
st.session_state.spatial_resolution_for_search = spatial_resolution_for_search

if spatial_resolution_for_search != "London - all":

    # get list of London Boroughs
    list_of_london_boroughs = ["","Havering","Southwark"]
    selected_london_borough = st.selectbox("Select which London borough you're interested in", list_of_london_boroughs)
    st.session_state.selected_london_borough = selected_london_borough

    pass

user_search_prompt = st.text_input("Type what you'd like to search for here...")
st.session_state.user_search_prompt = user_search_prompt

number_of_results = st.selectbox("Select how many results you'd like", ["",10,20,50])
st.session_state.number_of_results = number_of_results

col_l, col_run, col_clear, col_r = st.columns([3, 1, 1, 3])

# --- Run button -------------------------------------------------------------
if col_run.button(" ▶  Run", type="primary"):          # nicer label
    if st.session_state.user_search_prompt.strip():
        with st.spinner("Searching..."):
            # st.session_state.df = fetch_recommendations(st.session_state.query)

            if user_search_prompt is not None:

                col1, col2 = st.columns(2) 

                search_api_endpoint = st.secrets["general"]["api_url_1"]

                payload = {"prompt": st.session_state.user_search_prompt,
                    "h3_level_integer":9,
                    "top_k": number_of_results}
                r = requests.post(search_api_endpoint, json=payload, timeout=120)
                if r.ok:
                    json_obj = json.loads(r.json()["query_search_results"])
                    # unpack json response
                    location_ids_container = []
                    description_container = []
                    search_results_container = []
                    geometry_container = []

                    counter = 0
                    for i in json_obj:
                        if counter ==0:
                            for x in range(len(json_obj[i])):
                                # location_ids = pd.json_normalize(json_obj[i])
                                location_ids_container.append(json_obj[i][str(x)])
                        elif counter ==1:
                            for x in range(len(json_obj[i])):
                                description_container.append(json_obj[i][str(x)])     
                        elif counter ==2:
                            for x in range(len(json_obj[i])):
                                search_results_container.append(json_obj[i][str(x)])   
                        elif counter ==3:
                            for x in range(len(json_obj[i])):
                                geometry_container.append(json_obj[i][str(x)])   
                        counter += 1
                    if st.session_state.df_results is None:

                        search_query_results_df = pd.DataFrame([location_ids_container,
                                                                description_container,
                                                                search_results_container,
                                                                geometry_container
                                                                ]).T
                        search_query_results_df.columns = ["location_id","description","search_results","wkt_geom"]
                        search_query_results_df["geometry"] = search_query_results_df["wkt_geom"].apply(wkt.loads)
                        st.session_state.df_results = search_query_results_df

                    else:
                        search_query_results_df = st.session_state.df_results

                    if st.session_state.init_gdf is None:
                        gdf = gp.GeoDataFrame(search_query_results_df, geometry = "geometry", crs=27700).to_crs(4326)
                        gdf["search_results"] = gdf["search_results"].astype(float)
                        gdf = gdf.set_index("location_id")
                        st.session_state.init_gdf = gdf                        
                    else:
                        gdf = st.session_state.init_gdf

                    with col1:

                        # generate a map with the results from the users search
                        # search_query_results_df
                        # # Create a Plotly Express map
                        fig = px.choropleth_mapbox(
                            gdf,
                            geojson=gdf.geometry,
                            locations=gdf.index,
                            color='search_results',
                            center={"lat": gdf.geometry.centroid.y.mean(), "lon": gdf.geometry.centroid.x.mean()},
                            mapbox_style="carto-positron",
                            zoom=8,
                            opacity=0.8,
                            color_continuous_scale="Oranges",
                            height=600

                        )

                        # to revmove the border of the polygons:
                        fig.update_traces(marker_line_width=0)

                        fig

                        with col2:


                            gdf = gdf.reset_index()

                            for location_idx in range(len(gdf)):
                                with st.expander(f"Location: {location_idx+1}"):
                                    viz_col1, viz_col2 = st.columns([3,8])

                                    location_lat_coord = gdf[gdf.index==location_idx].geometry.centroid.y.mean()
                                    location_lon_coord = gdf[gdf.index==location_idx].geometry.centroid.x.mean()
                                    print(location_lat_coord,location_lon_coord)

                                    with viz_col1:
                                        st.write(f"Selected Location: {location_idx}")
                                        # load image of location 
                                        
                                        # this is for wimbledon low resolution 
                                        payload = {"text": "placeholder",
                                                "location": [location_lat_coord, location_lon_coord],
                                                "resolution":"medium"}
                                        r = requests.post(st.secrets["general"]["api_url_2"], json=payload, timeout=120)
                                        if r.ok:
                                            b64_str = r.json()["location_image"]
                                            # 2) —— DECODE bytes -------------------------------------------------
                                            img_bytes = base64.b64decode(b64_str)

                                            # 3) —— LOAD into an image object -----------------------------------
                                            # Pillow can open from a bytes-buffer
                                            img = Image.open(io.BytesIO(img_bytes))
                                            st.image(img)

                                            # st.image(img, width=256)
                                    with viz_col2:
                                        
                                        st.write(f"Location description")
                                        location_description = gdf[gdf.index==location_idx]['description'].values[0]
                                        st.write(f"{location_description}")

                                    
                        #             pass

                        # except Exception as e:
                    
                        #     pass        
                else:
                    print(r.status_code, r.text)
                    st.stop()










    else:
        st.warning("Please enter something first!")

# --- Clear button -----------------------------------------------------------
if col_clear.button("⟲ Clear"):
    st.session_state.clear()  # This clears all session state variables
    st.rerun()  # This reruns the entire app          # immediately refresh the page


