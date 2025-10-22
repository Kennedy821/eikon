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
import shapely
import numpy as np
import time
# import folium
import os
import re
import pydeck as pdk
import asyncio
import aiohttp
import threading
# define some functions 
def clear_inputs() -> None:
    """Reset the text box and forget the previous results."""
    st.session_state["spatial_resolution_for_search"] = ""      # resets <input>
    st.session_state["selected_london_borough"] = ""      # resets <input>
    st.session_state["user_search_prompt"] = ""      # resets <input>
    st.session_state["number_of_results"] = ""      # resets <input>
    st.session_state["df_results"] = ""      # resets <input>
    st.session_state["init_gdf"] = ""      # resets <input>
    st.session_state["relevant_locations_to_consider"] = ""      # resets <input>
    st.session_state["top_k_results_gdf"] = ""      # resets <input>


def flatten_list(nested_list):
    """
    Recursively flattens a nested list.

    Args:
        nested_list (list): The list to flatten.

    Returns:
        list: A flattened version of the input list.
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def log_search_process_completion(prompt,
                                   completion,
                                   location_description,
                                   user_api_key,
                                 interaction_sentiment):

    location_evaluator_endpoint = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["logging_completion_search_endpoint"]}'
    
    users_search = prompt
    completion = completion 
    location_description = location_description
    user_api_key = user_api_key
    interaction_sentiment = interaction_sentiment
    
    payload = {"prompt": prompt,
               "completion":completion,
               "location_description":location_description,
               "api_key":user_api_key,
               "interaction_sentiment":interaction_sentiment
              }
    r = requests.post(location_evaluator_endpoint, json=payload, timeout=120)
    if r.ok:
        return 1

# make a function to use in the web application side
def detect_objects_at_location(location_id, user_api_key):
    import requests
    # ping the endpoint to do the initial user search
    base_api_address = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["object_detection_endpoint"]}'
    payload = {"location_id": location_id,
              "api_key":user_api_key
              }
    r = requests.post(base_api_address, json=payload, timeout=200)
    if r.ok:
        objects_found = r.json()["objects"]
        return objects_found



def search_api(
                my_search_prompt,
                user_api_key,
                effort_selection,
                spatial_resolution_for_search,
                selected_london_borough = None
                ): 
    
    import requests
    import pandas as pd
    # ping the endpoint to do the initial user search
    base_api_address = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["search_endpoint"]}'
    payload = {
            "prompt": my_search_prompt,
            "api_key":user_api_key,
            "effort_selection":effort_selection,
            "spatial_resolution_for_search": spatial_resolution_for_search,
            "selected_london_borough":selected_london_borough,
    }
        
    if spatial_resolution_for_search == "London - all" and selected_london_borough is None:
   
        r = requests.post(base_api_address, json=payload, timeout=1000)
        if r.ok:
            results_json = r.json()["successful_job_completion"]
            results_df = pd.DataFrame.from_dict(json.loads(results_json))
            return results_df
    elif spatial_resolution_for_search != "London - all" and selected_london_borough is not None:
        r = requests.post(base_api_address, json=payload, timeout=1000)
        if r.ok:
            results_json = r.json()["successful_job_completion"]
            results_df = pd.DataFrame.from_dict(json.loads(results_json))
            return results_df
    else:
        return ("You have made an incompatible query")

import asyncio
import aiohttp

async def search_api_async(
                my_search_prompt,
                user_api_key,
                effort_selection,
                spatial_resolution_for_search,
                selected_london_borough = None
                ): 
    
    import requests
    import pandas as pd
    # ping the endpoint to do the initial user search
    base_api_address = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["search_endpoint_web"]}'
    payload = {
            "prompt": my_search_prompt,
            "api_key":user_api_key,
            "effort_selection":effort_selection,
            "spatial_resolution_for_search": spatial_resolution_for_search,
            "selected_london_borough":selected_london_borough,
    }
        
    if spatial_resolution_for_search == "London - all" and selected_london_borough is None:
   
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(base_api_address, json=payload) as resp:
                resp.raise_for_status()                  # raises on 4xx/5xx
                data = await resp.json()         
        return "job_triggered" 

    elif spatial_resolution_for_search != "London - all" and selected_london_borough is not None:

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(base_api_address, json=payload) as resp:
                resp.raise_for_status()                  # raises on 4xx/5xx
                data = await resp.json()                
        return "job_triggered"  
    else:
        return ("You have made an incompatible query")

import threading
import requests

def _post_in_background(url, payload):
    try:
        requests.post(url, json=payload, timeout=5)  # short timeout to avoid blocking
    except Exception as e:
        print("POST failed:", e)

def search_api_async(
    my_search_prompt,
    user_api_key,
    effort_selection,
    spatial_resolution_for_search,
    selected_london_borough=None
):
    import requests
    import pandas as pd
    # ping the endpoint to do the initial user search
    base_api_address = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["search_endpoint_web"]}'
    payload = {
        "prompt": my_search_prompt,
        "api_key": user_api_key,
        "effort_selection": effort_selection,
        "spatial_resolution_for_search": spatial_resolution_for_search,
        "selected_london_borough": selected_london_borough,
    }

    # Launch the POST in a separate daemon thread — does not block
    threading.Thread(
        target=_post_in_background,
        args=(base_api_address, payload),
        daemon=True
    ).start()

    # Return immediately without waiting for the POST
    return "job_triggered"

# make client side function to get latest results 
def client_get_last_search_results(api_key):
    base_api_address = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["get_latest_results_endpoint"]}'
    payload = {
        "api_key":api_key
    }
    r = requests.post(base_api_address, json=payload, timeout=120)
    if r.ok:
        latest_results_json = r.json()["latest_search_results"]
        results_df = pd.DataFrame.from_dict(json.loads(latest_results_json))
        return results_df
    
# make client side function to get latest results 
def client_get_last_search_results_many(api_key, num_searches=5):
    base_api_address = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["get_latest_results_many_endpoint"]}'
    payload = {
        "api_key":api_key,
        "num_searches":num_searches
    }
    r = requests.post(base_api_address, json=payload, timeout=120)
    # st.write(r)
    if r.ok:
        latest_results_json = r.json()["latest_search_results"]
        list_of_results_json = latest_results_json.split("|")
        df_container = []
        for result_json in list_of_results_json:
            results_df = pd.DataFrame.from_dict(json.loads(result_json))
            results_df["search_instance_id"] = "search_"+str(len(df_container)+1)
            df_container.append(results_df)
        results_df = pd.concat(df_container).reset_index().drop(columns="index")
        return results_df



def client_check_for_completed_job(api_key):
    base_api_address = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["check_job_completion_endpoint"]}'
    payload = {
        "api_key":api_key
    }
    r = requests.post(base_api_address, json=payload, timeout=120)
    if r.ok:
        job_status = r.json()["job_complete"]
        # st.write(job_status)
        if job_status==1:
            return "completed_job_found"
        else:
            latest_ckpt = r.json()["latest_ckpt"]
            return f"no_completed_job_found_{latest_ckpt}"

def client_model_thoughts_inspection(api_key):
    base_api_address = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["inspect_search_model_endpoint"]}'
    payload = {
        "api_key": api_key
    }
    r = requests.post(base_api_address, json=payload, timeout=360)
    # st.write(r)
    if r.ok:
        # Return the actual model thoughts content
        # st.write(r.json())
        latest_ckpt = r.json()["latest_ckpt"]
        return latest_ckpt
    else:
        return None
        
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

if 'relevant_locations_to_consider' not in st.session_state:
    st.session_state.relevant_locations_to_consider = None

if 'top_k_results_gdf' not in st.session_state:
    st.session_state.top_k_results_gdf = None

im = Image.open('slug_logo.png')
# Set page config
st.set_page_config(
    page_title="EIKON",
    page_icon=im,
    layout="wide"
)
# Inject CSS for banner
st.markdown(
    """
    <style>
    .banner {
        background-image: url("https://picsum.photos/id/162/1200/200?grayscale");
        background-size: cover;
        background-position: center;
        height: 200px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# we're going to add another tab
tab1, tab2 = st.tabs(["Search", "Previous searches"])

with tab1:
    # Render the banner
    st.markdown('<div class="banner"></div>', unsafe_allow_html=True)
    st.title(f"EIKON - alpha app")

    spatial_resolution_for_search = st.selectbox("Select the level you want to search for things", ["","London - all", "London - boroughs"])
    st.session_state.spatial_resolution_for_search = spatial_resolution_for_search

    if spatial_resolution_for_search != "London - all":

        # load list of london boroughs
        boro_mapping_df = pd.read_parquet("london_boroughs.parquet.gzip")
        # st.dataframe(boro_mapping_df.head().collect())
        london_boros = sorted(boro_mapping_df["lad11nm"].unique().tolist())

        
        # get list of London Boroughs
        list_of_london_boroughs = [""]+sorted(london_boros)
        selected_london_borough = st.selectbox("Select which London borough you're interested in", list_of_london_boroughs)
        st.session_state.selected_london_borough = selected_london_borough

        pass

    user_search_prompt = st.text_input("Type what you'd like to search for here...")
    st.session_state.user_search_prompt = user_search_prompt

    effort_selection = st.selectbox("Select how much effort you want to designate to your search", ["","test","quick","moderate","exhaustive"])
    if effort_selection=="quick":
        number_of_results = 10
        st.session_state.number_of_results = number_of_results
    elif effort_selection=="moderate":
        number_of_results = 20
        st.session_state.number_of_results = number_of_results
    elif effort_selection=="exhaustive":
        number_of_results = 50
        st.session_state.number_of_results = number_of_results
    elif effort_selection=="test":
        number_of_results = 3
        st.session_state.number_of_results = number_of_results
    col_l, col_run, col_clear, col_r = st.columns([3, 1, 1, 3])

    # --- Run button -------------------------------------------------------------
    if col_run.button(" ▶  Run", type="primary"):          # nicer label
        if st.session_state.user_search_prompt.strip():
            with st.spinner("Searching..."):
                # st.session_state.df = fetch_recommendations(st.session_state.query)

                if user_search_prompt is not None:






                    site_api_key = st.secrets["general"]["user_admin_api_key"]

                    if spatial_resolution_for_search == "London - all":
                        top_k_results_gdf = search_api_async(my_search_prompt=user_search_prompt,
                                                        user_api_key=site_api_key,
                                                        effort_selection=effort_selection,
                                                        spatial_resolution_for_search=spatial_resolution_for_search)
                    elif spatial_resolution_for_search != "London - all" and selected_london_borough is not None:
                        top_k_results_gdf = search_api_async(my_search_prompt=user_search_prompt,
                                                        user_api_key=site_api_key,
                                                        effort_selection=effort_selection,
                                                        spatial_resolution_for_search=spatial_resolution_for_search,
                                                        selected_london_borough=selected_london_borough)
                        
                    
                        

                    processing_stage_progress_placeholder = st.empty()

                    processing_stage_progress_placeholder.info(top_k_results_gdf)
                    time.sleep(7)
                    processing_stage_progress_placeholder.empty()
                    processing_stage_progress_placeholder.info("Initiating search... This may take a few minutes depending on the effort level you've selected.")
                    time.sleep(30)
                    processing_stage_progress_placeholder.empty()

                    # now we're going to check if the job is completed
                    exit_status = 0
                    prev_ckpt_completed = "No checkpoints yet"
                    prev_model_cot = "Not started"

                    model_cot_inspector = st.empty()
                    while exit_status==0:
                        try:
                            job_completed = client_check_for_completed_job(api_key=site_api_key)
                            if job_completed!="completed_job_found":
                                # get the latest checkpoint
                                latest_ckpt_completed = job_completed.split("_found_")[-1]
                                ckpt_message = latest_ckpt_completed.split("complete")[0].replace("_"," ").strip()[:1].upper() + latest_ckpt_completed.split("complete")[0].replace("_"," ").strip()[1:].lower() + " : " + latest_ckpt_completed.split("complete")[-1].split(".txt")[0].replace("_"," ").strip()[:1].upper() + latest_ckpt_completed.split("complete")[-1].split(".txt")[0].replace("_"," ").strip()[1:].lower()
                                processing_stage_progress_placeholder.info(ckpt_message)
    
                                # Poll model thoughts every loop while we're in Stage 4 (not only when the checkpoint text changed)
                                if "Stage 4" in ckpt_message:
                                    if "Not started" in prev_model_cot:
                                        model_cot_inspector.empty()
                                        model_cot_inspector.info("Model is now evaluating locations... This may take a few minutes.")
                                        time.sleep(1)
    
    
                                    # Poll the model thoughts endpoint every iteration
                                    current_model_cot = client_model_thoughts_inspection(api_key=site_api_key)
    
                                    if current_model_cot is not None:
                                        current_model_cot_eval = current_model_cot.lower().split("rationale:")[0]
                                        current_model_cot = current_model_cot.lower().split("rationale:")[-1]
                                        current_model_cot = current_model_cot.replace("_"," ").strip()[:1].upper() + current_model_cot.replace("_"," ").strip()[1:].lower()
    
                                        if current_model_cot != prev_model_cot:
                                            model_cot_inspector.empty()
                                            if "1" in current_model_cot_eval:
                                                model_cot_inspector.success(current_model_cot)
                                            else:
                                                model_cot_inspector.info(current_model_cot)
                                            prev_model_cot = current_model_cot
                                if "Stage 4" not in ckpt_message:
                                    model_cot_inspector.empty() 
    
                                if ckpt_message != prev_ckpt_completed:
                                    processing_stage_progress_placeholder.empty()
    
                                    time.sleep(1)
                                    processing_stage_progress_placeholder.info(ckpt_message)
                                    prev_ckpt_completed = ckpt_message
    
                                    if "Stage 4" in ckpt_message:
                                        if "Not started" in prev_model_cot:
                                            model_cot_inspector.empty()
                                            model_cot_inspector.info("Model is now evaluating locations... This may take a few minutes.")
                                                                        
                                        time.sleep(5)  # Add delay between inspection calls
                                        current_model_cot = client_model_thoughts_inspection(api_key=site_api_key)
    
                                        if current_model_cot is not None and "_found_" in current_model_cot and "rationale:" in current_model_cot:
                                            current_model_cot_eval = current_model_cot.split("_found_")[-1].split("rationale:")[0]
                                            current_model_cot = current_model_cot.split("_found_")[-1].split("rationale:")[-1]
                                            current_model_cot = current_model_cot.replace("_"," ").strip()[:1].upper() + current_model_cot.replace("_"," ").strip()[1:].lower()
    
                                            if current_model_cot != prev_model_cot:
                                                model_cot_inspector.empty()
    
                                                if "1" in current_model_cot_eval:
                                                    model_cot_inspector.success(current_model_cot)
                                                else:
                                                    model_cot_inspector.info(current_model_cot)
    
                                                prev_model_cot = current_model_cot
                            
                                time.sleep(10)
                        except Exception as e:
                            pass
                        else:
                            exit_status=1
                    processing_stage_progress_placeholder.success("Search completed!")
                    processing_stage_progress_placeholder.empty()
                    model_cot_inspector.empty()

                    time.sleep(3)

                    # once the job has completed we'll collect the results
                    top_k_results_gdf = client_get_last_search_results(api_key=site_api_key)


                    # -------------------------
                    # a stage will be added here for our custom models to consider all the locations
                    # -------------------------
                    processing_stage_progress_placeholder = st.empty()
                    # processing_stage_progress_placeholder.info("1/3 - Initial screening...")

                    # processing_stage_progress_placeholder.empty()
                    # processing_stage_progress_placeholder.info("2/3 - Shortlisting... ")

                    # processing_stage_progress_placeholder.empty()
                    # processing_stage_progress_placeholder.info("3/3 - Final checks...")
                    

                    progress_placeholder = st.empty()

                
                    # make sure the evaluation is a number
                    # top_k_results_gdf["ai_model_evaluation"] = top_k_results_gdf["ai_model_evaluation"].str.strip().astype(float)
                    
                    # st.dataframe(top_k_results_gdf)
                    if len(top_k_results_gdf[top_k_results_gdf["ai_model_evaluation"]==1])>0:
                        top_k_results_gdf = top_k_results_gdf[top_k_results_gdf["ai_model_evaluation"]==1].reset_index().drop(columns="index")

                    else:
                        progress_placeholder.error("It looks like there weren't any suitable locations found that match what you're looking for.")


                    if st.session_state.init_gdf is None:
                        top_k_results_gdf["geometry"] = top_k_results_gdf["wkt_geom"].apply(wkt.loads)
                        gdf = gp.GeoDataFrame(top_k_results_gdf, geometry="geometry", crs=27700).to_crs(4326).set_index("location_id")
                    else:
                        gdf = st.session_state.init_gdf

                    processing_stage_progress_placeholder.empty()

                    processing_stage_progress_placeholder.success("Here are your results...")


                    col1, col2 = st.columns(2) 

                    with col1:

                        # Ensure WGS84 for map tiles + get centroids for point-based layers
                        gdf_wgs = gdf.to_crs(4326).copy()
                        gdf_wgs["lat"] = gdf_wgs.geometry.centroid.y
                        gdf_wgs["lon"] = gdf_wgs.geometry.centroid.x
                        gdf_wgs["coordinates"] = gdf_wgs.apply(lambda x: [x["lon"],x["lat"]], axis=1)
                        map_element_status = st.empty()
                        try:

                            # fig = px.scatter_mapbox(
                            # gdf_wgs,
                            # lat="lat", lon="lon",
                            # color="search_results",
                            # color_continuous_scale=px.colors.cyclical.IceFire,
                            # # size="search_results",
                            # center={"lat": gdf.geometry.centroid.y.mean(), "lon": gdf.geometry.centroid.x.mean()},
                            # size_max=30,
                            # # color_continuous_scale="Oranges",
                            # mapbox_style="carto-positron",
                            # zoom=8, opacity=0.8, height=600
                            # )

                            # st.plotly_chart(fig)
                            gdf_wgs = gdf_wgs.reset_index()
                            gdf_wgs["search_results"] = gdf_wgs["search_results"].astype(float)
                            gdf_wgs["viz_results_value"] = round(gdf_wgs["search_results"] * 100,0)
                            # alternative option would be to use pydeck for vizualisations 
                            deck = pdk.Deck(
                                            map_provider="carto",
                                            map_style=pdk.map_styles.CARTO_LIGHT,

                                            # map_style=pdk.map_styles.SATELLITE

                                            initial_view_state = pdk.ViewState(
                                            latitude=gdf_wgs.geometry.centroid.y.mean(),
                                            longitude=-gdf_wgs.geometry.centroid.x.mean(),
                                            zoom=8,
                                            pitch=0,
                                            bearing=0,
                                            height=1000,width='100%'
                                            ),

                                            layers = [
                                                    pdk.Layer(
                                                    "ScatterplotLayer",
                                                    gdf_wgs,
                                                    pickable=True,
                                                    opacity=0.8,
                                                    stroked=True,
                                                    filled=True,
                                                    radius_scale=6,
                                                    radius_min_pixels=1,
                                                    radius_max_pixels=100,
                                                    line_width_min_pixels=1,
                                                    get_position="coordinates",
                                                    get_radius="viz_results_value",
                                                    get_fill_color=[255, 140, 0],
                                                    get_line_color=[0, 0, 0],
                                                )
                                                ],
                                                tooltip={
                                    # "html": "<b>Hex cell:</b> {h3_index} <br/> Constituency: {Constituency} <br/>Winner: {Party}"
                                    "html": "Location: {location_id} <br/>Site is relevant: {ai_model_evaluation} <br/>Summary: {ai_model_rationale}"

                                }
                            )

                            
                            st.pydeck_chart(deck)

                        except Exception as e:
                            st.error(e)
                            map_element_status.error("There was an issue rendering your map...")


                        

                        with col2:


                            gdf = gdf.reset_index()

                            
                            for location_idx in range(len(gdf)):
                                with st.expander(f"Location: {location_idx+1}"):
                                    viz_col1, viz_col2 = st.columns([3,8])

                                    location_lat_coord = gdf[gdf.index==location_idx].geometry.centroid.y.mean()
                                    location_lon_coord = gdf[gdf.index==location_idx].geometry.centroid.x.mean()
                                    location_id = gdf[gdf.index==location_idx]["location_id"].values[0]
                                    print(location_lat_coord,location_lon_coord)

                                    with viz_col1:
                                            st.write(f"Selected Location: {location_idx+1}")
                                            st.write(f"{location_id}")
                                            # load image of location 
                                            
                                            # this is for wimbledon low resolution 
                                            payload = {"text": "placeholder",
                                                    "location": [location_lat_coord, location_lon_coord],
                                                    "resolution":"high",
                                                    "api_key": site_api_key}
                                            r = requests.post(st.secrets["general"]["api_url_2"], json=payload, timeout=120)
                                            if r.ok:
                                                b64_str = r.json()["location_image"]
                                                # 2) —— DECODE bytes -------------------------------------------------
                                                img_bytes = base64.b64decode(b64_str)

                                                # 3) —— LOAD into an image object -----------------------------------
                                                # Pillow can open from a bytes-buffer
                                                img = Image.open(io.BytesIO(img_bytes))
                                                st.image(img)
                                    with viz_col2:
                                        
                                        st.write(f"Location description")
                                        location_description = gdf[gdf.index==location_idx]['description'].values[0]
                                        selection_rationale = gdf[gdf.index==location_idx]['ai_model_rationale'].values[0]
                                        selection_evaluation = gdf[gdf.index==location_idx]["ai_model_evaluation"].values[0]
                                        st.write(f"{location_description}")
                                        st.write(f"{selection_rationale}")
    

                                    time.sleep(2)
                                    processing_stage_progress_placeholder.empty()

        else:
            st.warning("Please enter something first!")

    # --- Clear button -----------------------------------------------------------
    if col_clear.button("⟲ Clear"):
        st.session_state.clear()  # This clears all session state variables
        st.rerun()  # This reruns the entire app          # immediately refresh the page

# ensure guard exists
if 'load_previous_searches' not in st.session_state:
    st.session_state.load_previous_searches = False

with tab2:
    # this is going to be the tab to get your previous searches
    st.markdown('<div class="banner"></div>', unsafe_allow_html=True)
    st.title("EIKON - previous searches")

    # Show a button to explicitly load previous searches to avoid running on tab render
    if not st.session_state.load_previous_searches:
        if st.button("Load previous searches", key="load_prev_button"):
            st.session_state.load_previous_searches = True
            st.rerun()   # optional: refresh so UI shows loaded state
        else:
            st.info("Click 'Load previous searches' to fetch results.")
    else:
        site_api_key = st.secrets["general"]["user_admin_api_key"]

        # we will call the endpoint to get previous searches
        previous_searches_df = client_get_last_search_results_many(api_key=site_api_key, num_searches=3)
        # figure out how many unique searches there are
        unique_search_instances = previous_searches_df["search_instance_id"].unique().tolist()

        # now we're going to make a container with each one
        for search_result in unique_search_instances:
            with st.expander(f"Previous search: {search_result}"):
                progress_placeholder = st.empty()

                top_k_results_gdf = previous_searches_df[previous_searches_df["search_instance_id"]==search_result].reset_index().drop(columns="index")
                if len(top_k_results_gdf[top_k_results_gdf["ai_model_evaluation"]==1])>0:
                    top_k_results_gdf = top_k_results_gdf[top_k_results_gdf["ai_model_evaluation"]==1].reset_index().drop(columns="index")
                else:
                    progress_placeholder.error("It looks like there weren't any suitable locations found that match what you're looking for.")

                if st.session_state.init_gdf is None:
                    top_k_results_gdf["geometry"] = top_k_results_gdf["wkt_geom"].apply(wkt.loads)
                    gdf = gp.GeoDataFrame(top_k_results_gdf, geometry="geometry", crs=27700).to_crs(4326).set_index("location_id")
                else:
                    gdf = st.session_state.init_gdf

                gdf = gdf.reset_index()

                for location_idx in range(len(top_k_results_gdf)):
                    with st.container(border=True):
                        st.write(f"Location: {location_idx+1}")

                        viz_col1, viz_col2 = st.columns([3,8])

                        location_lat_coord = gdf[gdf.index==location_idx].geometry.centroid.y.mean()
                        location_lon_coord = gdf[gdf.index==location_idx].geometry.centroid.x.mean()
                        location_id = gdf[gdf.index==location_idx]["location_id"].values[0]

                        with viz_col1:
                            st.write(f"Selected Location: {location_idx+1}")
                            st.write(f"{location_id}")
                            payload = {"text": "placeholder",
                                       "location": [location_lat_coord, location_lon_coord],
                                       "resolution":"high",
                                       "api_key": site_api_key}
                            r = requests.post(st.secrets["general"]["api_url_2"], json=payload, timeout=120)
                            if r.ok:
                                b64_str = r.json()["location_image"]
                                img_bytes = base64.b64decode(b64_str)
                                img = Image.open(io.BytesIO(img_bytes))
                                st.image(img)
                        with viz_col2:
                            st.write("Location description")
                            location_description = gdf[gdf.index==location_idx]['description'].values[0]
                            selection_rationale = gdf[gdf.index==location_idx]['ai_model_rationale'].values[0]
                            selection_evaluation = gdf[gdf.index==location_idx]["ai_model_evaluation"].values[0]
                            st.write(f"{location_description}")
                            st.write(f"{selection_rationale}")
