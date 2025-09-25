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


# make a function to use in the web application side
def process_users_initial_prompt(user_prompt_str, user_api_key):
    import requests
    # ping the endpoint to do the initial user search
    base_api_address = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["process_initial_user_prompt_endpoint"]}'
    payload = {"prompt": user_prompt_str,
              "api_key":user_api_key
              }
    r = requests.post(base_api_address, json=payload, timeout=200)
    if r.ok:
        processed_user_prompt = r.json()["processed_user_prompt"]
        return processed_user_prompt

# creating a new function to extract results from API
def search_locations_based_on_prompt_first_pass(user_search_prompt_str, h3_level_res,number_of_results, api_key,lad_filter=None ):
    search_api_endpoint = st.secrets["general"]["api_url_1"]
    site_api_key = st.secrets["general"]["user_admin_api_key"]
    payload = {"prompt": user_search_prompt_str,
        "h3_level_integer":h3_level_res,
        "top_k": number_of_results,
        "api_key": api_key,
        "lad_filter":lad_filter}
    r = requests.post(search_api_endpoint, json=payload, timeout=600)
    if r.ok:
        json_obj = json.loads(r.json()["query_search_results"])

        # st.write(json_obj)
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

        search_query_results_df = pd.DataFrame([location_ids_container,
                                                description_container,
                                                search_results_container,
                                                geometry_container
                                                ]).T
        search_query_results_df.columns = ["location_id","description","search_results","wkt_geom"]
        search_query_results_df["geometry"] = search_query_results_df["wkt_geom"].apply(wkt.loads)
        return search_query_results_df

# creating a new function to extract results from API
def search_locations_based_on_prompt_second_pass(user_search_prompt_str, h3_level_res,number_of_results, api_key,lad_filter=None ):
    search_api_endpoint = st.secrets["general"]["api_url_1"]
    site_api_key = st.secrets["general"]["user_admin_api_key"]
    payload = {"prompt": user_search_prompt_str,
        "h3_level_integer":h3_level_res,
        "top_k": number_of_results,
        "api_key": api_key,
        "lad_filter":lad_filter}
    r = requests.post(search_api_endpoint, json=payload, timeout=600)
    if r.ok:
        json_obj = json.loads(r.json()["query_search_results"])
        # st.write(json_obj)

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

        search_query_results_df = pd.DataFrame([location_ids_container,
                                                description_container,
                                                search_results_container,
                                                geometry_container
                                                ]).T
        search_query_results_df.columns = ["location_id","description","search_results","wkt_geom"]
        search_query_results_df["geometry"] = search_query_results_df["wkt_geom"].apply(wkt.loads)

        return search_query_results_df

# make a function to use in the web application side for the contiguous descriptions endpoint
def get_contiguous_location_descriptions(origin_location,kring_integer, user_api_key):
    import requests
    # ping the endpoint to do the initial user search
    contiguous_desc_api_endpoint = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["contiguous_location_description_endpoint"]}'
    
    payload = {"origin_location": origin_location,
               "kring_integer":kring_integer,
              "api_key":user_api_key
              }
    r = requests.post(contiguous_desc_api_endpoint, json=payload, timeout=600)
    if r.ok:
        processed_location_contiguous_description = r.json()["contiguous_location_description"]
        return processed_location_contiguous_description


def get_ai_evaluation_of_results_df(users_search_original,
                                   results_df,
                                   user_api_key):

    location_evaluator_endpoint = f'{st.secrets["general"]["persistent_api"]}{st.secrets["general"]["location_evaluator_endpoint"]}'
    
    users_search = users_search_original
    results_df = results_df 
    user_api_key = user_api_key
    
    payload = {"user_search": users_search,
               "results_df":results_df.to_json(),
              "api_key":user_api_key
              }
    r = requests.post(location_evaluator_endpoint, json=payload, timeout=1200)
    if r.ok:
        eval_binary_response = r.json()["eval_binary_response"]
        eval_rationale = r.json()["eval_rationale"]
        return eval_binary_response, eval_rationale

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

effort_selection = st.selectbox("Select how much effort you want to designate to your search", ["","quick","moderate","exhaustive"])
if effort_selection=="quick":
    number_of_results = 10
    st.session_state.number_of_results = number_of_results
elif effort_selection=="moderate":
    number_of_results = 20
    st.session_state.number_of_results = number_of_results
elif effort_selection=="exhaustive":
    number_of_results = 50
    st.session_state.number_of_results = number_of_results

col_l, col_run, col_clear, col_r = st.columns([3, 1, 1, 3])

# --- Run button -------------------------------------------------------------
if col_run.button(" ▶  Run", type="primary"):          # nicer label
    if st.session_state.user_search_prompt.strip():
        with st.spinner("Searching..."):
            # st.session_state.df = fetch_recommendations(st.session_state.query)

            if user_search_prompt is not None:


                site_api_key = st.secrets["general"]["user_admin_api_key"]


                initial_prompt_alignment_progress_placeholder = st.empty()
                initial_prompt_alignment_progress_placeholder.info("Searching for your ideal location...")
                time.sleep(3)
                initial_prompt_alignment_progress_placeholder.empty()


                cleaned_description = process_users_initial_prompt(user_search_prompt,site_api_key)

                col1, col2 = st.columns(2) 

                processing_stage_progress_placeholder = st.empty()
                processing_stage_progress_placeholder.info("1/3 - Initial screening...")


                user_query = cleaned_description # this is experimental
                
                if spatial_resolution_for_search=="London - all":

                    h7_desc_df = search_locations_based_on_prompt_first_pass(    
                        user_search_prompt_str=user_query,
                        h3_level_res = 8,
                        number_of_results = 50,
                        api_key = site_api_key,
                        lad_filter=None
                    )
                elif spatial_resolution_for_search=="London - boroughs":
                    h7_desc_df = search_locations_based_on_prompt_first_pass(    
                        user_search_prompt_str=user_query,
                        h3_level_res = 8,
                        number_of_results = 50,
                        api_key = site_api_key,
                        lad_filter=selected_london_borough
                    )

                relevant_locations_to_consider = h7_desc_df["location_id"].head(50)
                st.session_state.relevant_locations_to_consider = relevant_locations_to_consider
                
                # going into stage 2
                processing_stage_progress_placeholder.empty()
                processing_stage_progress_placeholder.info("2/3 - Shortlisting... ")


                if spatial_resolution_for_search=="London - all":
                    search_results_df = search_locations_based_on_prompt_second_pass(
                        user_search_prompt_str=user_query,
                        h3_level_res = 9,
                        number_of_results = number_of_results * 1,
                        api_key = site_api_key,
                        lad_filter=None
                    )
                elif spatial_resolution_for_search=="London - boroughs":
                    search_results_df = search_locations_based_on_prompt_second_pass(
                        user_search_prompt_str=user_query,
                        h3_level_res = 9,
                        number_of_results = number_of_results * 1,
                        api_key = site_api_key,
                        lad_filter=selected_london_borough
                    )


                # merge this back to the geodataframe
                search_results_gdf = gp.GeoDataFrame(search_results_df, geometry = "geometry", crs=27700) 



                if spatial_resolution_for_search == "London - all":

                    search_results_gdf["contig_loc_descriptions"] = search_results_gdf["location_id"].apply(lambda x: get_contiguous_location_descriptions(origin_location=x,
                                                                                                                                                            kring_integer=1,
                                                                                                                                                            user_api_key=site_api_key))



                else:
                        
                    # this will now add on the location descriptions paragraph for immediately contiguous locations
                    search_results_gdf["contig_loc_descriptions"] = search_results_gdf["location_id"].apply(lambda x: get_contiguous_location_descriptions(origin_location=x,
                                                                                                                                                            kring_integer=1,
                                                                                                                                                            user_api_key=site_api_key))
                
                top_k_results_gdf = search_results_gdf.sort_values("search_results", ascending=False).head(number_of_results).reset_index().drop(columns="index")
                top_k_results_gdf = top_k_results_gdf[["location_id","description","contig_loc_descriptions","search_results","wkt_geom"]].reset_index().drop(columns="index")
                st.session_state.top_k_results_gdf = top_k_results_gdf


                processing_stage_progress_placeholder.empty()
                processing_stage_progress_placeholder.info("3/3 - Final checks...")




                # -------------------------
                # a stage will be added here for our custom models to consider all the locations
                # -------------------------

                
                progress_bar = st.progress(0, text=":grey[Evaluating locations!]")

                progress_placeholder = st.empty()

                # st.write(top_k_results_gdf["contig_loc_descriptions"].values[0])
                llm_binary_evalutions_container = []
                llm_rationale_container = []

                for n in range(len(top_k_results_gdf)):
                     
                    eval_binary_response, eval_rationale = get_ai_evaluation_of_results_df(users_search_original=user_search_prompt,
                               results_df=top_k_results_gdf[top_k_results_gdf.index==n].reset_index().drop(columns="index").head(1),
                               user_api_key=site_api_key)
                    llm_binary_evalutions_container.append(eval_binary_response)
                    llm_rationale_container.append(eval_rationale)

                    if float(eval_binary_response)==1 and len(eval_rationale)>10 :
                        progress_placeholder.success(eval_binary_response +":"+eval_rationale)
                        time.sleep(1)
                    elif float(eval_binary_response)==0 and len(eval_rationale)>10:
                        progress_placeholder.warning(eval_binary_response +":"+eval_rationale)
                        time.sleep(1)

                    progress_placeholder.empty()
                    progress_bar.progress(n/(len(top_k_results_gdf)-1))

                progress_bar.empty()


                top_k_results_gdf["ai_model_evaluation"] = llm_binary_evalutions_container
                top_k_results_gdf["ai_model_rationale"] = llm_rationale_container

                # make sure the evaluation is a number
                top_k_results_gdf["ai_model_evaluation"] = top_k_results_gdf["ai_model_evaluation"].astype(float)
                
                # st.dataframe(top_k_results_gdf)
                if len(top_k_results_gdf[top_k_results_gdf["ai_model_evaluation"]==1])>1:
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


                # gdf

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
                        gdf_wgs["viz_results_value"] = round(gdf_wgs["search_results"] * 100,0)
                        # alternative option would be to use pydeck for vizualisations 
                        deck = pdk.Deck(
                                        map_provider="mapbox",
                                        map_style=pdk.map_styles.MAPBOX_LIGHT,

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
                                    if selection_evaluation ==1:
                                        log_search_process_completion(prompt=user_search_prompt,
                                                                        completion=cleaned_description,
                                                                        location_description=location_description,
                                                                        user_api_key=site_api_key,
                                                                        interaction_sentiment=1)

                                time.sleep(2)
                                processing_stage_progress_placeholder.empty()

    else:
        st.warning("Please enter something first!")

# --- Clear button -----------------------------------------------------------
if col_clear.button("⟲ Clear"):
    st.session_state.clear()  # This clears all session state variables
    st.rerun()  # This reruns the entire app          # immediately refresh the page


