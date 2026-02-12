"""
EIKON Streamlit Application
A comprehensive web application for interacting with the Eikon satellite imagery APIs.

Features:
- Search API: Natural language search across London locations
- Context Module: Get descriptions of any location
- Similarity Module: Compare visual similarity between locations
- Portfolio Comparison: Batch compare multiple location pairs
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time
import io
import os
import base64
import requests
import json
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image as PILImage
from io import BytesIO

# Try to import eikonsai - if not available, we'll use mock functions for demo
try:
    import eikonsai as eikon
    EIKON_AVAILABLE = True
except ImportError:
    EIKON_AVAILABLE = False

# Import h3 for converting H3 cell IDs to lat/lon coordinates
try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False

import PIL.Image as Image
im = Image.open('eikon_logo_tes_v4.png')

# API endpoints
EIKON_API_BASE_URL = "https://slugai.pagekite.me"
EIKON_API_ENDPOINTS = {
    "base_url": EIKON_API_BASE_URL,
    "check_credits": f"{EIKON_API_BASE_URL}/check_eikon_api_credits",
    "search_queue": f"{EIKON_API_BASE_URL}/eikon_search_agent_api_queue",
    "objects_detected": f"{EIKON_API_BASE_URL}/get_objects_detected_in_location",
    "yolo_detection": f"{EIKON_API_BASE_URL}/yolo_object_detection_on_image",
    "check_job_complete": f"{EIKON_API_BASE_URL}/check_if_eikon_search_agent_api_job_complete_web",
}

# Page configuration
st.set_page_config(
    page_title="EIKON",
    page_icon=im,
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        font-weight: 500;
    }
    .credits-box {
        position: fixed;
        top: 60px;
        right: 20px;
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        color: #1E3A5F;
        padding: 12px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(30, 58, 95, 0.12), 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid rgba(30, 58, 95, 0.1);
        z-index: 1000;
        min-width: 140px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .credits-box .credits-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6b7280;
        margin-bottom: 4px;
    }
    .credits-box .credits-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E3A5F;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .credits-box .credits-currency {
        font-size: 0.85rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-right: 2px;
    }
</style>
""", unsafe_allow_html=True)


# Session state initialization
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'authenticated': False,
        'api_key': None,
        'user_email': None,
        'search_results': None,
        'search_history': [],
        'search_in_progress': False,  # Track if search is currently running
        'context_results': None,
        'similarity_results': None,
        'comparison_results': None,
        'portfolio_results': None,
        'portfolio_input_df': None,
        'object_detection_results': None,
        # AI Chat state
        'chat_messages': [],  # List of {"role": "user"|"assistant", "content": str, "images": [base64_str]}
        'chat_model_cot': [],  # Model's chain of thought history
        'chat_conversation': [],  # Conversation history for API
        'eikon_pending_request': None,  # Pending API request dict (set during thinking state)
        'previous_search_results': None,  # Cached previous search results from API
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def authenticate_user(email: str, password: str) -> Tuple[bool, Optional[str]]:
    """
    Authenticate user and retrieve API key.

    Args:
        email: User's registered email
        password: User's password

    Returns:
        Tuple of (success: bool, api_key: Optional[str])
    """
    if not EIKON_AVAILABLE:
        # Demo mode - accept any credentials
        if email and password:
            return True, f"demo_api_key_{email.split('@')[0]}"
        return False, None

    try:
        api_key = eikon.utils.get_api_key_from_credentials(
            email=email,
            password=str(password)
        )
        if api_key:
            return True, api_key
        return False, None
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return False, None


def get_user_credit_balance(api_key: str) -> Optional[float]:
    """
    Get the current credit balance for a user.

    Args:
        api_key: User's API key

    Returns:
        Current credit balance or None if unavailable
    """
    if not EIKON_AVAILABLE:
        # Demo mode - return mock balance
        return 1000.0

    try:
        base_api_address = EIKON_API_ENDPOINTS["check_credits"]
        payload = {"api_key": api_key}
        r = requests.post(base_api_address, json=payload, timeout=10)
        if r.ok:
            return r.json().get("current_api_credit_balance")
        return None
    except Exception:
        return None


def search_locations(
    prompt: str,
    api_key: str,
    effort: str = "test",
    spatial_resolution: str = "London - all",
    borough: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Execute a search query using the Eikon Search API.

    Uses the queue endpoint to process requests one at a time,
    preventing OOM errors on the server.

    Args:
        prompt: Natural language search query
        api_key: User's API key
        effort: Search effort level (test, quick, moderate, exhaustive)
        spatial_resolution: Area scope for search
        borough: Specific London borough (if applicable)

    Returns:
        DataFrame with search results or None
    """
    if not EIKON_AVAILABLE:
        # Return mock data for demo
        return _generate_mock_search_results(prompt)

    try:
        # Use the queue endpoint to avoid OOM errors on the server
        # The queue processes requests one at a time
        base_api_address = EIKON_API_ENDPOINTS["search_queue"]
        payload = {
            "prompt": prompt,
            "api_key": api_key,
            "effort_selection": effort,
            "spatial_resolution_for_search": spatial_resolution,
        }
        if spatial_resolution == "London - boroughs" and borough:
            payload['selected_london_borough'] = borough

        r = requests.post(base_api_address, json=payload, timeout=10000)

        if r.ok:
            response_data = r.json()
            if "successful_job_completion" in response_data:
                # Parse results from the response
                results_json = response_data["successful_job_completion"]
                results = pd.DataFrame.from_dict(json.loads(results_json))
            else:
                st.error(f"Search failed: {response_data}")
                return None
        else:
            st.error(f"Search request failed: {r.status_code}")
            return None

        # Convert H3 location IDs to latitude/longitude if not already present
        if results is not None and isinstance(results, pd.DataFrame) and not results.empty:
            # Check if latitude/longitude columns are missing or all zeros
            needs_coords = (
                'latitude' not in results.columns or
                'longitude' not in results.columns or
                (results['latitude'] == 0).all() or
                (results['longitude'] == 0).all()
            )

            if needs_coords and H3_AVAILABLE and 'location_id' in results.columns:
                # Extract coordinates from H3 cell IDs
                results['latitude'] = results['location_id'].apply(
                    lambda x: h3.h3_to_geo(x)[0] if x else 0
                )
                results['longitude'] = results['location_id'].apply(
                    lambda x: h3.h3_to_geo(x)[1] if x else 0
                )

        return results
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return None


# ============================================================================
# Search Agent Observability Functions
# ============================================================================

SEARCH_STAGES = {
    "stage_1_complete_initial_processing.txt": {
        "name": "Initial Processing",
        "description": "Processing your search query...",
        "icon": "🔍",
        "progress": 15
    },
    "stage_2_complete_initial_screening.txt": {
        "name": "Initial Screening",
        "description": "Scanning locations across the search area...",
        "icon": "📍",
        "progress": 30
    },
    "stage_3_complete_secondary_screening.txt": {
        "name": "Secondary Screening",
        "description": "Refining candidate locations...",
        "icon": "🎯",
        "progress": 50
    },
    "stage_4_complete_additional_location_context_considered.txt": {
        "name": "Context Analysis",
        "description": "Gathering additional location context...",
        "icon": "🗺️",
        "progress": 65
    },
    "stage_5_complete_evaluation_and_reflection.txt": {
        "name": "AI Evaluation",
        "description": "AI models evaluating results...",
        "icon": "🤖",
        "progress": 85
    },
    "stage_6_complete_final_results.txt": {
        "name": "Complete",
        "description": "Search complete!",
        "icon": "✅",
        "progress": 100
    }
}


def get_search_progress_dir(api_key: str) -> str:
    """Get the progress directory path for a user's search job."""
    return f"/Users/tariromashongamhende/Local Files/ml_projects/satellite_slug/project_eikon/mapping_tables/user_data_tables/users/{api_key}"


def get_model_thoughts_dir(api_key: str) -> str:
    """Get the model thoughts directory path for AI evaluation stage."""
    return f"/Users/tariromashongamhende/Local Files/ml_projects/satellite_slug/project_eikon/mapping_tables/user_data_tables/users/{api_key}/model_thoughts"


def check_search_progress(api_key: str) -> Dict[str, Any]:
    """
    Check the current progress of a search job by examining stage files.

    Returns:
        Dictionary with progress info including current stage, progress percentage,
        and any available stage content.
    """
    import os

    progress_dir = get_search_progress_dir(api_key)

    if not os.path.exists(progress_dir):
        return {
            "stage": None,
            "progress": 0,
            "description": "Initializing search...",
            "icon": "⏳",
            "content": None
        }

    # Get all stage files present
    files = os.listdir(progress_dir)
    stage_files = [f for f in files if f.startswith("stage_") and f.endswith(".txt")]

    if not stage_files:
        return {
            "stage": None,
            "progress": 5,
            "description": "Starting search process...",
            "icon": "🚀",
            "content": None
        }

    # Sort to get the latest stage
    stage_files.sort()
    latest_stage_file = stage_files[-1]

    # Get stage info
    stage_info = SEARCH_STAGES.get(latest_stage_file, {
        "name": "Processing",
        "description": "Processing...",
        "icon": "⚙️",
        "progress": 50
    })

    # Read stage content if available
    content = None
    try:
        with open(os.path.join(progress_dir, latest_stage_file), "r") as f:
            content = f.read()
    except:
        pass

    return {
        "stage": stage_info["name"],
        "progress": stage_info["progress"],
        "description": stage_info["description"],
        "icon": stage_info["icon"],
        "content": content,
        "is_complete": latest_stage_file == "stage_6_complete_final_results.txt"
    }


def get_ai_model_thoughts(api_key: str) -> Dict[str, Any]:
    """
    Get the latest AI model thoughts during the evaluation stage.

    Returns:
        Dictionary with model thoughts info including count, latest thought, etc.
    """
    import os

    model_dir = get_model_thoughts_dir(api_key)

    if not os.path.exists(model_dir):
        return {
            "available": False,
            "count": 0,
            "latest": None,
            "is_final": False
        }

    files = os.listdir(model_dir)
    thought_files = [f for f in files if f.startswith("thoughts_")]

    if not thought_files:
        return {
            "available": False,
            "count": 0,
            "latest": None,
            "is_final": False
        }

    thought_files.sort()
    latest_file = thought_files[-1]
    is_final = latest_file == "thoughts_final.txt"

    # Read latest thought
    latest_thought = None
    try:
        with open(os.path.join(model_dir, latest_file), "r") as f:
            latest_thought = f.read()
    except:
        pass

    return {
        "available": True,
        "count": len(thought_files),
        "latest": latest_thought,
        "is_final": is_final
    }


def parse_model_thought(thought_text: str) -> Dict[str, str]:
    """Parse a model thought text file into evaluation and rationale."""
    if not thought_text:
        return {"evaluation": None, "rationale": None}

    evaluation = None
    rationale = None

    for line in thought_text.split("\n"):
        if line.startswith("AI Evaluation:"):
            evaluation = line.replace("AI Evaluation:", "").strip()
        elif line.startswith("AI Rationale:"):
            rationale = line.replace("AI Rationale:", "").strip()

    return {"evaluation": evaluation, "rationale": rationale}


def get_search_final_results(api_key: str) -> Optional[str]:
    """Get the final results JSON from stage 6 file."""
    import os

    progress_dir = get_search_progress_dir(api_key)
    results_file = os.path.join(progress_dir, "stage_6_complete_final_results.txt")

    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                return f.read()
        except:
            pass
    return None


def get_stage_1_info(api_key: str) -> Optional[Dict[str, str]]:
    """Get the processed prompt info from stage 1."""
    import os

    progress_dir = get_search_progress_dir(api_key)
    stage_file = os.path.join(progress_dir, "stage_1_complete_initial_processing.txt")

    if os.path.exists(stage_file):
        try:
            with open(stage_file, "r") as f:
                content = f.read()

            original = None
            cleaned = None
            for line in content.split("\n"):
                if line.startswith("Original Prompt:"):
                    original = line.replace("Original Prompt:", "").strip()
                elif line.startswith("Cleaned Prompt:"):
                    cleaned = line.replace("Cleaned Prompt:", "").strip()

            return {"original": original, "cleaned": cleaned}
        except:
            pass
    return None


def get_relevant_location_count(api_key: str) -> Optional[int]:
    """Get the count of locations being considered from stage 2."""
    import os

    progress_dir = get_search_progress_dir(api_key)
    file_for_consideration = sorted([x for x in os.listdir(progress_dir) if x.startswith("stage_")])[-1]
    stage_file = os.path.join(progress_dir,file_for_consideration)

    if os.path.exists(stage_file):
        try:
            with open(stage_file, "r") as f:
                content = f.read()
            # Extract location count from the content
            if "Relevant Locations to Consider:" in content:
                # Parse the list to count locations
                import re
                match = re.search(r'\[([^\]]+)\]', content)
                if match:
                    locations = match.group(1).split(",")
                    return len([l for l in locations if l.strip()])
        except:
            pass
    return None


def get_location_description(
    lat: float,
    lon: float,
    resolution: str,
    api_key: str
) -> Optional[str]:
    """
    Get a description of what exists at a specific location.

    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        resolution: Detail level (low, medium, high)
        api_key: User's API key

    Returns:
        Text description of the location
    """
    if not EIKON_AVAILABLE:
        return _generate_mock_context(lat, lon, resolution)

    try:
        description = eikon.context.get_location_description(
            lat=lat,
            lon=lon,
            resolution=resolution,
            user_api_key=api_key
        )
        return description
    except Exception as e:
        st.error(f"Context error: {str(e)}")
        return None


def calculate_visual_similarity(
    location_1: List[float],
    location_2: List[float],
    resolution: str,
    api_key: str
) -> Optional[float]:
    """
    Calculate visual similarity between two locations.

    Args:
        location_1: [lat, lon] of first location
        location_2: [lat, lon] of second location
        resolution: Comparison resolution (low, medium, high)
        api_key: User's API key

    Returns:
        Similarity score between 0 and 1
    """
    if not EIKON_AVAILABLE:
        return _generate_mock_similarity(location_1, location_2, resolution)

    try:
        similarity = eikon.similarity.visual_similarity(
            location_1_lat_lon_list=location_1,
            location_2_lat_lon_list=location_2,
            resolution=resolution,
            user_api_key=api_key
        )
        return similarity
    except Exception as e:
        st.error(f"Similarity error: {str(e)}")
        return None


def calculate_descriptive_similarity(
    location_1: List[float],
    location_2: List[float],
    resolution: str,
    api_key: str
) -> Optional[float]:
    """
    Calculate descriptive similarity between two locations using VLM semantic features.

    Args:
        location_1: [lat, lon] of first location
        location_2: [lat, lon] of second location
        resolution: Comparison resolution (low, medium, high)
        api_key: User's API key

    Returns:
        Similarity score between 0 and 1
    """
    if not EIKON_AVAILABLE:
        return _generate_mock_similarity(location_1, location_2, resolution)

    try:
        similarity = eikon.similarity.descriptive_similarity(
            location_1_lat_lon_list=location_1,
            location_2_lat_lon_list=location_2,
            resolution=resolution,
            user_api_key=api_key
        )
        return similarity
    except Exception as e:
        st.error(f"Similarity error: {str(e)}")
        return None


def calculate_combined_similarity(
    location_1: List[float],
    location_2: List[float],
    resolution: str,
    api_key: str
) -> Optional[float]:
    """
    Calculate combined similarity between two locations (visual + descriptive).

    Args:
        location_1: [lat, lon] of first location
        location_2: [lat, lon] of second location
        resolution: Comparison resolution (low, medium, high)
        api_key: User's API key

    Returns:
        Similarity score between 0 and 1
    """
    if not EIKON_AVAILABLE:
        return _generate_mock_similarity(location_1, location_2, resolution)

    try:
        similarity = eikon.similarity.combined_similarity(
            location_1_lat_lon_list=location_1,
            location_2_lat_lon_list=location_2,
            resolution=resolution,
            user_api_key=api_key
        )
        return similarity
    except Exception as e:
        st.error(f"Similarity error: {str(e)}")
        return None


def run_portfolio_comparison(
    location_df: pd.DataFrame,
    api_key: str,
    resolution: str = "medium",
    similarity_type: str = "combined"
) -> Optional[pd.DataFrame]:
    """
    Run batch portfolio comparison between multiple location pairs.

    Args:
        location_df: DataFrame with columns: orig, dest, orig_latitude, orig_longitude,
                     dest_latitude, dest_longitude
        api_key: User's API key
        resolution: Comparison resolution (low, medium, high)
        similarity_type: Type of similarity (visual, descriptive, combined)

    Returns:
        DataFrame with columns: orig, dest, similarity
    """
    if not EIKON_AVAILABLE:
        return _generate_mock_portfolio_results(location_df)

    try:
        results = eikon.jobs.eikon_portfolio_comparison(
            orig_uniq_id=location_df["orig"].tolist(),
            dest_uniq_id=location_df["dest"].tolist(),
            orig_lat_list=location_df["orig_latitude"].tolist(),
            orig_lon_list=location_df["orig_longitude"].tolist(),
            dest_lat_list=location_df["dest_latitude"].tolist(),
            dest_lon_list=location_df["dest_longitude"].tolist(),
            user_api_key=api_key,
            resolution=resolution,
            similarity_type=similarity_type
        )
        return results
    except Exception as e:
        st.error(f"Portfolio comparison error: {str(e)}")
        return None


def detect_objects_at_location(
    lat: float,
    lon: float,
    resolution: str,
    api_key: str
) -> Optional[Dict[str, Any]]:
    """
    Detect objects at a specific location using YOLO model.

    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        resolution: Detection resolution (low, medium, high)
        api_key: User's API key

    Returns:
        Dictionary with detected objects or None
    """
    if not EIKON_AVAILABLE:
        return _generate_mock_object_detection(lat, lon, resolution)

    try:
        base_api_address = EIKON_API_ENDPOINTS["objects_detected"]
        payload = {
            "lat": lat,
            "lon": lon,
            "resolution": resolution,
            "api_key": api_key
        }
        r = requests.post(base_api_address, json=payload, timeout=200)
        if r.ok:
            return r.json()
        return None
    except Exception as e:
        st.error(f"Object detection error: {str(e)}")
        return None


def detect_objects_with_image(
    location_id: str,
    api_key: str
) -> Optional[Tuple[Any, str]]:
    """
    Detect objects at a location and return annotated image with bounding boxes.

    Args:
        location_id: H3 index location ID
        api_key: User's API key

    Returns:
        Tuple of (objects_json, base64_image_string) or None
    """
    if not EIKON_AVAILABLE:
        return _generate_mock_object_detection_with_image(location_id)

    try:
        base_api_address = EIKON_API_ENDPOINTS["yolo_detection"]
        payload = {
            "location_id": location_id,
            "api_key": api_key
        }
        r = requests.post(base_api_address, json=payload, timeout=200)
        if r.ok:
            response_payload = r.json()
            objects_found = response_payload.get("objects")
            img_w_objects = response_payload.get("img_w_objects_detected")
            return objects_found, img_w_objects
        return None
    except Exception as e:
        st.error(f"Object detection with image error: {str(e)}")
        return None


H3_RESOLUTION_MAP = {
    "low": 7,
    "medium": 8,
    "high": 9,
}

INVALID_IMAGE_SENTINELS = {None, "", "no_objects_found", "no_image_in_demo_mode"}


def _coords_to_h3_location_id(lat: float, lon: float, resolution: str) -> Optional[str]:
    """Convert coordinates to an H3 location ID matching the selected resolution."""
    if not H3_AVAILABLE:
        return None

    h3_resolution = H3_RESOLUTION_MAP.get(resolution, 8)
    try:
        return h3.geo_to_h3(lat, lon, h3_resolution)
    except Exception:
        return None


def _sanitize_image_data(image_value: Optional[str]) -> Optional[str]:
    """Normalize image payloads and drop sentinel values from API responses."""
    if not image_value or not isinstance(image_value, str):
        return None

    trimmed = image_value.strip()
    return trimmed if trimmed not in INVALID_IMAGE_SENTINELS else None


def send_chat_message(
    user_message: str,
    model_cot_history: List[str],
    conversation_history: List[str],
    api_key: str
) -> Optional[Dict[str, Any]]:
    """
    Send a message to the EIKON AI Chat endpoint.

    Args:
        user_message: The user's message
        model_cot_history: List of model's chain of thought entries
        conversation_history: List of previous conversation turns
        api_key: User's API key

    Returns:
        Dictionary with:
        - in_conversation_information: Model's reasoning/thoughts
        - model_response: The response text (with XML tags)
        - map_bytes: Optional list of base64-encoded images
    """
    import re

    if not EIKON_AVAILABLE:
        return _generate_mock_chat_response(user_message)

    try:
        base_url = EIKON_API_ENDPOINTS["base_url"]
        queue_submit_url = f'{base_url}/eikon_ai_chat_queue'
        queue_status_url = f'{base_url}/eikon_ai_chat_queue_status'

        # Format the conversation history
        cleaned_user_message = f"USER: {user_message}"

        payload = {
            "model_cot_history": "\n -".join(model_cot_history[-3:]),
            "conversation_history": "\n\n ".join(conversation_history[-3:] + [cleaned_user_message]),
            "api_key": api_key,
        }

        # Step 1: Submit to the queue
        submit_response = requests.post(queue_submit_url, json=payload, timeout=120)
        if not submit_response.ok:
            st.error(f"Failed to submit chat request: {submit_response.status_code}")
            return None

        job_id = submit_response.json().get("job_id")
        if not job_id:
            st.error("No job_id returned from queue endpoint")
            return None

        # Step 2: Poll for the result
        import time as _time
        max_wait = 900  # 15 minutes max
        poll_interval = 3  # seconds between polls
        elapsed = 0

        while elapsed < max_wait:
            _time.sleep(poll_interval)
            elapsed += poll_interval

            status_response = requests.get(
                queue_status_url,
                params={"job_id": job_id},
                timeout=200
            )

            if not status_response.ok and status_response.status_code != 500:
                st.error(f"Queue status check failed: {status_response.status_code}")
                return None

            status_data = status_response.json()

            if status_data["status"] == "completed":
                return status_data["result"]
            elif status_data["status"] == "failed":
                st.error(f"Chat processing failed: {status_data.get('error', 'Unknown error')}")
                return None
            # else still queued/processing — keep polling

        st.error("Chat request timed out after waiting in queue. Please try again.")
        return None
    except requests.exceptions.Timeout:
        st.error("Chat request timed out. The server may be busy, please try again.")
        return None
    except Exception as e:
        st.error(f"Chat error: {str(e)}")
        return None


def decode_and_display_image(img_b64: str, caption: str = "Image") -> bool:
    """
    Decode a base64 image string and display it in Streamlit.

    Args:
        img_b64: Base64 encoded image string
        caption: Caption for the image

    Returns:
        True if image was displayed successfully, False otherwise
    """
    if not img_b64:
        return False

    # Skip placeholder values
    if img_b64 in ["no_objects_found", "no_image_in_demo_mode", None, ""]:
        return False

    try:
        # Decode base64 string to bytes
        img_bytes = base64.b64decode(img_b64)

        # Open as PIL Image
        img = PILImage.open(BytesIO(img_bytes))

        # Display in Streamlit
        st.image(img, caption=caption, use_container_width=True)
        return True

    except Exception as e:
        st.warning(f"Could not display image: {str(e)[:100]}")
        return False


def extract_chat_response(model_response: str) -> str:
    """
    Extract the clean response text from the model's XML-tagged response.

    Args:
        model_response: Raw model response with XML tags

    Returns:
        Clean response text without any XML tags
    """
    import re

    if not model_response:
        return ""

    text = model_response

    # Try to extract from <output><response>...</response></output> format
    output_match = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
    if output_match:
        text = output_match.group(1)

    # Try to extract from <response>...</response> format
    response_match = re.search(r"<response>(.*?)</response>", text, re.DOTALL)
    if response_match:
        text = response_match.group(1)

    # Remove any remaining XML-like tags (including partial/malformed ones)
    # Remove complete tags like <tag>...</tag>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<tool>.*?</tool>", "", text, flags=re.DOTALL)
    text = re.sub(r"<map>.*?</map>", "", text, flags=re.DOTALL)

    # Remove any standalone opening or closing tags
    text = re.sub(r"</?(output|response|think|tool|map|topic|area|similarity|vision|sat_img|yolo|inputs|map_type|portfolio_comparison|area_comparison|dist_check|loc_context|area_filter)>", "", text)

    # Clean up any leftover XML-style tags (catch-all)
    text = re.sub(r"</?[a-z_]+>", "", text)

    # Clean up extra whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # Reduce multiple newlines
    text = text.strip()

    return text


# List of detectable object classes
DETECTABLE_OBJECTS = [
    "building", "tree", "water", "road", "parking_lot", "roundabout",
    "sports_stadium", "field", "agricultural_land", "swimming_pool",
    "industrial_land", "lake", "tennis_court", "cars", "golf_course",
    "railway_line", "golf_sand_bunker", "forest", "motorway", "major_road",
    "minor_road", "train_depot", "solar_panels", "quarry", "airplane",
    "outdoor_sports_court", "fuel_station", "circular_sedimentation_tank",
    "outdoor_sports_field", "cemetry", "electricity_pylon", "residential_buildings",
    "industrial_buildings", "factory_chimney", "power_substation", "parking_spaces",
    "tent", "airplane_runway", "construction_land", "solar_battery_storage",
    "water_treatment_site", "crane", "shipping_container", "ship", "boat",
    "bridge", "canoes", "tree_canopy", "airport_terminals", "airport_runway"
]


# Mock data generators for demo mode
def _generate_mock_search_results(prompt: str) -> pd.DataFrame:
    """Generate mock search results for demonstration with rich data."""
    np.random.seed(hash(prompt) % 2**32)
    n_results = np.random.randint(3, 8)

    base_locations = [
        {
            "name": "Heathrow Airport",
            "lat": 51.4700,
            "lon": -0.4543,
            "description": "A major international airport with multiple terminals, runways, and extensive parking facilities. The area shows significant aviation infrastructure including control towers, hangars, and aircraft maintenance facilities. Surrounding areas feature hotels, cargo facilities, and transport links.",
            "objects": [
                {"name": "airplane_runway", "count": 2, "coverage": "15.2%"},
                {"name": "airport_terminals", "count": 4, "coverage": "8.5%"},
                {"name": "parking_lot", "count": 12, "coverage": "12.3%"},
                {"name": "road", "count": 8, "coverage": "6.1%"}
            ]
        },
        {
            "name": "London City Airport",
            "lat": 51.5048,
            "lon": 0.0495,
            "description": "A compact city airport situated in the Docklands area. Features a single runway extending over the water, modern terminal building, and proximity to the financial district. The surrounding area shows urban development with high-rise buildings.",
            "objects": [
                {"name": "airplane_runway", "count": 1, "coverage": "18.7%"},
                {"name": "water", "count": 2, "coverage": "25.3%"},
                {"name": "building", "count": 15, "coverage": "10.2%"},
                {"name": "parking_lot", "count": 3, "coverage": "4.5%"}
            ]
        },
        {
            "name": "Hyde Park",
            "lat": 51.5073,
            "lon": -0.1657,
            "description": "One of London's largest and most famous Royal Parks. Features the Serpentine lake, extensive lawns, walking paths, and various monuments. The park shows dense tree coverage with open meadows and recreational facilities.",
            "objects": [
                {"name": "tree_canopy", "count": 1, "coverage": "45.2%"},
                {"name": "water", "count": 1, "coverage": "8.3%"},
                {"name": "field", "count": 3, "coverage": "22.1%"},
                {"name": "road", "count": 6, "coverage": "5.8%"}
            ]
        },
        {
            "name": "Regent's Park",
            "lat": 51.5313,
            "lon": -0.1570,
            "description": "A historic Royal Park featuring formal gardens, sports facilities, and London Zoo. The landscape includes manicured lawns, a boating lake, and the iconic Queen Mary's Gardens with extensive rose collections.",
            "objects": [
                {"name": "tree_canopy", "count": 1, "coverage": "35.8%"},
                {"name": "field", "count": 5, "coverage": "28.4%"},
                {"name": "building", "count": 8, "coverage": "6.2%"},
                {"name": "water", "count": 1, "coverage": "3.5%"}
            ]
        },
        {
            "name": "Olympic Park",
            "lat": 51.5430,
            "lon": -0.0134,
            "description": "The Queen Elizabeth Olympic Park, legacy of the 2012 Olympics. Features world-class sports venues including the London Stadium, Aquatics Centre, and velodrome. Surrounded by parkland, waterways, and new residential developments.",
            "objects": [
                {"name": "sports_stadium", "count": 1, "coverage": "12.4%"},
                {"name": "swimming_pool", "count": 1, "coverage": "2.1%"},
                {"name": "field", "count": 4, "coverage": "18.6%"},
                {"name": "residential_buildings", "count": 20, "coverage": "15.3%"}
            ]
        },
        {
            "name": "Battersea Power Station",
            "lat": 51.4822,
            "lon": -0.1443,
            "description": "An iconic decommissioned coal-fired power station, now redeveloped into a mixed-use complex. The distinctive four chimneys are a London landmark. Surrounding area shows modern residential towers, retail spaces, and riverside walkways.",
            "objects": [
                {"name": "industrial_buildings", "count": 1, "coverage": "8.5%"},
                {"name": "residential_buildings", "count": 12, "coverage": "22.1%"},
                {"name": "water", "count": 1, "coverage": "15.2%"},
                {"name": "parking_lot", "count": 4, "coverage": "5.8%"}
            ]
        },
        {
            "name": "The O2 Arena",
            "lat": 51.5030,
            "lon": 0.0032,
            "description": "A large entertainment complex on the Greenwich Peninsula, featuring the distinctive dome structure. Includes a major concert venue, cinema, restaurants, and exhibition spaces. Surrounded by new developments and the River Thames.",
            "objects": [
                {"name": "building", "count": 1, "coverage": "18.9%"},
                {"name": "parking_lot", "count": 6, "coverage": "12.4%"},
                {"name": "water", "count": 1, "coverage": "20.5%"},
                {"name": "road", "count": 5, "coverage": "8.2%"}
            ]
        },
        {
            "name": "Wembley Stadium",
            "lat": 51.5560,
            "lon": -0.2795,
            "description": "England's national football stadium with its iconic arch. The 90,000-seat venue hosts major sporting events and concerts. Surrounded by Wembley Arena, shopping outlets, and extensive parking facilities.",
            "objects": [
                {"name": "sports_stadium", "count": 1, "coverage": "22.3%"},
                {"name": "parking_lot", "count": 8, "coverage": "18.5%"},
                {"name": "building", "count": 10, "coverage": "12.1%"},
                {"name": "road", "count": 6, "coverage": "9.4%"}
            ]
        },
    ]

    selected = np.random.choice(len(base_locations), min(n_results, len(base_locations)), replace=False)

    results = []
    for idx in selected:
        loc = base_locations[idx]
        relevance = np.random.uniform(0.6, 0.99)

        # Generate AI rationale based on the prompt and location
        rationale = f"This location matches your search for '{prompt[:30]}...' because {loc['description'][:100].lower()}. "
        if relevance > 0.8:
            rationale += "High relevance score indicates strong alignment with search criteria."
        else:
            rationale += "Moderate relevance - some features match but not all criteria fully met."

        # Format objects detected as JSON string
        objects_json = json.dumps(loc['objects'])

        results.append({
            'location_id': f"89195d{np.random.randint(100000, 999999):x}ffff",
            'name': loc['name'],
            'latitude': loc['lat'] + np.random.uniform(-0.005, 0.005),
            'longitude': loc['lon'] + np.random.uniform(-0.005, 0.005),
            'search_results': relevance,
            'description': loc['description'],
            'objects_detected': objects_json,
            'ai_rationale': rationale,
            'ai_evaluation': 1 if relevance > 0.7 else 0
        })

    # Sort by relevance score descending
    results = sorted(results, key=lambda x: x['search_results'], reverse=True)

    return pd.DataFrame(results)


def _generate_mock_context(lat: float, lon: float, resolution: str) -> str:
    """Generate mock context description for demonstration."""
    detail_levels = {
        "low": "This area contains urban development with mixed land use.",
        "medium": "This location features a mix of residential and commercial buildings, with nearby green spaces and transportation infrastructure.",
        "high": "Detailed analysis reveals multi-story buildings, parking facilities, pedestrian pathways, and vegetation. The area shows signs of recent development with modern architectural features."
    }
    return f"[{resolution.upper()} RESOLUTION] Coordinates ({lat:.4f}, {lon:.4f}): {detail_levels.get(resolution, detail_levels['medium'])}"


def _generate_mock_similarity(loc1: List[float], loc2: List[float], resolution: str) -> float:
    """Generate mock similarity score for demonstration."""
    np.random.seed(int((loc1[0] + loc2[0]) * 1000) % 2**32)
    base_score = np.random.uniform(0.3, 0.9)

    # Higher resolution typically shows more differences
    resolution_adjustments = {"low": 0.1, "medium": 0, "high": -0.15}
    adjusted = base_score + resolution_adjustments.get(resolution, 0)

    return max(0.0, min(1.0, adjusted))


def _generate_mock_portfolio_results(location_df: pd.DataFrame) -> pd.DataFrame:
    """Generate mock portfolio comparison results for demonstration."""
    results = []
    for idx, row in location_df.iterrows():
        # Generate deterministic but varied similarity scores
        seed_val = hash(f"{row['orig']}_{row['dest']}") % 2**32
        np.random.seed(seed_val)
        similarity = np.random.uniform(0.35, 0.75)

        results.append({
            'orig': row['orig'],
            'dest': row['dest'],
            'similarity': round(similarity, 4)
        })

    return pd.DataFrame(results)


def _generate_mock_object_detection(lat: float, lon: float, resolution: str) -> Dict[str, Any]:
    """Generate mock object detection results for demonstration."""
    np.random.seed(int((lat + lon) * 10000) % 2**32)

    # Common objects based on location type
    urban_objects = ["building", "road", "cars", "parking_lot", "tree", "residential_buildings"]
    park_objects = ["tree", "field", "water", "lake", "tree_canopy", "outdoor_sports_field"]
    industrial_objects = ["industrial_buildings", "parking_lot", "road", "shipping_container", "crane"]
    airport_objects = ["airplane", "airplane_runway", "airport_terminals", "parking_lot", "road"]

    # Randomly select a location type
    location_type = np.random.choice(["urban", "park", "industrial", "airport"], p=[0.5, 0.25, 0.15, 0.1])

    if location_type == "urban":
        base_objects = urban_objects
    elif location_type == "park":
        base_objects = park_objects
    elif location_type == "industrial":
        base_objects = industrial_objects
    else:
        base_objects = airport_objects

    # Generate detection results
    num_objects = np.random.randint(3, min(8, len(base_objects) + 1))
    selected_objects = np.random.choice(base_objects, num_objects, replace=False)

    results = []
    for obj in selected_objects:
        count = np.random.randint(1, 15)
        area_pct = np.random.uniform(0.5, 25.0)
        results.append({
            "name": obj,
            "objects_detected": count,
            "proportion_of_area_that_is_label": f"{area_pct:.4f}%"
        })

    return {"objects": json.dumps(results)}


def _generate_mock_object_detection_with_image(location_id: str) -> Tuple[str, str]:
    """Generate mock object detection with image for demonstration."""
    np.random.seed(hash(location_id) % 2**32)

    # Generate mock detection results
    objects = ["building", "road", "tree", "parking_lot", "cars"]
    num_objects = np.random.randint(2, 5)
    selected = np.random.choice(objects, num_objects, replace=False)

    results = []
    for obj in selected:
        count = np.random.randint(1, 10)
        area_pct = np.random.uniform(1.0, 20.0)
        results.append({
            "name": obj,
            "objects_detected": count,
            "proportion_of_area_that_is_label": f"{area_pct:.4f}%"
        })

    # Return mock data - in demo mode, we won't have an actual image
    return json.dumps(results), "no_image_in_demo_mode"


def _generate_mock_map_image() -> str:
    """Generate a simple mock map image as base64 for demo purposes."""
    from PIL import ImageDraw

    # Create a simple placeholder image
    width, height = 400, 300
    img = PILImage.new('RGB', (width, height), color=(240, 248, 255))  # AliceBlue background

    draw = ImageDraw.Draw(img)

    # Draw a simple map-like pattern
    # Draw grid lines
    for i in range(0, width, 40):
        draw.line([(i, 0), (i, height)], fill=(200, 200, 200), width=1)
    for i in range(0, height, 40):
        draw.line([(0, i), (width, i)], fill=(200, 200, 200), width=1)

    # Draw some "location" markers
    marker_positions = [(100, 100), (200, 150), (300, 80), (150, 220)]
    for x, y in marker_positions:
        draw.ellipse([(x-8, y-8), (x+8, y+8)], fill=(255, 100, 100), outline=(200, 50, 50))

    # Add a title
    draw.rectangle([(10, 10), (200, 35)], fill=(30, 58, 95))
    draw.text((15, 12), "EIKON Demo Map", fill=(255, 255, 255))

    # Draw a legend
    draw.rectangle([(width-100, height-50), (width-10, height-10)], fill=(255, 255, 255), outline=(100, 100, 100))
    draw.ellipse([(width-90, height-40), (width-80, height-30)], fill=(255, 100, 100))
    draw.text((width-75, height-42), "Location", fill=(50, 50, 50))

    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_bytes = buffer.read()
    return base64.b64encode(img_bytes).decode('utf-8')


def _generate_mock_chat_response(user_message: str) -> Dict[str, Any]:
    """Generate mock chat response for demonstration."""
    # Predefined responses based on keywords
    message_lower = user_message.lower()

    if any(word in message_lower for word in ["hello", "hi", "hey", "greetings"]):
        response_text = "Hello! I'm EIKON, your geospatial intelligence assistant. I can help you explore locations, find places matching specific criteria, compare areas, and detect objects in satellite imagery. What would you like to know about?"
        images = None

    elif any(word in message_lower for word in ["park", "green", "nature"]):
        response_text = "I've analyzed several parks in the London area. Hyde Park is one of the largest, covering about 350 acres with diverse landscapes including the Serpentine lake. Regent's Park offers beautiful gardens and sports facilities. Richmond Park is the largest Royal Park, home to wild deer. Here's a map showing these locations:"
        # Include a demo image
        images = [_generate_mock_map_image()]

    elif any(word in message_lower for word in ["airport", "fly", "plane"]):
        response_text = "London has several major airports. Heathrow (LHR) is the largest and busiest, located to the west. I can see multiple runways, terminal buildings, and extensive parking facilities in the satellite imagery. London City Airport is closer to central London, with a distinctive single runway over the Thames. Here's a map of the airport locations:"
        images = [_generate_mock_map_image()]

    elif any(word in message_lower for word in ["stadium", "sports", "football"]):
        response_text = "I can identify several major stadiums in London through satellite analysis. Wembley Stadium is the most prominent, with its distinctive arch visible from space. The Olympic Stadium in Stratford, Emirates Stadium (Arsenal), and Tottenham Hotspur Stadium are also clearly visible. Here's a map showing their locations:"
        images = [_generate_mock_map_image()]

    elif any(word in message_lower for word in ["show", "map", "image", "satellite", "see"]):
        response_text = "Here's a satellite view of the area you requested. The markers indicate points of interest that match your query. You can see the urban layout, green spaces, and infrastructure patterns in the imagery."
        images = [_generate_mock_map_image()]

    elif any(word in message_lower for word in ["find", "search", "looking for", "where"]):
        response_text = "I can help you search for specific locations or features. To search effectively, please tell me:\n\n1. What type of place are you looking for?\n2. Any specific features it should have?\n3. A preferred area or borough in London?\n\nI'll analyze satellite imagery to find matching locations and can show you the results on a map."
        images = None

    elif any(word in message_lower for word in ["compare", "similar", "difference"]):
        response_text = "I can compare locations using visual similarity analysis. This helps identify areas with similar land use patterns, building density, or natural features. Would you like me to compare two specific locations, or would you prefer to find areas similar to a reference location you have in mind?"
        images = None

    elif any(word in message_lower for word in ["detect", "object", "identify", "what is"]):
        response_text = "My object detection capabilities allow me to identify 50 different types of features in satellite imagery, including buildings, roads, vehicles, water bodies, vegetation, industrial facilities, airports, and more. If you provide coordinates or a location name, I can analyze what objects are present and estimate their coverage area."
        images = None

    else:
        response_text = f"I understand you're asking about: '{user_message}'. As EIKON, I can help you with:\n\n- **Location Search**: Find places matching specific criteria\n- **Context Analysis**: Understand what's at a given location\n- **Similarity Comparison**: Compare how alike two places are\n- **Object Detection**: Identify features in satellite imagery\n- **Portfolio Analysis**: Batch compare multiple locations\n\nHow can I assist you today?"
        images = None

    # Format as the API would return
    formatted_response = f"<output><think>Processing user query about: {user_message[:50]}...</think><response>{response_text}</response></output>"

    result = {
        "in_conversation_information": f"Processed query: {user_message[:100]}...",
        "model_response": formatted_response,
    }

    if images:
        result["map_bytes"] = images

    return result


# London boroughs list
LONDON_BOROUGHS = [
    "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley",
    "Camden", "City of London", "Croydon", "Ealing", "Enfield",
    "Greenwich", "Hackney", "Hammersmith and Fulham", "Haringey", "Harrow",
    "Havering", "Hillingdon", "Hounslow", "Islington", "Kensington and Chelsea",
    "Kingston upon Thames", "Lambeth", "Lewisham", "Merton", "Newham",
    "Redbridge", "Richmond upon Thames", "Southwark", "Sutton", "Tower Hamlets",
    "Waltham Forest", "Wandsworth", "Westminster"
]


def render_login_page():
    """Render the login/authentication page."""
    st.markdown('<p class="main-header">EIKON</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Understand your world</p>', unsafe_allow_html=True)

    if not EIKON_AVAILABLE:
        st.warning("Running in demo mode - eikonsai package not installed. Install with: `pip install eikonsai`")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.subheader("Sign In")

        with st.form("login_form"):
            email = st.text_input("Email", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                if not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    with st.spinner("Authenticating..."):
                        success, api_key = authenticate_user(email, password)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.api_key = api_key
                            st.session_state.user_email = email
                            st.rerun()
                        else:
                            st.error("Authentication failed. Please check your credentials.")

        st.markdown("---")
        st.caption("Don't have an account? Contact support to register.")


def render_location_cards(results_df: pd.DataFrame):
    """
    Render location results as interactive cards with carousel navigation.

    Args:
        results_df: DataFrame with search results
    """
    if results_df is None or results_df.empty:
        st.info("No results to display.")
        return

    # Initialize card index in session state
    if 'card_index' not in st.session_state:
        st.session_state.card_index = 0

    num_results = len(results_df)

    # Navigation controls
    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])

    with nav_col1:
        if st.button("<- Previous", use_container_width=True, disabled=st.session_state.card_index == 0):
            st.session_state.card_index -= 1
            st.rerun()

    with nav_col2:
        st.markdown(f"<h4 style='text-align: center;'>Location {st.session_state.card_index + 1} of {num_results}</h4>", unsafe_allow_html=True)

    with nav_col3:
        if st.button("Next ->", use_container_width=True, disabled=st.session_state.card_index >= num_results - 1):
            st.session_state.card_index += 1
            st.rerun()

    # Ensure card_index is valid
    st.session_state.card_index = max(0, min(st.session_state.card_index, num_results - 1))

    # Get current location data
    current_loc = results_df.iloc[st.session_state.card_index]

    # Location Card Container
    st.markdown("""
    <style>
        .location-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 2px;
            margin: 10px 0;
        }
        .location-card-inner {
            background: white;
            border-radius: 13px;
            padding: 20px;
        }
        .location-name {
            font-size: 1.5rem;
            font-weight: bold;
            color: #1E3A5F;
            margin-bottom: 5px;
        }
        .location-id {
            font-size: 0.8rem;
            color: #888;
            font-family: monospace;
            margin-bottom: 15px;
        }
        .relevance-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }
        .relevance-high { background-color: #28a745; }
        .relevance-medium { background-color: #ffc107; color: #333; }
        .relevance-low { background-color: #dc3545; }
        .section-header {
            font-size: 1rem;
            font-weight: 600;
            color: #1E3A5F;
            margin-top: 15px;
            margin-bottom: 8px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 5px;
        }
        .object-tag {
            display: inline-block;
            background-color: #e9ecef;
            padding: 3px 10px;
            border-radius: 15px;
            margin: 3px;
            font-size: 0.85rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Build the card
    with st.container():
        # Header with name and relevance
        header_col1, header_col2 = st.columns([3, 1])

        with header_col1:
            location_name = current_loc.get('location_id', 'Unknown Location')
            location_id = current_loc.get('location_id', 'N/A')
            st.markdown(f"### {location_name}")
            st.caption(f"ID: `{location_id}`")

        with header_col2:
            relevance = current_loc.get('search_results', 0)
            if isinstance(relevance, str):
                relevance = float(relevance.replace('%', '')) / 100
            relevance_pct = f"{relevance * 100:.1f}%"

            if relevance >= 0.8:
                badge_class = "relevance-high"
                badge_label = "High Match"
            elif relevance >= 0.6:
                badge_class = "relevance-medium"
                badge_label = "Good Match"
            else:
                badge_class = "relevance-low"
                badge_label = "Partial Match"

            st.metric("Relevance", relevance_pct)

        st.markdown("---")

        # Two-column layout for details
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            # Coordinates
            lat = current_loc.get('latitude', 0)
            lon = current_loc.get('longitude', 0)
            st.markdown("**Coordinates**")
            st.code(f"Lat: {lat:.6f}\nLon: {lon:.6f}")

            if h3.h3_get_resolution(location_id)==9:
                appropriate_resolution = "high"
            if h3.h3_get_resolution(location_id)==8:
                appropriate_resolution = "medium"
            if h3.h3_get_resolution(location_id)==7:
                appropriate_resolution = "low"
            st.image(eikon.context.get_location_image(lat=lat,
                                                        lon=lon,
                                                        resolution=appropriate_resolution,
                                                        user_api_key=st.session_state.api_key))

            # AI Evaluation
            ai_eval = current_loc.get('ai_evaluation', None)
            if ai_eval is not None:
                st.markdown("**AI Evaluation**")
                if ai_eval == 1:
                    st.success("✅ Recommended")
                else:
                    st.warning("⚠️ Review Suggested")

        with detail_col2:
            # Objects Detected
            objects_data = current_loc.get('objects_detected', None)
            if objects_data:
                st.markdown("**Objects Detected**")
                try:
                    if isinstance(objects_data, str):
                        objects_list = json.loads(objects_data)
                    else:
                        objects_list = objects_data

                    for obj in objects_list[:6]:  # Limit to 6 objects
                        obj_name = obj.get('name', 'unknown').replace('_', ' ').title()
                        obj_coverage = obj.get('coverage', obj.get('proportion_of_area_that_is_label', 'N/A'))
                        st.markdown(f"• **{obj_name}**: {obj_coverage}")
                except (json.JSONDecodeError, TypeError):
                    st.text("Object data unavailable")

        # Description Section
        st.markdown("---")
        st.markdown("**Location Description**")
        description = current_loc.get('description', 'No description available.')
        st.info(description)

        # AI Rationale Section
        ai_rationale = current_loc.get('ai_rationale', None)
        if ai_rationale:
            with st.expander("🧠 AI Analysis & Rationale", expanded=False):
                st.write(ai_rationale)

        # Quick Actions
        st.markdown("---")
        action_col1, action_col2, action_col3 = st.columns(3)

        with action_col1:
            if st.button("View on Map", key=f"map_{st.session_state.card_index}", use_container_width=True):
                st.session_state.context_lat = lat
                st.session_state.context_lon = lon
                st.toast(f"Location coordinates copied! Lat: {lat:.4f}, Lon: {lon:.4f}")

        with action_col2:
            if st.button("Analyze Location", key=f"analyze_{st.session_state.card_index}", use_container_width=True):
                st.toast("Switch to Context tab to analyze this location")

        with action_col3:
            if st.button("Copy ID", key=f"copy_{st.session_state.card_index}", use_container_width=True):
                st.code(location_id)

    # Quick jump to specific result
    st.markdown("---")
    jump_col1, jump_col2 = st.columns([3, 1])
    with jump_col1:
        selected_idx = st.selectbox(
            "Jump to location:",
            options=range(num_results),
            format_func=lambda x: f"{x + 1}. {results_df.iloc[x].get('name', 'Location ' + str(x + 1))} ({results_df.iloc[x].get('search_results', 0) * 100:.0f}%)",
            index=st.session_state.card_index,
            key="location_jump_select"
        )
    with jump_col2:
        if st.button("Go", use_container_width=True):
            st.session_state.card_index = selected_idx
            st.rerun()


def render_search_tab():
    """Render the Search API tab."""
    st.header("Location Search")
    st.markdown("Search for locations across London using natural language queries.")

    # Initialize search-specific session state
    if 'search_triggered' not in st.session_state:
        st.session_state.search_triggered = False
    if 'last_search_params' not in st.session_state:
        st.session_state.last_search_params = None
    if 'search_spatial_resolution' not in st.session_state:
        st.session_state.search_spatial_resolution = "London - all"

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Search Parameters")

        # Spatial resolution selector OUTSIDE the form so it can control borough visibility
        spatial_resolution = st.selectbox(
            "Spatial Resolution",
            options=["London - all", "London - boroughs"],
            help="Search across all of London or focus on a specific borough",
            key="search_spatial_resolution"
        )

        # Borough selector - only shown when "London - boroughs" is selected
        selected_borough = None
        if spatial_resolution == "London - boroughs":
            selected_borough = st.selectbox(
                "Select Borough",
                options=LONDON_BOROUGHS,
                help="Choose a specific London borough to search within",
                key="search_selected_borough"
            )

        # Use a form for the search query and button to prevent re-triggering
        with st.form(key="search_form", clear_on_submit=False):
            search_prompt = st.text_area(
                "Search Query",
                placeholder="e.g., I'm looking for an airport",
                help="Describe what you're looking for in natural language"
            )

            effort_level = st.select_slider(
                "Search Effort",
                options=["test", "quick", "moderate", "exhaustive"],
                value="test",
                help="Higher effort = more thorough search, but takes longer"
            )

            # Form submit button - only triggers search when explicitly clicked
            search_button = st.form_submit_button("Search", type="primary", use_container_width=True)

        # Only execute search when form is submitted
        if search_button:
            if search_prompt:
                # Create a hash of search params to detect if it's a new search
                current_params = {
                    'prompt': search_prompt,
                    'effort': effort_level,
                    'resolution': spatial_resolution,
                    'borough': selected_borough
                }

                user_api_key = st.session_state.api_key

                # Show search progress UI
                st.markdown("### Search in Progress")
                st.markdown(f"**Query:** {search_prompt}")
                st.markdown(f"**Effort Level:** {effort_level}")

                # Create UI containers for live updates
                progress_bar = st.progress(0)
                status_container = st.empty()
                stage_detail_container = st.empty()
                ai_thoughts_container = st.empty()

                # Start the search API call in a background thread
                import threading
                search_result_holder = {'result': None, 'error': None, 'done': False}

                def run_search_thread():
                    try:
                        search_result_holder['result'] = search_locations(
                            prompt=search_prompt,
                            api_key=user_api_key,
                            effort=effort_level,
                            spatial_resolution=spatial_resolution,
                            borough=selected_borough
                        )
                    except Exception as e:
                        search_result_holder['error'] = str(e)
                    finally:
                        search_result_holder['done'] = True

                search_thread = threading.Thread(target=run_search_thread)
                search_thread.start()

                # Poll for progress while search is running
                last_stage = None
                last_thought_count = 0

                while not search_result_holder['done']:
                    # Check progress via the API endpoint
                    try:
                        progress_response = requests.post(
                            EIKON_API_ENDPOINTS["check_job_complete"],
                            json={"api_key": user_api_key},
                            timeout=5
                        )
                        if progress_response.ok:
                            progress_data = progress_response.json()

                            if progress_data.get('job_complete'):
                                progress_bar.progress(100)
                                status_container.success("**Search Complete!**")
                            else:
                                latest_ckpt = progress_data.get('latest_ckpt', '')

                                # Update UI based on checkpoint
                                if 'stage_1' in latest_ckpt:
                                    progress_bar.progress(15)
                                    status_container.info("**Stage 1/6:** Done processing your search query...")
                                elif 'stage_2' in latest_ckpt:
                                    progress_bar.progress(30)
                                    status_container.info("**Stage 2/6:** Done initial screening of locations...")
                                    # Show location count
                                    loc_count = get_relevant_location_count(user_api_key)
                                    if loc_count:
                                        stage_detail_container.caption(f"Scanning {loc_count} candidate locations...")
                                elif 'stage_3' in latest_ckpt:
                                    progress_bar.progress(50)
                                    status_container.info("**Stage 3/6:** Done secondary screening...")
                                elif 'stage_4' in latest_ckpt:
                                    progress_bar.progress(65)
                                    status_container.info("**Stage 4/6:** Done gathering location context...")
                                elif 'stage_5' in latest_ckpt:
                                    progress_bar.progress(85)
                                    status_container.info("**Stage 5/6:** AI evaluation completed...")
                                    # Show AI model thoughts
                                    thoughts = get_ai_model_thoughts(user_api_key)
                                    if thoughts['available'] and thoughts['count'] > last_thought_count:
                                        last_thought_count = thoughts['count']
                                        parsed = parse_model_thought(thoughts['latest'])
                                        if parsed['rationale']:
                                            eval_icon = "✅" if parsed['evaluation'] == "1" else "❌"
                                            ai_thoughts_container.caption(f"Evaluated {thoughts['count']} locations... Latest: {eval_icon}")
                                elif 'stage_6' in latest_ckpt:
                                    progress_bar.progress(100)
                                    status_container.success("**Stage 6/6:** Finalizing results...")
                                    time.sleep(2)

                    except requests.exceptions.RequestException:
                        pass  # Continue polling even if one request fails

                    time.sleep(10)  # Poll every 2 seconds

                # Wait for thread to complete
                search_thread.join()

                # Clear progress UI
                progress_bar.empty()
                status_container.empty()
                stage_detail_container.empty()
                ai_thoughts_container.empty()

                # Get results
                if search_result_holder['error']:
                    st.error(f"Search error: {search_result_holder['error']}")
                    results = None
                else:
                    results = search_result_holder['result']

                if results is not None and isinstance(results, pd.DataFrame) and not results.empty:
                    st.session_state.search_results = results
                    st.session_state.last_search_params = current_params
                    st.session_state.search_history.append({
                        'query': search_prompt,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'results_count': len(results)
                    })

                    # Show search process summary
                    with st.expander("📊 Search Process Summary", expanded=True):
                        # Stage 1: Show processed prompt
                        stage_1_info = get_stage_1_info(user_api_key)
                        if stage_1_info:
                            st.markdown("**Stage 1 - Query Processing:**")
                            if stage_1_info.get('cleaned'):
                                st.markdown(f"- Original: *{stage_1_info.get('original', search_prompt)}*")
                                st.markdown(f"- Optimized: *{stage_1_info.get('cleaned')}*")
                            st.markdown("---")

                        # Stage 2: Show location count
                        loc_count = get_relevant_location_count(user_api_key)
                        if loc_count:
                            st.markdown(f"**Stage 2 - Initial Screening:** {loc_count} candidate locations identified")
                            st.markdown("---")

                        # Show final progress
                        final_progress = check_search_progress(user_api_key)
                        st.markdown(f"**Final Status:** {final_progress['icon']} {final_progress['stage']}")

                    # Show summary of AI evaluations
                    if 'ai_model_evaluation' in results.columns:
                        recommended = (results['ai_model_evaluation'] >= 0.5).sum()
                        not_recommended = len(results) - recommended
                        st.success(f"✅ Found {len(results)} locations! ({recommended} AI-recommended, {not_recommended} not recommended)")

                        # Show AI evaluation breakdown
                        with st.expander("🤖 AI Model Evaluation Details", expanded=False):
                            st.markdown("The AI evaluated each location against your search criteria:")
                            st.markdown(f"- **Recommended:** {recommended} locations (shown in 🟢 green on map)")
                            st.markdown(f"- **Not Recommended:** {not_recommended} locations (shown in 🔴 red on map)")

                            if 'ai_model_rationale' in results.columns:
                                st.markdown("---")
                                st.markdown("**Sample AI Reasoning:**")
                                # Show first recommended and first not-recommended rationale
                                recommended_rows = results[results['ai_model_evaluation'] >= 0.5]
                                not_recommended_rows = results[results['ai_model_evaluation'] < 0.5]

                                if len(recommended_rows) > 0:
                                    sample_rationale = recommended_rows.iloc[0].get('ai_model_rationale', '')
                                    if sample_rationale:
                                        st.markdown("*✅ Recommended location:*")
                                        st.caption(sample_rationale[:400] + "..." if len(str(sample_rationale)) > 400 else sample_rationale)

                                if len(not_recommended_rows) > 0:
                                    sample_rationale = not_recommended_rows.iloc[0].get('ai_model_rationale', '')
                                    if sample_rationale:
                                        st.markdown("*❌ Not recommended location:*")
                                        st.caption(sample_rationale[:400] + "..." if len(str(sample_rationale)) > 400 else sample_rationale)
                    else:
                        st.success(f"✅ Found {len(results)} locations!")

                    st.rerun()
                else:
                    st.warning("No results found for your query.")
            else:
                st.warning("Please enter a search query.")

    with col2:
        st.subheader("Results")

        if st.session_state.search_results is not None:
            results_df = st.session_state.search_results

            # Create tabs for different views
            view_tab1, view_tab2, view_tab3 = st.tabs(["Map View", "Location Profile", "Data Table"])

            with view_tab1:
                # Map visualization
                if 'latitude' in results_df.columns and 'longitude' in results_df.columns:
                    # Add formatted score for tooltip
                    map_df = results_df.copy()
                    map_df['score_pct'] = (map_df['search_results'] * 100).round(1).astype(str) + '%'

                    # Color grading based on AI model evaluation
                    # Green (recommended) = [40, 167, 69, 200], Red (not recommended) = [220, 53, 69, 200]
                    if 'ai_model_evaluation' in map_df.columns:
                        def get_color_from_evaluation(eval_score):
                            if eval_score >= 0.5:
                                return [40, 167, 69, 200]  # Green - AI recommended
                            else:
                                return [220, 53, 69, 200]  # Red - AI not recommended

                        map_df['color'] = map_df['ai_model_evaluation'].apply(get_color_from_evaluation)
                        map_df['ai_status'] = map_df['ai_model_evaluation'].apply(
                            lambda x: 'Recommended' if x >= 0.5 else 'Not Recommended'
                        )
                    else:
                        # Fallback to static color if ai_model_evaluation not available
                        map_df['color'] = [[30, 58, 95, 200]] * len(map_df)
                        map_df['ai_status'] = 'N/A'

                    view_state = pdk.ViewState(
                        latitude=results_df['latitude'].mean(),
                        longitude=results_df['longitude'].mean(),
                        zoom=10,
                        pitch=0
                    )

                    layer = pdk.Layer(
                        'ScatterplotLayer',
                        data=map_df,
                        get_position='[longitude, latitude]',
                        get_color='color',
                        get_radius=200,
                        pickable=True,
                        auto_highlight=True
                    )

                    deck = pdk.Deck(
                        layers=[layer],
                        initial_view_state=view_state,
                        tooltip={"text": "{location_id}\nAI Evaluation: {ai_status}\nRelevance: {score_pct}"}
                    )

                    st.pydeck_chart(deck)

            with view_tab2:
                # Location Cards Carousel
                render_location_cards(results_df)

            with view_tab3:
                # Results table
                display_df = results_df.copy()
                # Format relevance score as percentage
                if 'search_results' in display_df.columns:
                    display_df['relevance_score'] = display_df['search_results'].apply(lambda x: f"{x:.1%}")

                # Select columns to display (hide long text columns)
                display_cols = [col for col in display_df.columns if col not in ['objects_detected', 'ai_rationale', 'description']]
                st.dataframe(
                    display_df[display_cols],
                    use_container_width=True,
                    hide_index=True
                )

                # Download option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Full Results (CSV)",
                    data=csv,
                    file_name="eikon_search_results.csv",
                    mime="text/csv"
                )
        else:
            st.info("Enter a search query and click 'Search' to find locations.")


def render_context_tab():
    """Render the Context Module tab."""
    st.header("Location Context")
    st.markdown("Get detailed descriptions of any location based on satellite imagery analysis.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Location Input")

        input_method = st.radio(
            "Input Method",
            options=["Coordinates", "Click on Map"],
            horizontal=True
        )

        if input_method == "Coordinates":
            lat = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=51.5074,
                format="%.6f",
                help="Enter latitude (e.g., 51.5074 for London)"
            )
            lon = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-0.1278,
                format="%.6f",
                help="Enter longitude (e.g., -0.1278 for London)"
            )
        else:
            st.info("Click on the map to select a location (coming soon)")
            lat, lon = 51.5074, -0.1278

        resolution = st.select_slider(
            "Analysis Resolution",
            options=["low", "medium", "high"],
            value="medium",
            help="Higher resolution provides more detailed descriptions"
        )

        analyze_button = st.button("Analyze Location", type="primary", use_container_width=True)

        if analyze_button:
            with st.spinner("Analyzing location..."):
                description = get_location_description(
                    lat=lat,
                    lon=lon,
                    resolution=resolution,
                    api_key=st.session_state.api_key
                )

                if description:
                    st.session_state.context_results = {
                        'lat': lat,
                        'lon': lon,
                        'resolution': resolution,
                        'description': description
                    }
                    st.success("Analysis complete!")

    with col2:
        st.subheader("Analysis Results")

        if st.session_state.context_results:
            results = st.session_state.context_results

            # Show location on map
            location_df = pd.DataFrame([{
                'lat': results['lat'],
                'lon': results['lon']
            }])

            view_state = pdk.ViewState(
                latitude=results['lat'],
                longitude=results['lon'],
                zoom=14,
                pitch=0
            )

            layer = pdk.Layer(
                'ScatterplotLayer',
                data=location_df,
                get_position='[lon, lat]',
                get_color='[255, 0, 0, 200]',
                get_radius=100,
            )

            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

            # Show description
            st.markdown("### Location Description")
            st.markdown(f"**Coordinates:** {results['lat']:.6f}, {results['lon']:.6f}")
            st.markdown(f"**Resolution:** {results['resolution'].upper()}")
            st.markdown("---")
            st.write(results['description'])
        else:
            st.info("Enter coordinates and click 'Analyze Location' to get a description.")


def render_similarity_tab():
    """Render the Similarity Module tab."""
    st.header("Location Similarity")
    st.markdown("Compare the similarity between two locations.")

    similarity_type = st.selectbox(
        "Similarity Type",
        options=["Visual", "Descriptive", "Combined"],
        key="sim_type",
        help="Visual: satellite imagery features | Descriptive: semantic/textual features | Combined: weighted blend of both"
    )

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Location 1")
        lat1 = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=51.5313,
            format="%.6f",
            key="sim_lat1",
            help="e.g., 51.5313 (Regent's Park)"
        )
        lon1 = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-0.1570,
            format="%.6f",
            key="sim_lon1",
            help="e.g., -0.1570 (Regent's Park)"
        )
        st.caption("Example: Regent's Park")

    with col2:
        st.subheader("Location 2")
        lat2 = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=51.4337,
            format="%.6f",
            key="sim_lat2",
            help="e.g., 51.4337 (Wimbledon)"
        )
        lon2 = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-0.2144,
            format="%.6f",
            key="sim_lon2",
            help="e.g., -0.2144 (Wimbledon)"
        )
        st.caption("Example: Wimbledon Tennis Club")

    with col3:
        st.subheader("Settings")
        resolution = st.select_slider(
            "Comparison Resolution",
            options=["low", "medium", "high"],
            value="medium",
            key="sim_resolution",
            help="Higher resolution = more detailed comparison"
        )

        st.markdown("")
        st.markdown("")
        compare_button = st.button("Compare Locations", type="primary", use_container_width=True)

    if compare_button:
        similarity_type_lower = similarity_type.lower()
        with st.spinner(f"Calculating {similarity_type_lower} similarity..."):
            if similarity_type_lower == "visual":
                similarity = calculate_visual_similarity(
                    location_1=[lat1, lon1],
                    location_2=[lat2, lon2],
                    resolution=resolution,
                    api_key=st.session_state.api_key
                )
            elif similarity_type_lower == "descriptive":
                similarity = calculate_descriptive_similarity(
                    location_1=[lat1, lon1],
                    location_2=[lat2, lon2],
                    resolution=resolution,
                    api_key=st.session_state.api_key
                )
            else:
                similarity = calculate_combined_similarity(
                    location_1=[lat1, lon1],
                    location_2=[lat2, lon2],
                    resolution=resolution,
                    api_key=st.session_state.api_key
                )

            if similarity is not None:
                st.session_state.similarity_results = {
                    'location_1': [lat1, lon1],
                    'location_2': [lat2, lon2],
                    'resolution': resolution,
                    'similarity': similarity,
                    'similarity_type': similarity_type
                }

    # Display results
    if st.session_state.similarity_results:
        st.markdown("---")
        results = st.session_state.similarity_results

        col_map, col_score = st.columns([2, 1])

        with col_map:
            # Show both locations on map
            locations_df = pd.DataFrame([
                {'lat': results['location_1'][0], 'lon': results['location_1'][1], 'name': 'Location 1', 'color': [255, 0, 0]},
                {'lat': results['location_2'][0], 'lon': results['location_2'][1], 'name': 'Location 2', 'color': [0, 0, 255]}
            ])

            center_lat = (results['location_1'][0] + results['location_2'][0]) / 2
            center_lon = (results['location_1'][1] + results['location_2'][1]) / 2

            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=11,
                pitch=0
            )

            layer = pdk.Layer(
                'ScatterplotLayer',
                data=locations_df,
                get_position='[lon, lat]',
                get_color='color',
                get_radius=300,
                pickable=True
            )

            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "{name}"}
            ))

        with col_score:
            st.markdown("### Similarity Score")

            score = results['similarity']
            result_type = results.get('similarity_type', 'Visual')

            # Color coding based on score
            if score >= 0.7:
                color = "green"
                label = "High Similarity"
            elif score >= 0.4:
                color = "orange"
                label = "Moderate Similarity"
            else:
                color = "red"
                label = "Low Similarity"

            st.metric(
                label=f"{result_type} Similarity",
                value=f"{score:.2%}",
                delta=label
            )

            st.progress(score)

            st.markdown(f"**Resolution:** {results['resolution'].upper()}")
            type_descriptions = {
                "Visual": "Higher scores indicate more visually similar locations based on satellite imagery analysis.",
                "Descriptive": "Higher scores indicate more semantically similar locations based on scene description analysis.",
                "Combined": "Higher scores indicate overall similarity combining both visual and descriptive features."
            }
            st.caption(type_descriptions.get(result_type, type_descriptions["Visual"]))


def render_portfolio_tab():
    """Render the Portfolio Comparison tab."""
    st.header("Portfolio Comparison")
    st.markdown("Compare multiple location pairs in batch. Upload a CSV or manually enter location pairs.")

    # Settings in sidebar-style column
    col_input, col_results = st.columns([1, 2])

    with col_input:
        st.subheader("Configuration")

        resolution = st.select_slider(
            "Resolution",
            options=["low", "medium", "high"],
            value="medium",
            key="portfolio_resolution",
            help="Low (~2.5km) uses fewer credits, High (~500m) provides more detail"
        )

        similarity_type = st.selectbox(
            "Similarity Type",
            options=["combined", "visual", "descriptive"],
            index=0,
            help="Combined uses both visual and descriptive features"
        )

        st.markdown("---")
        st.subheader("Input Data")

        input_method = st.radio(
            "Input Method",
            options=["Upload CSV", "Manual Entry", "Use Example Data"],
            horizontal=False
        )

        location_df = None

        if input_method == "Upload CSV":
            st.markdown("""
            **Required CSV columns:**
            - `orig` - Origin location ID
            - `dest` - Destination location ID
            - `orig_latitude` - Origin latitude
            - `orig_longitude` - Origin longitude
            - `dest_latitude` - Destination latitude
            - `dest_longitude` - Destination longitude
            """)

            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="Upload a CSV with location pairs to compare"
            )

            if uploaded_file is not None:
                try:
                    location_df = pd.read_csv(uploaded_file)
                    required_cols = ['orig', 'dest', 'orig_latitude', 'orig_longitude',
                                     'dest_latitude', 'dest_longitude']
                    missing_cols = [col for col in required_cols if col not in location_df.columns]

                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                        location_df = None
                    else:
                        st.success(f"Loaded {len(location_df)} location pairs")
                        st.session_state.portfolio_input_df = location_df
                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")

        elif input_method == "Manual Entry":
            st.markdown("Enter location pairs (one per row):")

            num_pairs = st.number_input(
                "Number of pairs",
                min_value=1,
                max_value=20,
                value=3,
                key="num_portfolio_pairs"
            )

            manual_data = []
            for i in range(int(num_pairs)):
                with st.expander(f"Pair {i + 1}", expanded=(i == 0)):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Origin**")
                        orig_id = st.text_input("ID", value=f"orig_{i+1}", key=f"orig_id_{i}")
                        orig_lat = st.number_input("Latitude", value=51.5560, format="%.6f", key=f"orig_lat_{i}")
                        orig_lon = st.number_input("Longitude", value=-0.2795, format="%.6f", key=f"orig_lon_{i}")
                    with col_b:
                        st.markdown("**Destination**")
                        dest_id = st.text_input("ID", value=f"dest_{i+1}", key=f"dest_id_{i}")
                        dest_lat = st.number_input("Latitude", value=51.5074, format="%.6f", key=f"dest_lat_{i}")
                        dest_lon = st.number_input("Longitude", value=-0.1278, format="%.6f", key=f"dest_lon_{i}")

                    manual_data.append({
                        'orig': orig_id,
                        'dest': dest_id,
                        'orig_latitude': orig_lat,
                        'orig_longitude': orig_lon,
                        'dest_latitude': dest_lat,
                        'dest_longitude': dest_lon
                    })

            location_df = pd.DataFrame(manual_data)
            st.session_state.portfolio_input_df = location_df

        else:  # Use Example Data
            st.markdown("Using example data: Wembley Stadium compared to various London locations")

            # Example: Wembley Stadium vs several London landmarks
            example_data = [
                {'orig': 'wembley', 'dest': 'olympic_park', 'orig_latitude': 51.5560, 'orig_longitude': -0.2795,
                 'dest_latitude': 51.5430, 'dest_longitude': -0.0134},
                {'orig': 'wembley', 'dest': 'o2_arena', 'orig_latitude': 51.5560, 'orig_longitude': -0.2795,
                 'dest_latitude': 51.5030, 'dest_longitude': 0.0032},
                {'orig': 'wembley', 'dest': 'hyde_park', 'orig_latitude': 51.5560, 'orig_longitude': -0.2795,
                 'dest_latitude': 51.5073, 'dest_longitude': -0.1657},
                {'orig': 'wembley', 'dest': 'regents_park', 'orig_latitude': 51.5560, 'orig_longitude': -0.2795,
                 'dest_latitude': 51.5313, 'dest_longitude': -0.1570},
                {'orig': 'wembley', 'dest': 'heathrow', 'orig_latitude': 51.5560, 'orig_longitude': -0.2795,
                 'dest_latitude': 51.4700, 'dest_longitude': -0.4543},
            ]
            location_df = pd.DataFrame(example_data)
            st.session_state.portfolio_input_df = location_df
            st.info(f"Loaded {len(location_df)} example location pairs")

        # Run comparison button
        st.markdown("---")
        run_button = st.button(
            "Run Portfolio Comparison",
            type="primary",
            use_container_width=True,
            disabled=(location_df is None or len(location_df) == 0)
        )

        if run_button and location_df is not None:
            with st.spinner(f"Comparing {len(location_df)} location pairs..."):
                progress_bar = st.progress(0)

                # Simulate progress for demo mode
                for i in range(10):
                    time.sleep(0.1)
                    progress_bar.progress((i + 1) * 10)

                results = run_portfolio_comparison(
                    location_df=location_df,
                    api_key=st.session_state.api_key,
                    resolution=resolution,
                    similarity_type=similarity_type
                )

                progress_bar.empty()

                if results is not None and not results.empty:
                    # Merge with input data for full context
                    full_results = location_df.merge(results, on=['orig', 'dest'])
                    st.session_state.portfolio_results = full_results
                    st.success(f"Comparison complete! Processed {len(results)} pairs.")
                else:
                    st.warning("No results returned from comparison.")

    with col_results:
        st.subheader("Results")

        if st.session_state.portfolio_results is not None:
            results_df = st.session_state.portfolio_results

            # Summary statistics
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Total Pairs", len(results_df))
            with col_stat2:
                st.metric("Avg Similarity", f"{results_df['similarity'].mean():.2%}")
            with col_stat3:
                st.metric("Max Similarity", f"{results_df['similarity'].max():.2%}")
            with col_stat4:
                st.metric("Min Similarity", f"{results_df['similarity'].min():.2%}")

            st.markdown("---")

            # Visualization tabs
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Map", "Chart", "Table"])

            with viz_tab1:
                # Map showing all locations with lines between pairs
                map_data = []

                for _, row in results_df.iterrows():
                    # Origin point
                    map_data.append({
                        'lat': row['orig_latitude'],
                        'lon': row['orig_longitude'],
                        'name': str(row['orig']),
                        'type': 'Origin',
                        'color': [255, 100, 100, 200]
                    })
                    # Destination point
                    map_data.append({
                        'lat': row['dest_latitude'],
                        'lon': row['dest_longitude'],
                        'name': str(row['dest']),
                        'type': 'Destination',
                        'color': [100, 100, 255, 200]
                    })

                map_df = pd.DataFrame(map_data).drop_duplicates(subset=['lat', 'lon'])

                # Calculate center
                center_lat = map_df['lat'].mean()
                center_lon = map_df['lon'].mean()

                view_state = pdk.ViewState(
                    latitude=center_lat,
                    longitude=center_lon,
                    zoom=10,
                    pitch=0
                )

                # Points layer - with tooltip for points
                points_layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=map_df,
                    get_position='[lon, lat]',
                    get_color='color',
                    get_radius=400,
                    pickable=True
                )

                # Lines layer showing connections
                line_data = []
                for _, row in results_df.iterrows():
                    # Color based on similarity (green = high, red = low)
                    sim = row['similarity']
                    r = int(255 * (1 - sim))
                    g = int(255 * sim)
                    # Pre-format the similarity as a percentage string for tooltip
                    sim_pct = f"{sim * 100:.1f}%"
                    line_data.append({
                        'start': [row['orig_longitude'], row['orig_latitude']],
                        'end': [row['dest_longitude'], row['dest_latitude']],
                        'name': f"{row['orig']} → {row['dest']}",
                        'type': f"Similarity: {sim_pct}",
                        'color': [r, g, 100, 150]
                    })

                lines_df = pd.DataFrame(line_data)

                lines_layer = pdk.Layer(
                    'LineLayer',
                    data=lines_df,
                    get_source_position='start',
                    get_target_position='end',
                    get_color='color',
                    get_width=3,
                    pickable=True
                )

                # Use simple text tooltip with column names that exist in both dataframes
                st.pydeck_chart(pdk.Deck(
                    layers=[lines_layer, points_layer],
                    initial_view_state=view_state,
                    tooltip={"text": "{name}\n{type}"}
                ))

                st.caption("Red points = Origins, Blue points = Destinations. Line color indicates similarity (green = high, red = low)")

            with viz_tab2:
                # Bar chart of similarity scores
                import plotly.express as px

                # Create a label for each pair
                chart_df = results_df.copy()
                chart_df['pair'] = chart_df['orig'] + ' → ' + chart_df['dest']
                chart_df = chart_df.sort_values('similarity', ascending=True)

                fig = px.bar(
                    chart_df,
                    x='similarity',
                    y='pair',
                    orientation='h',
                    title='Similarity Scores by Location Pair',
                    labels={'similarity': 'Similarity Score', 'pair': 'Location Pair'},
                    color='similarity',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=max(400, len(chart_df) * 40))
                st.plotly_chart(fig, use_container_width=True)

            with viz_tab3:
                # Full results table
                display_df = results_df.copy()
                display_df['similarity'] = display_df['similarity'].apply(lambda x: f"{x:.2%}")

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Download options
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="portfolio_comparison_results.csv",
                        mime="text/csv"
                    )
                with col_dl2:
                    json_data = results_df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name="portfolio_comparison_results.json",
                        mime="application/json"
                    )

        elif st.session_state.portfolio_input_df is not None:
            # Show preview of input data
            st.markdown("### Input Data Preview")
            st.dataframe(
                st.session_state.portfolio_input_df,
                use_container_width=True,
                hide_index=True
            )
            st.info("Click 'Run Portfolio Comparison' to process these location pairs.")

        else:
            st.info("Upload a CSV file, enter data manually, or use example data to get started.")

            # Show template download
            st.markdown("### CSV Template")
            template_df = pd.DataFrame({
                'orig': ['location_a', 'location_a', 'location_b'],
                'dest': ['location_b', 'location_c', 'location_c'],
                'orig_latitude': [51.5560, 51.5560, 51.5073],
                'orig_longitude': [-0.2795, -0.2795, -0.1657],
                'dest_latitude': [51.5073, 51.5030, 51.5030],
                'dest_longitude': [-0.1657, 0.0032, 0.0032]
            })
            st.dataframe(template_df, use_container_width=True, hide_index=True)

            csv_template = template_df.to_csv(index=False)
            st.download_button(
                label="Download CSV Template",
                data=csv_template,
                file_name="portfolio_comparison_template.csv",
                mime="text/csv"
            )


def render_object_detection_tab():
    """Render the Object Detection tab."""
    st.header("Object Detection")
    st.markdown("Detect and identify objects in satellite imagery using YOLO-based computer vision models.")

    col_input, col_results = st.columns([1, 2])

    with col_input:
        st.subheader("Location Input")

        # Input method selection
        input_method = st.radio(
            "Input Method",
            options=["Coordinates", "H3 Location ID"],
            horizontal=True,
            key="od_input_method"
        )

        if input_method == "Coordinates":
            lat = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=51.433700,  # Wimbledone Tennis Club
                format="%.6f",
                key="od_lat",
                help="Enter latitude (e.g., 51.5560 for Wembley Stadium)"
            )
            lon = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-0.214400,
                format="%.6f",
                key="od_lon",
                help="Enter longitude (e.g., -0.2795 for Wembley Stadium)"
            )
            location_id = None
        else:
            location_id = st.text_input(
                "H3 Location ID",
                value="",
                placeholder="e.g., 87195da49ffffff",
                help="Enter an H3 index location ID"
            )
            lat, lon = None, None

        st.markdown("---")
        st.subheader("Detection Settings")

        resolution = st.select_slider(
            "Resolution",
            options=["low", "medium", "high"],
            value="medium",
            key="od_resolution",
            help="Low (~2.5km coverage), Medium (~1km), High (~500m) - Higher resolution detects smaller objects"
        )

        # Option to include annotated image
        include_image = st.checkbox(
            "Include annotated image with bounding boxes",
            value=True,
            help="Returns the satellite image with detected objects highlighted"
        )

        st.markdown("---")

        # Run detection button
        detect_button = st.button(
            "Detect Objects",
            type="primary",
            use_container_width=True
        )

        if detect_button:
            if input_method == "Coordinates" and lat is not None and lon is not None:
                with st.spinner("Running object detection..."):
                    if include_image:
                        if H3_AVAILABLE:
                            location_id_from_coords = _coords_to_h3_location_id(lat, lon, resolution)
                        else:
                            location_id_from_coords = None

                        if location_id_from_coords:
                            result = detect_objects_with_image(
                                location_id=location_id_from_coords,
                                api_key=st.session_state.api_key
                            )
                            if result:
                                objects_json, img_b64 = result
                                st.session_state.object_detection_results = {
                                    'lat': lat,
                                    'lon': lon,
                                    'resolution': resolution,
                                    'objects': objects_json,
                                    'image': _sanitize_image_data(img_b64),
                                    'input_method': 'coordinates'
                                }
                                st.success("Detection complete!")
                        else:
                            st.warning("H3 library unavailable. Annotated images require an H3 conversion of the coordinates. Running detection without image instead.")
                            results = detect_objects_at_location(
                                lat=lat,
                                lon=lon,
                                resolution=resolution,
                                api_key=st.session_state.api_key
                            )
                            if results:
                                st.session_state.object_detection_results = {
                                    'lat': lat,
                                    'lon': lon,
                                    'resolution': resolution,
                                    'objects': results.get('objects'),
                                    'image': None,
                                    'input_method': 'coordinates'
                                }
                                st.success("Detection complete!")
                    else:
                        results = detect_objects_at_location(
                            lat=lat,
                            lon=lon,
                            resolution=resolution,
                            api_key=st.session_state.api_key
                        )
                        if results:
                            st.session_state.object_detection_results = {
                                'lat': lat,
                                'lon': lon,
                                'resolution': resolution,
                                'objects': results.get('objects'),
                                'image': None,
                                'input_method': 'coordinates'
                            }
                            st.success("Detection complete!")

            elif input_method == "H3 Location ID" and location_id:
                with st.spinner("Running object detection..."):
                    if include_image:
                        result = detect_objects_with_image(
                            location_id=location_id,
                            api_key=st.session_state.api_key
                        )
                        if result:
                            objects_json, img_b64 = result
                            st.session_state.object_detection_results = {
                                'location_id': location_id,
                                'resolution': resolution,
                                'objects': objects_json,
                                'image': _sanitize_image_data(img_b64),
                                'input_method': 'location_id'
                            }
                            st.success("Detection complete!")
                    else:
                        result = detect_objects_with_image(
                            location_id=location_id,
                            api_key=st.session_state.api_key
                        )
                        if result:
                            objects_json, _ = result
                            st.session_state.object_detection_results = {
                                'location_id': location_id,
                                'resolution': resolution,
                                'objects': objects_json,
                                'image': None,
                                'input_method': 'location_id'
                            }
                            st.success("Detection complete!")
            else:
                st.warning("Please enter valid coordinates or a location ID.")

        # Show detectable objects reference
        with st.expander("View Detectable Object Classes"):
            st.markdown("The model can detect the following **50 object classes**:")

            # Display in columns
            obj_col1, obj_col2 = st.columns(2)
            half = len(DETECTABLE_OBJECTS) // 2

            with obj_col1:
                for obj in DETECTABLE_OBJECTS[:half]:
                    st.markdown(f"- {obj.replace('_', ' ').title()}")

            with obj_col2:
                for obj in DETECTABLE_OBJECTS[half:]:
                    st.markdown(f"- {obj.replace('_', ' ').title()}")

    with col_results:
        st.subheader("Detection Results")

        if st.session_state.object_detection_results is not None:
            results = st.session_state.object_detection_results

            # Parse objects JSON
            objects_data = results.get('objects')
            if objects_data and objects_data != "no_objects_found":
                try:
                    if isinstance(objects_data, str):
                        objects_list = json.loads(objects_data)
                    else:
                        objects_list = objects_data

                    # Handle different JSON formats
                    if isinstance(objects_list, dict):
                        # Format: {"object_name": count, ...}
                        objects_df = pd.DataFrame([
                            {"Object": k.replace('_', ' ').title(), "Count": v}
                            for k, v in objects_list.items()
                        ])
                    elif isinstance(objects_list, list):
                        # Format: [{"name": "...", "objects_detected": n, "proportion_of_area_that_is_label": "..."}]
                        objects_df = pd.DataFrame(objects_list)
                        if 'name' in objects_df.columns:
                            objects_df['Object'] = objects_df['name'].apply(lambda x: x.replace('_', ' ').title())
                            if 'objects_detected' in objects_df.columns:
                                objects_df['Count'] = objects_df['objects_detected']
                            if 'proportion_of_area_that_is_label' in objects_df.columns:
                                objects_df['Area Coverage'] = objects_df['proportion_of_area_that_is_label']
                    else:
                        objects_df = pd.DataFrame()

                except (json.JSONDecodeError, TypeError) as e:
                    st.error(f"Error parsing detection results: {e}")
                    objects_df = pd.DataFrame()

                if not objects_df.empty:
                    def _format_object_label(label: Any) -> str:
                        if isinstance(label, str):
                            return label.replace('_', ' ').strip().title()
                        if label is None or (isinstance(label, float) and np.isnan(label)):
                            return ""
                        return str(label)

                    def _to_number(value: Any) -> Optional[float]:
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            return float(value)
                        if isinstance(value, str):
                            try:
                                return float(value.replace('%', ''))
                            except ValueError:
                                return None
                        return None

                    def _normalize_object_records(df: pd.DataFrame) -> pd.DataFrame:
                        records: List[Dict[str, Any]] = []

                        for _, row in df.iterrows():
                            counts = row.get('Count')
                            coverage = row.get('Area Coverage')

                            def add_record(label: Any, count_val: Any):
                                formatted_label = _format_object_label(label)
                                numeric_count = _to_number(count_val)
                                if not formatted_label or numeric_count is None:
                                    return

                                record: Dict[str, Any] = {
                                    'Object': formatted_label,
                                    'Count': numeric_count,
                                }

                                coverage_value: Optional[Any] = None
                                if isinstance(coverage, dict):
                                    coverage_value = coverage.get(label)
                                    if coverage_value is None and isinstance(label, str):
                                        normalized_key = label.replace(' ', '_').lower()
                                        coverage_value = coverage.get(normalized_key)
                                elif coverage is not None:
                                    coverage_value = coverage

                                if coverage_value is not None:
                                    record['Area Coverage'] = coverage_value

                                records.append(record)

                            if isinstance(counts, dict):
                                for obj_label, obj_count in counts.items():
                                    add_record(obj_label, obj_count)
                            else:
                                add_record(row.get('Object'), counts)

                        return pd.DataFrame(records)

                    def _aggregate_records(df: pd.DataFrame) -> pd.DataFrame:
                        if df.empty:
                            return pd.DataFrame(columns=['Object', 'Count'])

                        agg_map: Dict[str, Any] = {'Count': 'sum'}
                        if 'Area Coverage' in df.columns:
                            agg_map['Area Coverage'] = lambda vals: next(
                                (val for val in vals if val not in [None, '', np.nan]),
                                None
                            )

                        aggregated = df.groupby('Object', as_index=False).agg(agg_map)
                        return aggregated.sort_values('Count', ascending=False)

                    normalized_records_df = _normalize_object_records(objects_df)
                    summary_df = _aggregate_records(normalized_records_df)

                    # Summary metrics
                    col_m1, col_m2, col_m3 = st.columns(3)
                    objects_df
                    with col_m1:
                        total_objects = summary_df['Count'].sum() if not summary_df.empty else 0
                        st.metric("Total Objects", int(total_objects))
                    with col_m2:
                        unique_classes = summary_df['Object'].nunique() if not summary_df.empty else 0
                        st.metric("Object Classes", unique_classes)
                    with col_m3:
                        st.metric("Resolution", results.get('resolution', 'N/A').upper())

                    st.markdown("---")

                    # Visualization tabs
                    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Chart", "Image", "Table"])

                    with viz_tab1:
                        # Bar chart of detected objects
                        import plotly.express as px

                        if not summary_df.empty and {'Count', 'Object'}.issubset(summary_df.columns):
                            chart_df = summary_df.sort_values('Count', ascending=True)
                            fig = px.bar(
                                chart_df,
                                x='Count',
                                y='Object',
                                orientation='h',
                                title='Detected Objects by Count',
                                color='Count',
                                color_continuous_scale='Viridis'
                            )
                            fig.update_layout(height=max(400, len(chart_df) * 35))
                            st.plotly_chart(fig, use_container_width=True)

                        # If area coverage is available, show pie chart
                        if 'Area Coverage' in summary_df.columns:
                            st.markdown("### Area Coverage Distribution")

                            # Parse percentage strings to floats
                            def parse_pct(val):
                                if isinstance(val, str):
                                    return float(val.replace('%', ''))
                                return float(val)

                            pie_df = summary_df.dropna(subset=['Area Coverage']).copy()
                            if pie_df.empty:
                                st.info("Area coverage values unavailable for the detected objects.")
                            else:
                                pie_df['Area_Float'] = pie_df['Area Coverage'].apply(parse_pct)

                                fig_pie = px.pie(
                                    pie_df,
                                    values='Area_Float',
                                    names='Object',
                                    title='Proportion of Image Area by Object Type'
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)

                    with viz_tab2:
                        # Display annotated image if available
                        img_data = results.get('image')
                        if img_data:
                            st.markdown("### Annotated Satellite Image")
                            try:
                                # Decode base64 image
                                img_bytes = base64.b64decode(img_data)
                                img = PILImage.open(BytesIO(img_bytes)).resize((512, 512))
                                with st.container():
                                    st.image(img,
                                              caption="Detected objects with bounding boxes",
                                            #   horizontal_alignment="center",
                                            #   use_container_width=True
                                              )
                            except Exception as e:
                                st.warning(f"Could not display image: {e}")
                        else:
                            st.info("No annotated image available. Use 'H3 Location ID' input method with 'Include annotated image' option to get bounding box visualization.")

                            # Show location on map if coordinates available
                            if results.get('lat') and results.get('lon'):
                                st.markdown("### Location")
                                location_df = pd.DataFrame([{
                                    'lat': results['lat'],
                                    'lon': results['lon']
                                }])

                                view_state = pdk.ViewState(
                                    latitude=results['lat'],
                                    longitude=results['lon'],
                                    zoom=14,
                                    pitch=0
                                )

                                layer = pdk.Layer(
                                    'ScatterplotLayer',
                                    data=location_df,
                                    get_position='[lon, lat]',
                                    get_color='[255, 100, 100, 200]',
                                    get_radius=200,
                                )

                                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

                    with viz_tab3:
                        # Display results table
                        display_cols = [col for col in ['Object', 'Count', 'Area Coverage'] if col in summary_df.columns]

                        st.dataframe(
                            summary_df[display_cols] if display_cols else summary_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        # Download options
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            csv = summary_df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name="object_detection_results.csv",
                                mime="text/csv"
                            )
                        with col_dl2:
                            json_data = summary_df.to_json(orient='records', indent=2)
                            st.download_button(
                                label="Download JSON",
                                data=json_data,
                                file_name="object_detection_results.json",
                                mime="application/json"
                            )
                else:
                    st.warning("No objects detected at this location.")
            else:
                st.warning("No objects detected at this location.")
        else:
            st.info("Enter coordinates or a location ID and click 'Detect Objects' to analyze satellite imagery.")

            # Show example locations
            st.markdown("### Example Locations")
            example_locations = pd.DataFrame([
                {"Name": "Wembley Stadium", "Latitude": 51.5560, "Longitude": -0.2795, "Expected Objects": "Stadium, Parking, Roads"},
                {"Name": "Heathrow Airport", "Latitude": 51.4700, "Longitude": -0.4543, "Expected Objects": "Runways, Terminals, Aircraft"},
                {"Name": "Hyde Park", "Latitude": 51.5073, "Longitude": -0.1657, "Expected Objects": "Trees, Water, Fields"},
                {"Name": "Canary Wharf", "Latitude": 51.5054, "Longitude": -0.0235, "Expected Objects": "Buildings, Roads, Water"},
            ])
            st.dataframe(example_locations, use_container_width=True, hide_index=True)


def get_eikon_avatar():
    """Load the EIKON logo for use as chat avatar."""
    import os
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eikon_logo_tes_v4.png")
    if os.path.exists(logo_path):
        return logo_path
    return "🛰️"  # Fallback to emoji if logo not found


def get_eikon_animated_avatar():
    """Load the animated EIKON logo GIF for use as chat avatar during thinking."""
    import os
    gif_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eikon_logo_animated.gif")
    if os.path.exists(gif_path):
        return gif_path
    return get_eikon_avatar()  # Fallback to static logo


def render_ai_chat_tab():
    """Render the AI Chat tab with conversation interface."""
    st.header("EIKON AI Chat")
    st.markdown("Have a conversation with EIKON about satellite imagery, locations, and geospatial analysis.")

    # Load EIKON avatars — static for history, animated for thinking state
    eikon_avatar = get_eikon_avatar()
    eikon_animated_avatar = get_eikon_animated_avatar()

    # Chat container styling
    st.markdown("""
    <style>
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            background-color: #e3f2fd;
        }
        .chat-message.assistant {
            background-color: #f5f5f5;
        }
        .chat-message .message-content {
            margin-top: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Main chat layout — chat on left, info sidebar on right
    col_chat, col_info = st.columns([3, 1])

    with col_info:
        st.subheader("Chat Info")

        # Clear chat button
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.chat_model_cot = []
            st.session_state.chat_conversation = []
            st.session_state.eikon_pending_request = None
            st.rerun()

        st.markdown("---")

        # Chat statistics
        st.markdown("**Conversation Stats**")
        st.metric("Messages", len(st.session_state.chat_messages))
        user_msgs = len([m for m in st.session_state.chat_messages if m['role'] == 'user'])
        st.metric("Your Messages", user_msgs)

        st.markdown("---")

        # Example prompts
        st.markdown("**Try asking:**")
        example_prompts = [
            "Find me a park with a lake",
            "Where are all the airports?",
            "Find me some large parks",
            "Fine me some industrial areas and run an object detection to see what's in them",
            "Find me the bridges in this city and plot them on a map"
        ]

        for prompt in example_prompts:
            if st.button(prompt, key=f"example_{hash(prompt)}", use_container_width=True):
                # Add this prompt to session state to be processed
                st.session_state.pending_message = prompt
                st.rerun()

        st.markdown("---")

        # Show model's reasoning (expandable)
        if st.session_state.chat_model_cot:
            with st.expander("View EIKON's Reasoning"):
                for i, thought in enumerate(st.session_state.chat_model_cot[-5:]):
                    st.markdown(f"**Step {i+1}:** {thought[:200]}...")

    with col_chat:
        # Display existing messages
        for message in st.session_state.chat_messages:
            role = message['role']
            content = message['content']
            images = message.get('images', [])

            if role == 'user':
                with st.chat_message("user"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant", avatar=eikon_avatar):
                    st.markdown(content)

                    # Display any images that came with the response
                    if images and len(images) > 0:
                        # Filter out invalid images
                        valid_images = [
                            img for img in images
                            if img and img not in ["no_objects_found", "no_image_in_demo_mode", None, ""]
                        ]

                        if valid_images:
                            st.markdown("**Attached Images:**")
                            # Create columns for images
                            num_images = len(valid_images)
                            if num_images == 1:
                                img_cols = [st.container()]
                            else:
                                img_cols = st.columns(min(num_images, 3))

                            for idx, img_b64 in enumerate(valid_images):
                                try:
                                    # Decode base64 to bytes
                                    img_bytes = base64.b64decode(img_b64)

                                    # Open as PIL Image
                                    img = PILImage.open(BytesIO(img_bytes))

                                    col_idx = idx % len(img_cols)
                                    with img_cols[col_idx]:
                                        st.image(img, caption=f"Image {idx + 1}", use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not display image {idx + 1}: {str(e)[:50]}")

        # Chat input — pinned inside the left column, below messages
        pending = st.session_state.get('pending_message', None)
        if pending:
            user_input = pending
            del st.session_state.pending_message
        else:
            user_input = st.chat_input("Type your message to EIKON...")

        # Process new user input
        if user_input:
            # Add user message to history immediately so it renders in the
            # message loop on next rerun
            st.session_state.chat_messages.append({
                'role': 'user',
                'content': user_input,
                'images': []
            })

            # Format for API
            cleaned_user_message = f"USER: {user_input}"
            st.session_state.chat_conversation.append(cleaned_user_message)

            # Flag that we're waiting for a response — triggers a rerun that
            # shows the user message from history + thinking indicator below it
            st.session_state.eikon_pending_request = {
                'user_message': user_input,
                'model_cot_history': list(st.session_state.chat_model_cot),
                'conversation_history': list(st.session_state.chat_conversation),
                'api_key': st.session_state.api_key
            }
            st.rerun()

        # If there's a pending request, show thinking indicator and make the API call.
        # At this point the user message is already in chat_messages and rendered
        # above by the message loop, so the flow is: messages → thinking → chat input
        if st.session_state.get('eikon_pending_request'):
            req = st.session_state.eikon_pending_request

            with st.chat_message("assistant", avatar=eikon_animated_avatar):
                st.markdown("*EIKON is thinking...*")

            response = send_chat_message(
                user_message=req['user_message'],
                model_cot_history=req['model_cot_history'],
                conversation_history=req['conversation_history'],
                api_key=req['api_key']
            )

            # Clear the pending flag
            del st.session_state.eikon_pending_request

            if response:
                # Extract the clean response text
                model_response_raw = response.get('model_response', '')
                clean_response = extract_chat_response(model_response_raw)

                # Get any images - handle both list and single image cases
                raw_images = response.get('map_bytes', None)

                # Normalize images to a list
                if raw_images is None:
                    images = []
                elif isinstance(raw_images, str):
                    images = [raw_images] if raw_images else []
                elif isinstance(raw_images, list):
                    images = raw_images
                else:
                    images = []

                # Filter out invalid images
                valid_images = [
                    img for img in images
                    if img and isinstance(img, str) and img not in ["no_objects_found", "no_image_in_demo_mode", ""]
                ]

                # Update model CoT history
                cot_info = response.get('in_conversation_information', '')
                if cot_info:
                    cot_items = cot_info.split("\n *")
                    for item in cot_items:
                        if item.strip() and item not in st.session_state.chat_model_cot:
                            st.session_state.chat_model_cot.append(item.strip())

                # Add assistant message to history
                st.session_state.chat_messages.append({
                    'role': 'assistant',
                    'content': clean_response,
                    'images': valid_images
                })

                # Update conversation history for API
                cleaned_eikon_response = f"EIKON: {clean_response}"
                st.session_state.chat_conversation.append(cleaned_eikon_response)

            else:
                error_msg = "I'm sorry, I couldn't process your request. Please try again."
                st.session_state.chat_messages.append({
                    'role': 'assistant',
                    'content': error_msg,
                    'images': []
                })

            # Rerun to show the response from history
            st.rerun()


def render_memory_tab():
    """Render the Memory tab — view and manage EIKON's memories about you."""
    st.header("EIKON Memory")
    st.markdown("View what EIKON remembers about you across conversations. You can delete any memory snippet.")

    if not st.session_state.get("authenticated") or not st.session_state.get("api_key"):
        st.warning("Please sign in to view your memories.")
        return

    api_key = st.session_state.api_key
    base_url = EIKON_API_ENDPOINTS["base_url"]

    # Fetch memory data from backend
    try:
        mem_response = requests.get(f"{base_url}/memory/{api_key}", timeout=10)
        if not mem_response.ok:
            st.error(f"Could not load memories (status {mem_response.status_code})")
            return
        memory_data = mem_response.json()
    except requests.exceptions.ConnectionError:
        st.warning("Cannot connect to the EIKON backend. Memory features require the API server to be running.")
        return
    except Exception as e:
        st.error(f"Error loading memories: {str(e)[:100]}")
        return

    # --- Reflection / User Profile ---
    reflection = memory_data.get("reflection")
    if reflection:
        st.subheader("User Profile")
        st.markdown(f"*Last updated: {reflection['created_at'][:10]}* (v{reflection['version']})")
        st.markdown(reflection["content"])
        st.markdown("---")

    # --- Manual Reflect Button ---
    col_reflect, col_stats = st.columns([1, 2])
    with col_reflect:
        if st.button("Generate Reflection", use_container_width=True):
            with st.spinner("Generating reflection from your memory snippets..."):
                try:
                    reflect_resp = requests.post(f"{base_url}/memory/{api_key}/reflect", timeout=120)
                    if reflect_resp.ok:
                        result = reflect_resp.json()
                        if result.get("reflection"):
                            st.success("Reflection generated!")
                            st.rerun()
                        else:
                            st.info("No snippets available to generate a reflection from.")
                    else:
                        st.error(f"Reflection failed (status {reflect_resp.status_code})")
                except Exception as e:
                    st.error(f"Error: {str(e)[:100]}")
    with col_stats:
        snippets = memory_data.get("snippets", [])
        since_reflection = memory_data.get("snippet_count_since_reflection", 0)
        st.metric("Total Active Snippets", len(snippets))
        st.metric("Snippets Since Last Reflection", since_reflection)

    st.markdown("---")

    # --- Snippets List ---
    snippets = memory_data.get("snippets", [])
    if snippets:
        st.subheader(f"Memory Snippets ({len(snippets)})")
        for i, snippet in enumerate(snippets):
            col_content, col_meta, col_delete = st.columns([5, 2, 1])
            with col_content:
                type_emoji = {"fact": "F", "preference": "P", "correction": "C", "project_context": "PC"}.get(snippet["type"], "?")
                st.markdown(f"**[{type_emoji}]** {snippet['content']}")
            with col_meta:
                st.caption(f"{snippet['created_at'][:10]} | {snippet['type']}")
            with col_delete:
                if st.button("Delete", key=f"del_snippet_{snippet['id']}"):
                    try:
                        del_resp = requests.delete(
                            f"{base_url}/memory/{api_key}/snippet/{snippet['id']}",
                            timeout=10,
                        )
                        if del_resp.ok:
                            st.rerun()
                        else:
                            st.error("Delete failed")
                    except Exception as e:
                        st.error(f"Error: {str(e)[:50]}")
    else:
        st.info("No memories yet. EIKON will start remembering things about you as you chat.")


def fetch_previous_searches(api_key: str, num_results: int = 10) -> Optional[list]:
    """
    Fetch previous search results from the API.

    Args:
        api_key: User's EIKON API key
        num_results: Number of past searches to retrieve

    Returns:
        List of parsed DataFrames, or None on failure
    """
    if not EIKON_AVAILABLE:
        return None
    try:
        raw_results = eikon.utils.get_previous_search_api_results(
            api_key=api_key,
            num_requested_results=num_results
        )
        if not raw_results:
            return None
        parsed = []
        for result_json in raw_results:
            df = pd.DataFrame.from_dict(json.loads(result_json))
            parsed.append(df)
        return parsed
    except Exception:
        return None


def render_history_tab():
    """Render the Search History tab with both session and API-backed history."""
    st.header("Search History")

    # --- Section 1: In-session history ---
    st.subheader("Current Session")
    if st.session_state.search_history:
        history_df = pd.DataFrame(st.session_state.search_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        if st.button("Clear Session History"):
            st.session_state.search_history = []
            st.rerun()
    else:
        st.info("No searches this session yet.")

    st.markdown("---")

    # --- Section 2: Previous searches from API ---
    st.subheader("Previous Searches")

    if not EIKON_AVAILABLE:
        st.info("Previous search history is available when connected to the EIKON API.")
        return

    col_refresh, _ = st.columns([1, 3])
    with col_refresh:
        if st.button("Load Previous Searches"):
            with st.spinner("Fetching previous searches..."):
                results = fetch_previous_searches(st.session_state.api_key, num_results=10)
                st.session_state.previous_search_results = results

    previous = st.session_state.previous_search_results
    if previous:
        st.caption(f"Showing {len(previous)} previous search(es)")
        for i, df in enumerate(previous):
            with st.expander(f"Search {i + 1} — {len(df)} result(s)", expanded=(i == 0)):
                # Show a compact summary rather than the full dataframe
                display_cols = [c for c in df.columns if c not in ['objects_detected', 'ai_rationale']]
                st.dataframe(
                    df[display_cols] if display_cols else df,
                    use_container_width=True,
                    hide_index=True
                )
    elif previous is not None:
        # previous was fetched but returned empty
        st.info("No previous searches found for this account.")
    else:
        st.caption("Click **Load Previous Searches** to retrieve your last 10 searches.")


def render_docs_tab():
    """Render the SDK documentation tab."""

    # Read the markdown documentation file
    docs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EIKON_SDK_DOCUMENTATION.md")
    try:
        with open(docs_path, "r") as f:
            docs_content = f.read()
        st.markdown(docs_content)
    except FileNotFoundError:
        st.warning("Documentation file not found.")
        st.info("Expected location: `EIKON_SDK_DOCUMENTATION.md` in the application directory.")


def render_main_app():
    """Render the main application after authentication."""
    # Sidebar
    with st.sidebar:
        st.markdown(f"**Signed in as:** {st.session_state.user_email}")

        if st.button("Sign Out"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("---")
        st.markdown("### About EIKON")
        st.markdown("""
        EIKON is a geospatial AI system that enables:
        - AI-powered chat interface
        - Natural language location search
        - Location context analysis
        - Visual similarity comparison
        - Batch portfolio comparison
        - Object detection in satellite imagery

        [Documentation](https://github.com/Kennedy821/eikon)
        """)

        if not EIKON_AVAILABLE:
            st.warning("Demo Mode Active")

    # Credits quota box (top right)
    credit_balance = get_user_credit_balance(st.session_state.api_key)
    if credit_balance is not None:
        st.markdown(f'''
        <div class="credits-box">
            <div class="credits-label">Credit Balance</div>
            <div class="credits-value"><span class="credits-currency">£</span>{credit_balance:,.2f}</div>
        </div>
        ''', unsafe_allow_html=True)

    # Main content
    st.markdown('<p class="main-header">EIKON</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Understand your world</p>', unsafe_allow_html=True)

    # Tabs
    tab_chat, tab_search, tab_context, tab_similarity, tab_portfolio, tab_objects, tab_history, tab_memory, tab_docs = st.tabs([
        "AI Chat",
        "Search",
        "Context",
        "Similarity",
        "Portfolio",
        "Object Detection",
        "History",
        "Memory",
        "Docs"
    ])

    with tab_chat:
        render_ai_chat_tab()

    with tab_search:
        render_search_tab()

    with tab_context:
        render_context_tab()

    with tab_similarity:
        render_similarity_tab()

    with tab_portfolio:
        render_portfolio_tab()

    with tab_objects:
        render_object_detection_tab()

    with tab_history:
        render_history_tab()

    with tab_memory:
        render_memory_tab()

    with tab_docs:
        render_docs_tab()


def main():
    """Main application entry point."""
    init_session_state()

    if st.session_state.authenticated:
        render_main_app()
    else:
        render_login_page()


if __name__ == "__main__":
    main()
