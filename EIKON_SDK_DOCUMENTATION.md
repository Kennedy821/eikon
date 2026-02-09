# Eikon Python SDK Documentation

**Version 0.9.1** | Python 3.9+ | `pip install eikonsai`

Eikon is a Python SDK for location intelligence and geospatial analysis. It provides programmatic access to the EIKON Location APIs, enabling natural language search over satellite imagery, location context descriptions, visual similarity comparisons, and batch portfolio analysis -- all focused on London and the UK.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [Core Modules](#core-modules)
   - [Context](#context-module)
   - [Similarity](#similarity-module)
   - [Search (Jobs)](#search-module)
   - [Portfolio Comparison (Jobs)](#portfolio-comparison-module)
   - [AI Chat (Jobs)](#ai-chat-module)
   - [Utilities](#utilities-module)
4. [Use Cases](#use-cases)
   - [Site Selection](#use-case-1-site-selection)
   - [Portfolio Benchmarking](#use-case-2-portfolio-benchmarking)
   - [Change Detection & Monitoring](#use-case-3-change-detection--monitoring)
   - [Location Profiling](#use-case-4-location-profiling)
   - [Conversational Geospatial Analysis](#use-case-5-conversational-geospatial-analysis)
5. [API Reference](#api-reference)
6. [Concepts](#concepts)

---

## Getting Started

### Installation

```bash
pip install --upgrade eikonsai
```

### Requirements

- Python 3.9+
- macOS, Linux, or Windows
- An EIKON API key (register at the [EIKON data store](https://slugai.pagekite.me))

### Quick Start

```python
import eikonsai as eikon

# Authenticate
api_key = eikon.utils.get_api_key_from_credentials(
    email="your_email@example.com",
    password="your_password"
)

# Describe a location
description = eikon.context.get_location_description(
    lat=51.5074,
    lon=-0.1278,
    resolution="medium",
    user_api_key=api_key
)
print(description)
```

---

## Authentication

All API calls require an API key. Obtain one by registering for an EIKON account. UK postgraduate students can access educational pricing with complimentary monthly credits.

```python
import eikonsai as eikon

api_key = eikon.utils.get_api_key_from_credentials(
    email="your_email@example.com",
    password="your_password"
)
```

The returned `api_key` string is passed to every subsequent SDK call via the `user_api_key` parameter.

---

## Core Modules

### Context Module

**What does this location contain?**

The Context module analyses satellite imagery at a given coordinate and returns a natural language description of what is visible at that location.

#### `eikon.context.get_location_description`

Returns a text description of the geographic features and land use at the specified coordinates.

```python
eikon.context.get_location_description(
    lat: float,
    lon: float,
    resolution: str,
    user_api_key: str
) -> str
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `lat` | `float` | Latitude of the location |
| `lon` | `float` | Longitude of the location |
| `resolution` | `str` | Analysis detail level: `"low"`, `"medium"`, or `"high"` |
| `user_api_key` | `str` | Your EIKON API key |

**Returns:** `str` -- A natural language description of the location.

**Resolution levels:**

| Level | Coverage | Best for |
|-------|----------|----------|
| `"low"` | ~2.5 km area | Broad neighbourhood character |
| `"medium"` | ~1 km area | District-level features |
| `"high"` | ~500 m area | Specific buildings and landmarks |

**Example:**

```python
# Describe Wimbledon Tennis Club
description = eikon.context.get_location_description(
    lat=51.433727,
    lon=-0.214443,
    resolution="high",
    user_api_key=api_key
)
print(description)
# At high resolution, the model identifies specific features such as
# tennis courts, surrounding parkland, and built structures.
```

#### `eikon.context.get_location_image`

Returns satellite imagery for the specified coordinates.

```python
eikon.context.get_location_image(
    lat: float,
    lon: float,
    resolution: str,
    user_api_key: str
) -> bytes
```

**Parameters:** Same as `get_location_description`.

**Returns:** Satellite image data for the location.

---

### Similarity Module

**How similar are two locations?**

The Similarity module compares two geographic locations using satellite imagery embeddings and returns a similarity score.

#### `eikon.similarity.visual_similarity`

Computes a visual similarity score between two locations based on their satellite imagery.

```python
eikon.similarity.visual_similarity(
    location_1_lat_lon_list: list,
    location_2_lat_lon_list: list,
    resolution: str,
    user_api_key: str
) -> float
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `location_1_lat_lon_list` | `list` | `[latitude, longitude]` of the first location |
| `location_2_lat_lon_list` | `list` | `[latitude, longitude]` of the second location |
| `resolution` | `str` | `"low"`, `"medium"`, or `"high"` |
| `user_api_key` | `str` | Your EIKON API key |

**Returns:** `float` -- Similarity score between 0.0 (completely different) and 1.0 (identical).

**Example:**

```python
# Compare Regent's Park and Wimbledon Tennis Club
score = eikon.similarity.visual_similarity(
    location_1_lat_lon_list=[51.531143, -0.159893],  # Regent's Park
    location_2_lat_lon_list=[51.433727, -0.214443],  # Wimbledon
    resolution="low",
    user_api_key=api_key
)
print(f"Similarity: {score}")  # e.g., 0.838
```

Higher resolution gives more granular comparisons. The same location pair yields different scores at different resolutions:

| Resolution | Example Score (Regent's Park vs Wimbledon) |
|------------|---------------------------------------------|
| `"low"` | 0.838 |
| `"medium"` | 0.657 |
| `"high"` | 0.433 |

At low resolution, both locations appear as green open space. At high resolution, the model distinguishes between manicured tennis courts and natural parkland.

---

### Search Module

**Find locations matching a natural language description.**

The Search module uses AI to find locations across London (or a specific borough) that match a natural language query. Searches are processed asynchronously -- the SDK handles job submission and polling automatically.

#### `eikon.jobs.search_api`

Searches for locations matching a text prompt.

```python
eikon.jobs.search_api(
    my_search_prompt: str,
    user_api_key: str,
    effort_selection: str,
    spatial_resolution_for_search: str,
    selected_london_borough: str = None
) -> pd.DataFrame
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `my_search_prompt` | `str` | Natural language description of the location you are looking for |
| `user_api_key` | `str` | Your EIKON API key |
| `effort_selection` | `str` | Search thoroughness: `"test"`, `"quick"`, `"moderate"`, or `"exhaustive"` |
| `spatial_resolution_for_search` | `str` | `"London - all"` or `"London - boroughs"` |
| `selected_london_borough` | `str` | Required when using `"London - boroughs"`. One of the 33 London boroughs |

**Returns:** `pd.DataFrame` with columns:

| Column | Description |
|--------|-------------|
| `location_id` | H3 cell identifier for the location |
| `description` | AI-generated description of the location |
| `search_results` | Detailed match information |
| `wkt_geom` | Well-Known Text geometry of the location |
| `ai_model_evaluation` | AI confidence score for the match |
| `ai_model_rationale` | Explanation of why this location matches the query |
| `datetime` | Timestamp of the search |

**Effort levels:**

| Level | Approximate Results | Use When |
|-------|---------------------|----------|
| `"test"` | ~3 | Quick validation, testing your query |
| `"quick"` | ~10 | Rapid exploration |
| `"moderate"` | ~20 | Balanced search |
| `"exhaustive"` | ~50 | Comprehensive coverage |

**Example -- search across all of London:**

```python
results = eikon.jobs.search_api(
    my_search_prompt="I'm looking for an airport",
    user_api_key=api_key,
    effort_selection="test",
    spatial_resolution_for_search="London - all"
)
print(results.head())
```

**Example -- search within a specific borough:**

```python
results = eikon.jobs.search_api(
    my_search_prompt="I'm looking for an airport",
    user_api_key=api_key,
    effort_selection="test",
    spatial_resolution_for_search="London - boroughs",
    selected_london_borough="Hillingdon"
)
```

**Available London boroughs:** Barking and Dagenham, Barnet, Bexley, Brent, Bromley, Camden, City of London, Croydon, Ealing, Enfield, Greenwich, Hackney, Hammersmith and Fulham, Haringey, Harrow, Havering, Hillingdon, Hounslow, Islington, Kensington and Chelsea, Kingston upon Thames, Lambeth, Lewisham, Merton, Newham, Redbridge, Richmond upon Thames, Southwark, Sutton, Tower Hamlets, Waltham Forest, Wandsworth, Westminster.

#### Search Pipeline

Searches pass through a multi-stage pipeline:

1. **Initial Processing** -- Query parsing and optimisation
2. **Initial Screening** -- Scanning candidate locations across the search area
3. **Secondary Screening** -- Narrowing candidates based on satellite analysis
4. **Context Analysis** -- Gathering additional location context
5. **AI Evaluation** -- AI models score each candidate and provide rationale
6. **Results Compilation** -- Final ranking and delivery

---

### Portfolio Comparison Module

**Compare many location pairs at once.**

The Portfolio Comparison module performs batch similarity analysis across multiple origin-destination location pairs. This is useful for benchmarking a portfolio of sites against each other or against target locations.

#### `eikon.jobs.eikon_portfolio_comparison`

Compares lists of origin and destination locations pairwise.

```python
eikon.jobs.eikon_portfolio_comparison(
    orig_uniq_id: list,
    dest_uniq_id: list,
    orig_lat_list: list,
    orig_lon_list: list,
    dest_lat_list: list,
    dest_lon_list: list,
    user_api_key: str,
    resolution: str,
    similarity_type: str = "combined"
) -> pd.DataFrame
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `orig_uniq_id` | `list` | Unique identifiers for origin locations |
| `dest_uniq_id` | `list` | Unique identifiers for destination locations |
| `orig_lat_list` | `list` | Latitudes of origin locations |
| `orig_lon_list` | `list` | Longitudes of origin locations |
| `dest_lat_list` | `list` | Latitudes of destination locations |
| `dest_lon_list` | `list` | Longitudes of destination locations |
| `user_api_key` | `str` | Your EIKON API key |
| `resolution` | `str` | `"low"`, `"medium"`, or `"high"` |
| `similarity_type` | `str` | `"visual"`, `"descriptive"`, or `"combined"` (default: `"combined"`) |

All list parameters must be the same length.

**Returns:** `pd.DataFrame` with columns:

| Column | Description |
|--------|-------------|
| `orig` | Origin location identifier |
| `dest` | Destination location identifier |
| `similarity_score` | Float between 0.0 and 1.0 |

**Similarity types:**

| Type | Description |
|------|-------------|
| `"visual"` | Comparison based on satellite imagery embeddings |
| `"descriptive"` | Comparison based on AI-generated location descriptions |
| `"combined"` | Weighted combination of visual and descriptive similarity |

**Example:**

```python
comparison = eikon.jobs.eikon_portfolio_comparison(
    orig_uniq_id=["site_A", "site_B", "site_C"],
    dest_uniq_id=["target_1", "target_2", "target_3"],
    orig_lat_list=[51.5074, 51.5155, 51.5236],
    orig_lon_list=[-0.1278, -0.1410, -0.1580],
    dest_lat_list=[51.5090, 51.5180, 51.5250],
    dest_lon_list=[-0.1300, -0.1450, -0.1600],
    user_api_key=api_key,
    resolution="medium",
    similarity_type="combined"
)
print(comparison)
#       orig      dest  similarity_score
# 0   site_A  target_1              0.85
# 1   site_B  target_2              0.78
# 2   site_C  target_3              0.92
```

---

### AI Chat Module

**Have a conversation with an AI agent that has geospatial knowledge.**

The AI Chat module provides an interactive conversational interface to EIKON. The agent can search for locations, generate maps, compare areas, and perform object detection -- all driven by natural language.

#### `eikon.jobs.eikon_ai_chat`

Starts an interactive CLI chat session with the EIKON AI agent.

```python
eikon.jobs.eikon_ai_chat(
    user_api_key: str
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_api_key` | `str` | Your EIKON API key |

**Usage:**

```python
eikon.jobs.eikon_ai_chat(user_api_key=api_key)
# Welcome to EIKON AI chat! Type /quit to exit.
# > What parks are near Canary Wharf?
# ---
# EIKON: There are several green spaces near Canary Wharf...
# ---
# > /quit
# Good-bye!
```

The agent can:
- Search for locations matching your description
- Generate interactive maps
- Describe what is at a specific location
- Compare two locations visually
- Detect objects in satellite imagery (buildings, roads, water features, etc.)

The chat maintains conversation history and chain-of-thought context across turns within a session.

---

### Utilities Module

#### `eikon.utils.get_api_key_from_credentials`

Authenticates a user and returns their API key.

```python
eikon.utils.get_api_key_from_credentials(
    email: str,
    password: str
) -> str
```

#### `eikon.utils.get_previous_search_api_results`

Retrieves results from previous searches.

```python
eikon.utils.get_previous_search_api_results(
    api_key: str,
    num_requested_results: int
) -> list
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | `str` | Your EIKON API key |
| `num_requested_results` | `int` | Number of past search results to retrieve |

**Returns:** `list` of JSON strings, each representing a past search result. Parse with:

```python
import json
import pandas as pd

previous = eikon.utils.get_previous_search_api_results(
    api_key=api_key,
    num_requested_results=3
)

# Each element is a JSON string; convert to DataFrame
for i, result_json in enumerate(previous):
    df = pd.DataFrame.from_dict(json.loads(result_json))
    print(f"Search {i+1}: {len(df)} results")
    print(df.head())
```

---

## Use Cases

### Use Case 1: Site Selection

**Scenario:** A retail company wants to find locations in London that resemble their most successful store site.

```python
import eikonsai as eikon

api_key = eikon.utils.get_api_key_from_credentials(
    email="analyst@retailco.com",
    password="password"
)

# Step 1: Describe what makes the existing site distinctive
description = eikon.context.get_location_description(
    lat=51.5145,    # Existing successful store
    lon=-0.0825,
    resolution="high",
    user_api_key=api_key
)
print(f"Existing site profile:\n{description}")

# Step 2: Search for similar locations using natural language
candidates = eikon.jobs.search_api(
    my_search_prompt="Dense commercial high street with mixed retail and food outlets near a transport hub",
    user_api_key=api_key,
    effort_selection="moderate",
    spatial_resolution_for_search="London - all"
)
print(f"Found {len(candidates)} candidate locations")

# Step 3: Score each candidate against the reference site
import h3

for _, row in candidates.iterrows():
    loc_id = row["location_id"]
    lat, lon = h3.h3_to_geo(loc_id)

    score = eikon.similarity.visual_similarity(
        location_1_lat_lon_list=[51.5145, -0.0825],   # reference
        location_2_lat_lon_list=[lat, lon],             # candidate
        resolution="high",
        user_api_key=api_key
    )
    print(f"  {loc_id}: similarity = {score:.3f}")
```

---

### Use Case 2: Portfolio Benchmarking

**Scenario:** A property fund wants to compare its portfolio of 5 locations against 3 benchmark sites to understand which holdings are most similar to the targets.

```python
import eikonsai as eikon

api_key = eikon.utils.get_api_key_from_credentials(
    email="analyst@fund.com",
    password="password"
)

# Define portfolio and benchmark locations
portfolio_ids   = ["asset_1", "asset_2", "asset_3", "asset_4", "asset_5"]
portfolio_lats  = [51.507, 51.515, 51.523, 51.480, 51.530]
portfolio_lons  = [-0.128, -0.141, -0.158, -0.190, -0.100]

benchmark_ids   = ["bench_A", "bench_B", "bench_C"]
benchmark_lats  = [51.509, 51.518, 51.525]
benchmark_lons  = [-0.130, -0.145, -0.160]

# Build all pairwise combinations
import itertools

pairs = list(itertools.product(range(len(portfolio_ids)), range(len(benchmark_ids))))

orig_ids  = [portfolio_ids[p]  for p, _ in pairs]
dest_ids  = [benchmark_ids[b]  for _, b in pairs]
orig_lats = [portfolio_lats[p] for p, _ in pairs]
orig_lons = [portfolio_lons[p] for p, _ in pairs]
dest_lats = [benchmark_lats[b] for _, b in pairs]
dest_lons = [benchmark_lons[b] for _, b in pairs]

# Run batch comparison
results = eikon.jobs.eikon_portfolio_comparison(
    orig_uniq_id=orig_ids,
    dest_uniq_id=dest_ids,
    orig_lat_list=orig_lats,
    orig_lon_list=orig_lons,
    dest_lat_list=dest_lats,
    dest_lon_list=dest_lons,
    user_api_key=api_key,
    resolution="high",
    similarity_type="combined"
)

# Pivot to see portfolio vs benchmark matrix
pivot = results.pivot(index="orig", columns="dest", values="similarity_score")
print(pivot)
```

---

### Use Case 3: Change Detection & Monitoring

**Scenario:** A planning authority wants to profile and compare the character of two boroughs to understand neighbourhood differences.

```python
import eikonsai as eikon

api_key = eikon.utils.get_api_key_from_credentials(
    email="planner@council.gov.uk",
    password="password"
)

# Search for green spaces in two boroughs
camden_parks = eikon.jobs.search_api(
    my_search_prompt="Parks and green spaces with trees",
    user_api_key=api_key,
    effort_selection="moderate",
    spatial_resolution_for_search="London - boroughs",
    selected_london_borough="Camden"
)

hackney_parks = eikon.jobs.search_api(
    my_search_prompt="Parks and green spaces with trees",
    user_api_key=api_key,
    effort_selection="moderate",
    spatial_resolution_for_search="London - boroughs",
    selected_london_borough="Hackney"
)

print(f"Camden: {len(camden_parks)} green spaces found")
print(f"Hackney: {len(hackney_parks)} green spaces found")

# Get detailed profiles for top results
import h3

for _, row in camden_parks.head(3).iterrows():
    lat, lon = h3.h3_to_geo(row["location_id"])
    desc = eikon.context.get_location_description(
        lat=lat, lon=lon,
        resolution="high",
        user_api_key=api_key
    )
    print(f"\n{row['location_id']}:\n{desc}")
```

---

### Use Case 4: Location Profiling

**Scenario:** An analyst needs a comprehensive profile of a specific site, combining description, imagery, and context at multiple resolutions.

```python
import eikonsai as eikon

api_key = eikon.utils.get_api_key_from_credentials(
    email="analyst@co.com",
    password="password"
)

target_lat, target_lon = 51.5007, -0.1246  # Westminster

# Multi-resolution profiling
for res in ["low", "medium", "high"]:
    desc = eikon.context.get_location_description(
        lat=target_lat,
        lon=target_lon,
        resolution=res,
        user_api_key=api_key
    )
    print(f"\n--- {res.upper()} resolution ---")
    print(desc)

# Compare against a known similar location
score = eikon.similarity.visual_similarity(
    location_1_lat_lon_list=[target_lat, target_lon],
    location_2_lat_lon_list=[51.5155, -0.0922],  # Tower of London
    resolution="medium",
    user_api_key=api_key
)
print(f"\nSimilarity to Tower of London: {score:.3f}")
```

---

### Use Case 5: Conversational Geospatial Analysis

**Scenario:** A researcher wants to explore London's geography interactively, asking follow-up questions and getting maps generated on the fly.

```python
import eikonsai as eikon

api_key = eikon.utils.get_api_key_from_credentials(
    email="researcher@uni.ac.uk",
    password="password"
)

# Start an interactive session
eikon.jobs.eikon_ai_chat(user_api_key=api_key)

# Example conversation:
# > What are the largest green spaces in central London?
# > Can you show me a map of Hyde Park and Regent's Park?
# > How visually similar are these two parks?
# > What objects can you detect in the satellite imagery of Hyde Park?
# > /quit
```

The AI agent supports these capabilities within a conversation:
- **Topic search** -- find locations by theme (e.g., "parks", "industrial areas")
- **Map generation** -- visualise locations on interactive maps
- **Visual comparison** -- compare satellite imagery of two locations
- **Object detection** -- identify features (buildings, roads, water, vegetation, etc.) in satellite imagery
- **Location context** -- get detailed descriptions of any coordinate

---

## API Reference

### Module Summary

| Module | Function | Description |
|--------|----------|-------------|
| `eikon.utils` | `get_api_key_from_credentials(email, password)` | Authenticate and get API key |
| `eikon.utils` | `get_previous_search_api_results(api_key, num_requested_results)` | Retrieve past search results |
| `eikon.context` | `get_location_description(lat, lon, resolution, user_api_key)` | Text description of a location |
| `eikon.context` | `get_location_image(lat, lon, resolution, user_api_key)` | Satellite imagery of a location |
| `eikon.similarity` | `visual_similarity(location_1, location_2, resolution, user_api_key)` | Visual similarity score (0-1) |
| `eikon.jobs` | `search_api(prompt, api_key, effort, spatial_resolution, borough)` | Natural language location search |
| `eikon.jobs` | `eikon_portfolio_comparison(orig_ids, dest_ids, orig_lats, orig_lons, dest_lats, dest_lons, api_key, resolution, similarity_type)` | Batch pairwise comparison |
| `eikon.jobs` | `eikon_ai_chat(user_api_key)` | Interactive CLI chat session |

### Parameter Reference

**Resolution** (used across all modules):

| Value | Coverage | H3 Level |
|-------|----------|----------|
| `"low"` | ~2.5 km | Resolution 7 |
| `"medium"` | ~1 km | Resolution 8 |
| `"high"` | ~500 m | Resolution 9 |

**Effort Selection** (Search module):

| Value | Results | Speed |
|-------|---------|-------|
| `"test"` | ~3 | Fastest |
| `"quick"` | ~10 | Fast |
| `"moderate"` | ~20 | Moderate |
| `"exhaustive"` | ~50 | Slowest |

**Similarity Type** (Portfolio Comparison module):

| Value | Method |
|-------|--------|
| `"visual"` | Satellite imagery embedding comparison |
| `"descriptive"` | AI-generated text description comparison |
| `"combined"` | Weighted blend of visual and descriptive |

---

## Concepts

### H3 Hexagonal Grid

EIKON uses Uber's [H3 hexagonal grid system](https://h3geo.org/) to represent locations. Each location is identified by an H3 cell index (e.g., `89195da5e7bffff`), which maps to a specific hexagonal area on the Earth's surface. The resolution parameter controls the size of these hexagons.

You can convert between H3 indices and coordinates using the `h3` library:

```python
import h3

# H3 index to lat/lon
lat, lon = h3.h3_to_geo("89195da5e7bffff")

# lat/lon to H3 index at resolution 9
cell_id = h3.geo_to_h3(51.5074, -0.1278, 9)
```

### Credits

API calls consume credits from your account balance. Credit usage varies by operation and resolution level. Higher resolutions and more exhaustive searches consume more credits. Check your balance at any time:

```python
# Via the EIKON web app or API
# Credits can be purchased through the EIKON data store
```

### Search Architecture

Searches are processed asynchronously through a multi-stage pipeline. The SDK handles job submission and polling transparently -- `search_api()` blocks until results are ready and returns a DataFrame directly.

Behind the scenes:
1. Your query is submitted to a job queue
2. The system confirms the backend is available
3. The query passes through AI screening, spatial filtering, satellite analysis, and final ranking
4. Results are returned as a pandas DataFrame

### Satellite Imagery Analysis

EIKON's analysis is built on satellite imagery processed through computer vision models. The platform can:

- **Describe** locations using vision-language models
- **Compare** locations using image embedding similarity
- **Detect objects** using YOLO-based models (50 object categories including buildings, roads, water features, vegetation, vehicles, solar panels, and more)

### Object Detection Categories

The platform can detect 50 categories of objects in satellite imagery, including: building, tree, water, road, parking lot, roundabout, sports stadium, field, agricultural land, swimming pool, industrial land, lake, tennis court, cars, golf course, railway line, forest, motorway, solar panels, airplane, bridge, ship, and more.
