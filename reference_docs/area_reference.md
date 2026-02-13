# Eikon Area Reference Guide

This document provides a complete reference of all supported area names for the **Area Similarity** feature on the Eikon platform. Use these area names as inputs when comparing the similarity between two geographic areas.

Area names follow the [GADM](https://gadm.org/) Level 3 administrative boundary naming convention for the United Kingdom.

---

## How Area Similarity Works

The Area Similarity function compares two areas and returns a similarity score between **0** and **1**:

- **1.0** = the two areas are identical
- **0.0** = the two areas are completely dissimilar

You can compare areas using three different similarity types:

| Similarity Type | Description |
|---|---|
| `"visual"` | Compares areas based on how they look from satellite imagery |
| `"descriptive"` | Compares areas based on descriptive characteristics and features |
| `"combined"` | A blended score using both visual and descriptive similarity |

---

## Quick Start

```python
import eikonsai as eikon

# Compare two areas visually
result = eikon.similarity.area_similarity(
    area_1="Hackney",
    area_2="Camden",
    similarity_type="visual",
    user_api_key="your_api_key_here"
)

print(result)
```

**Example output:**

| orig | dest | similarity_score |
|---|---|---|
| Hackney | Camden | 0.85 |

The function returns a pandas DataFrame with the origin area, destination area, and the similarity score.

---

## Supported Area Names

There are **404** supported areas covering cities, boroughs, districts, and local authorities across England, Scotland, Wales, and Northern Ireland.

> **Area names are not case-sensitive.** For example, `"hackney"`, `"Hackney"`, and `"HACKNEY"` are all valid and will return the same result.

> **Multi-word area names are written without spaces**, following the GADM Level 3 naming convention. For example, the London borough of Tower Hamlets is written as `"TowerHamlets"`, and Brighton and Hove is written as `"BrightonandHove"`.

You can also retrieve the full list of supported area names programmatically:

```python
import eikonsai as eikon

supported_areas = eikon.utils.list_of_supported_areas()
print(supported_areas)
```

---

### England (324 areas)

| # | Area Name |
|---|---|
| 1 | Adur |
| 2 | Allerdale |
| 3 | AmberValley |
| 4 | Arun |
| 5 | Ashfield |
| 6 | Ashford |
| 7 | AylesburyVale |
| 8 | Babergh |
| 9 | BarkingandDagenham |
| 10 | Barnet |
| 11 | Barnsley |
| 12 | Barrow-in-Furness |
| 13 | Basildon |
| 14 | BasingstokeandDeane |
| 15 | Bassetlaw |
| 16 | BathandNorthEastSomerset |
| 17 | Bedford |
| 18 | Bexley |
| 19 | Birmingham |
| 20 | Blaby |
| 21 | BlackburnwithDarwen |
| 22 | Blackpool |
| 23 | Bolsover |
| 24 | Bolton |
| 25 | Boston |
| 26 | Bournemouth |
| 27 | BracknellForest |
| 28 | Bradford |
| 29 | Braintree |
| 30 | Breckland |
| 31 | Brent |
| 32 | Brentwood |
| 33 | BrightonandHove |
| 34 | Bristol |
| 35 | Broadland |
| 36 | Bromley |
| 37 | Bromsgrove |
| 38 | Broxbourne |
| 39 | Broxtowe |
| 40 | Burnley |
| 41 | Bury |
| 42 | Calderdale |
| 43 | Cambridge |
| 44 | Camden |
| 45 | CannockChase |
| 46 | Canterbury |
| 47 | Carlisle |
| 48 | CentralBedfordshire |
| 49 | Charnwood |
| 50 | Chelmsford |
| 51 | Cheltenham |
| 52 | Cherwell |
| 53 | CheshireEast |
| 54 | CheshireWestandChester |
| 55 | Chesterfield |
| 56 | Chichester |
| 57 | Chiltern |
| 58 | Chorley |
| 59 | Christchurch |
| 60 | CityofLondon |
| 61 | CityofWestminster |
| 62 | Colchester |
| 63 | Copeland |
| 64 | Corby |
| 65 | Cornwall |
| 66 | Cotswold |
| 67 | Coventry |
| 68 | Craven |
| 69 | Crawley |
| 70 | Croydon |
| 71 | Dacorum |
| 72 | Darlington |
| 73 | Dartford |
| 74 | Daventry |
| 75 | Derby |
| 76 | DerbyshireDales |
| 77 | Doncaster |
| 78 | Dover |
| 79 | Dudley |
| 80 | Durham |
| 81 | Ealing |
| 82 | EastCambridgeshire |
| 83 | EastDevon |
| 84 | EastDorset |
| 85 | EastHampshire |
| 86 | EastHertfordshire |
| 87 | EastLindsey |
| 88 | EastNorthamptonshire |
| 89 | EastRidingofYorkshire |
| 90 | EastStaffordshire |
| 91 | Eastbourne |
| 92 | Eastleigh |
| 93 | Eden |
| 94 | Elmbridge |
| 95 | Enfield |
| 96 | EppingForest |
| 97 | EpsomandEwell |
| 98 | Erewash |
| 99 | Exeter |
| 100 | Fareham |
| 101 | Fenland |
| 102 | ForestHeath |
| 103 | ForestofDean |
| 104 | Fylde |
| 105 | Gateshead |
| 106 | Gedling |
| 107 | Gloucester |
| 108 | Gosport |
| 109 | Gravesham |
| 110 | GreatYarmouth |
| 111 | Greenwich |
| 112 | Guildford |
| 113 | Hackney |
| 114 | Halton |
| 115 | Hambleton |
| 116 | HammersmithandFulham |
| 117 | Harborough |
| 118 | Haringey |
| 119 | Harlow |
| 120 | Harrogate |
| 121 | Harrow |
| 122 | Hart |
| 123 | Hartlepool |
| 124 | Hastings |
| 125 | Havant |
| 126 | Havering |
| 127 | Herefordshire |
| 128 | Hertsmere |
| 129 | HighPeak |
| 130 | Hillingdon |
| 131 | HinckleyandBosworth |
| 132 | Horsham |
| 133 | Hounslow |
| 134 | Huntingdonshire |
| 135 | Hyndburn |
| 136 | Ipswich |
| 137 | IsleofWight |
| 138 | IslesofScilly |
| 139 | Islington |
| 140 | KensingtonandChelsea |
| 141 | Kettering |
| 142 | King'sLynnandWestNorfolk |
| 143 | KingstonuponHull |
| 144 | KingstonuponThames |
| 145 | Kirklees |
| 146 | Knowsley |
| 147 | Lambeth |
| 148 | Lancaster |
| 149 | Leeds |
| 150 | Leicester |
| 151 | Lewes |
| 152 | Lewisham |
| 153 | Lichfield |
| 154 | Lincoln |
| 155 | Liverpool |
| 156 | Luton |
| 157 | Maidstone |
| 158 | Maldon |
| 159 | MalvernHills |
| 160 | Manchester |
| 161 | Mansfield |
| 162 | Medway |
| 163 | Melton |
| 164 | Mendip |
| 165 | Merton |
| 166 | MidDevon |
| 167 | MidSuffolk |
| 168 | MidSussex |
| 169 | Middlesbrough |
| 170 | MiltonKeynes |
| 171 | MoleValley |
| 172 | NewForest |
| 173 | NewarkandSherwood |
| 174 | Newcastle-under-Lyme |
| 175 | NewcastleuponTyne |
| 176 | Newham |
| 177 | NorthDevon |
| 178 | NorthDorset |
| 179 | NorthEastDerbyshire |
| 180 | NorthEastLincolnshire |
| 181 | NorthHertfordshire |
| 182 | NorthKesteven |
| 183 | NorthLincolnshire |
| 184 | NorthNorfolk |
| 185 | NorthSomerset |
| 186 | NorthTyneside |
| 187 | NorthWarwickshire |
| 188 | NorthWestLeicestershire |
| 189 | Northampton |
| 190 | Northumberland |
| 191 | Norwich |
| 192 | Nottingham |
| 193 | NuneatonandBedworth |
| 194 | OadbyandWigston |
| 195 | Oldham |
| 196 | Oxford |
| 197 | Pendle |
| 198 | Peterborough |
| 199 | Plymouth |
| 200 | Poole |
| 201 | Portsmouth |
| 202 | Preston |
| 203 | Purbeck |
| 204 | Reading |
| 205 | Redbridge |
| 206 | RedcarandCleveland |
| 207 | Redditch |
| 208 | ReigateandBanstead |
| 209 | RibbleValley |
| 210 | Richmondshire |
| 211 | RichmonduponThames |
| 212 | Rochdale |
| 213 | Rochford |
| 214 | Rossendale |
| 215 | Rother |
| 216 | Rotherham |
| 217 | Rugby |
| 218 | Runnymede |
| 219 | Rushcliffe |
| 220 | Rushmoor |
| 221 | Rutland |
| 222 | Ryedale |
| 223 | SaintAlbans |
| 224 | SaintEdmundsbury |
| 225 | SaintHelens |
| 226 | Salford |
| 227 | Sandwell |
| 228 | Scarborough |
| 229 | Sedgemoor |
| 230 | Sefton |
| 231 | Selby |
| 232 | Sevenoaks |
| 233 | Sheffield |
| 234 | Shepway |
| 235 | Shropshire |
| 236 | Slough |
| 237 | Solihull |
| 238 | SouthBucks |
| 239 | SouthCambridgeshire |
| 240 | SouthDerbyshire |
| 241 | SouthGloucestershire |
| 242 | SouthHams |
| 243 | SouthHolland |
| 244 | SouthKesteven |
| 245 | SouthLakeland |
| 246 | SouthNorfolk |
| 247 | SouthNorthamptonshire |
| 248 | SouthOxfordshire |
| 249 | SouthRibble |
| 250 | SouthSomerset |
| 251 | SouthStaffordshire |
| 252 | SouthTyneside |
| 253 | Southampton |
| 254 | Southend-on-Sea |
| 255 | Spelthorne |
| 256 | Stafford |
| 257 | StaffordshireMoorlands |
| 258 | Stevenage |
| 259 | Stockport |
| 260 | Stockton-on-Tees |
| 261 | Stoke-on-Trent |
| 262 | Stratford-on-Avon |
| 263 | Stroud |
| 264 | Suffolkcoastal |
| 265 | Sunderland |
| 266 | SurreyHeath |
| 267 | Sutton |
| 268 | Swale |
| 269 | Swindon |
| 270 | Tameside |
| 271 | Tamworth |
| 272 | Tandridge |
| 273 | TauntonDeane |
| 274 | Teignbridge |
| 275 | TelfordandWrekin |
| 276 | Tendring |
| 277 | TestValley |
| 278 | Tewkesbury |
| 279 | Thanet |
| 280 | ThreeRivers |
| 281 | Thurrock |
| 282 | TonbridgeandMalling |
| 283 | Torbay |
| 284 | Torridge |
| 285 | TowerHamlets |
| 286 | Trafford |
| 287 | TunbridgeWells |
| 288 | Uttlesford |
| 289 | ValeofWhiteHorse |
| 290 | Wakefield |
| 291 | Walsall |
| 292 | WalthamForest |
| 293 | Wandsworth |
| 294 | Warrington |
| 295 | Warwick |
| 296 | Watford |
| 297 | Waveney |
| 298 | Waverley |
| 299 | Wealden |
| 300 | Wellingborough |
| 301 | WelwynHatfield |
| 302 | WestBerkshire |
| 303 | WestDevon |
| 304 | WestDorset |
| 305 | WestLancashire |
| 306 | WestLindsey |
| 307 | WestOxfordshire |
| 308 | WestSomerset |
| 309 | WeymouthandPortland |
| 310 | Wigan |
| 311 | Wiltshire |
| 312 | Winchester |
| 313 | WindsorandMaidenhead |
| 314 | Wirral |
| 315 | Woking |
| 316 | Wokingham |
| 317 | Wolverhampton |
| 318 | Worcester |
| 319 | Worthing |
| 320 | Wychavon |
| 321 | Wycombe |
| 322 | Wyre |
| 323 | WyreForest |
| 324 | York |

---

### Scotland (32 areas)

| # | Area Name |
|---|---|
| 1 | Aberdeen |
| 2 | Aberdeenshire |
| 3 | Angus |
| 4 | ArgyllandBute |
| 5 | Clackmannanshire |
| 6 | DumfriesandGalloway |
| 7 | Dundee |
| 8 | EastAyrshire |
| 9 | EastDunbartonshire |
| 10 | EastLothian |
| 11 | EastRenfrewshire |
| 12 | Edinburgh |
| 13 | EileanSiar |
| 14 | Falkirk |
| 15 | Fife |
| 16 | Glasgow |
| 17 | Highland |
| 18 | Inverclyde |
| 19 | Midlothian |
| 20 | Moray |
| 21 | NorthAyrshire |
| 22 | NorthLanarkshire |
| 23 | OrkneyIslands |
| 24 | PerthshireandKinross |
| 25 | Renfrewshire |
| 26 | ScottishBorders |
| 27 | ShetlandIslands |
| 28 | SouthAyrshire |
| 29 | SouthLanarkshire |
| 30 | Stirling |
| 31 | WestDunbartonshire |
| 32 | WestLothian |

---

### Wales (22 areas)

| # | Area Name |
|---|---|
| 1 | Anglesey |
| 2 | BlaenauGwent |
| 3 | Bridgend |
| 4 | Caerphilly |
| 5 | Cardiff |
| 6 | Carmarthenshire |
| 7 | Ceredigion |
| 8 | Conwy |
| 9 | Denbighshire |
| 10 | Flintshire |
| 11 | Gwynedd |
| 12 | MerthyrTydfil |
| 13 | Monmouthshire |
| 14 | NeathPortTalbot |
| 15 | Newport |
| 16 | Pembrokeshire |
| 17 | Powys |
| 18 | Rhondda,Cynon,Taff |
| 19 | Swansea |
| 20 | Torfaen |
| 21 | ValeofGlamorgan |
| 22 | Wrexham |

---

### Northern Ireland (26 areas)

| # | Area Name |
|---|---|
| 1 | Antrim |
| 2 | Ards |
| 3 | Armagh |
| 4 | Ballymena |
| 5 | Ballymoney |
| 6 | Banbridge |
| 7 | Belfast |
| 8 | Carrickfergus |
| 9 | Castlereagh |
| 10 | Coleraine |
| 11 | Cookstown |
| 12 | Craigavon |
| 13 | Derry |
| 14 | Down |
| 15 | Dungannon |
| 16 | Fermanagh |
| 17 | Larne |
| 18 | Limavady |
| 19 | Lisburn |
| 20 | Magherafelt |
| 21 | Moyle |
| 22 | NewryandMourne |
| 23 | Newtownabbey |
| 24 | NorthDown |
| 25 | Omagh |
| 26 | Strabane |

---

## Usage Examples

### Example 1: Compare two London boroughs (visual similarity)

```python
import eikonsai as eikon

result = eikon.similarity.area_similarity(
    area_1="Hackney",
    area_2="Islington",
    similarity_type="visual",
    user_api_key="your_api_key_here"
)

print(result)
```

### Example 2: Compare two cities (descriptive similarity)

```python
import eikonsai as eikon

result = eikon.similarity.area_similarity(
    area_1="Manchester",
    area_2="Liverpool",
    similarity_type="descriptive",
    user_api_key="your_api_key_here"
)

print(result)
```

### Example 3: Combined similarity between a London borough and another area

```python
import eikonsai as eikon

result = eikon.similarity.area_similarity(
    area_1="Camden",
    area_2="BrightonandHove",
    similarity_type="combined",
    user_api_key="your_api_key_here"
)

print(result)
```

### Example 4: Comparing areas across different UK nations

```python
import eikonsai as eikon

# Compare Edinburgh (Scotland) to Cardiff (Wales)
result = eikon.similarity.area_similarity(
    area_1="Edinburgh",
    area_2="Cardiff",
    similarity_type="visual",
    user_api_key="your_api_key_here"
)

print(result)
```

### Example 5: Case-insensitive input

```python
import eikonsai as eikon

# All of these are equivalent and will return the same result
result = eikon.similarity.area_similarity(
    area_1="belfast",
    area_2="birmingham",
    similarity_type="combined",
    user_api_key="your_api_key_here"
)

print(result)
```

### Example 6: Listing all supported areas programmatically

```python
import eikonsai as eikon

# Get the full list of supported area names
supported_areas = eikon.utils.list_of_supported_areas()
print(supported_areas)
```

---

## Important Notes

- **Area names are not case-sensitive** - `"hackney"`, `"Hackney"`, and `"HACKNEY"` will all work.
- **Multi-word area names have no spaces** - follow the GADM Level 3 convention (e.g., `"TowerHamlets"`, `"NewcastleuponTyne"`, `"BrightonandHove"`).
- **Hyphenated names keep their hyphens** - e.g., `"Barrow-in-Furness"`, `"Stockton-on-Tees"`, `"Stoke-on-Trent"`, `"Stratford-on-Avon"`.
- Both `area_1` and `area_2` must be valid area names from the supported list above.
- You need a valid API key to use this endpoint. Register at [https://slugai.pagekite.me/register](https://slugai.pagekite.me/register) to get one.
- The function returns a pandas DataFrame if successful, or `None` if the request fails.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `None` returned | Check that both area names are valid names from the supported list above |
| API error | Verify your API key is valid and you have sufficient credits |
| Area not found | Make sure multi-word names have no spaces (e.g., `"TowerHamlets"` not `"Tower Hamlets"`) |
