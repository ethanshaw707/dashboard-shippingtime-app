import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import polars as pl
import pydeck as pdk
from pathlib import Path
from itertools import combinations

from network_utils import compute_best_origin_map


DATASETS = {
    "Small": "complete_shipping_data_size_s.csv",
    "Medium": "complete_shipping_data_size_m.csv",
    "Large": "complete_shipping_data_size_l.csv",
}
EXCLUDED_CITIES = {
    "Accord",
    "Acton",
    "Ada",
    "Adams Run",
    "Adamsville",
    "Adirondack",
    "Adrian",
    "Afton",
    "Ahwahnee",
    "Alamo",
    "Alapaha",
    "Alba",
}
STATE_COUNTS = {
    "CA": 832,
    "TX": 759,
    "FL": 573,
    "NY": 555,
    "PA": 440,
    "NC": 400,
    "OH": 386,
    "GA": 357,
    "IL": 334,
    "VA": 317,
    "NJ": 309,
    "MI": 300,
    "CO": 268,
    "MA": 266,
    "TN": 266,
    "WI": 230,
    "IN": 211,
    "WA": 210,
    "AZ": 207,
    "MO": 205,
    "MD": 200,
    "MN": 180,
    "OR": 154,
    "SC": 150,
    "CT": 132,
    "OK": 132,
    "AL": 129,
    "KY": 128,
    "LA": 102,
    "IA": 96,
    "ID": 96,
    "AR": 93,
    "KS": 85,
    "UT": 82,
    "NH": 75,
    "WV": 75,
    "ME": 72,
    "NV": 70,
    "NE": 62,
    "MS": 57,
    "NM": 53,
    "MT": 50,
    "ND": 49,
    "RI": 47,
    "DE": 44,
    "VT": 32,
    "WY": 30,
    "AK": 29,
    "HI": 21,
    "SD": 21,
    "DC": 10,
    "AE": 2,
    "AP": 2,
}
V3_CITY_COUNTS = {
    "brooklyn, ny": 39,
    "phoenix, az": 37,
    "houston, tx": 38,
    "san antonio, tx": 33,
    "jacksonville, fl": 21,
    "chicago, il": 28,
    "dallas, tx": 17,
    "wilmington, de": 7,
    "wilmington, nc": 8,
    "austin, tx": 26,
    "canton, oh": 4,
    "canton, ga": 15,
    "charlotte, nc": 26,
    "new york, ny": 25,
    "atlanta, ga": 24,
    "franklin, tn": 13,
    "lexington, ky": 11,
    "orlando, fl": 24,
    "rochester, ny": 14,
    "las vegas, nv": 23,
    "louisville, ky": 19,
    "marietta, ga": 19,
    "colorado springs, co": 22,
    "columbus, oh": 9,
    "marble falls, tx": 22,
    "tampa, fl": 22,
    "san diego, ca": 22,
    "columbia, sc": 3,
    "columbia, mo": 13,
    "greenville, sc": 4,
    "lebanon, tn": 8,
    "salem, or": 11,
    "cleveland, oh": 2,
    "cleveland, tn": 11,
    "denver, co": 17,
    "omaha, ne": 17,
    "portland, or": 17,
    "raleigh, nc": 19,
    "richmond, va": 7,
    "jackson, ms": 1,
    "albany, ny": 11,
    "fort worth, tx": 18,
    "oklahoma city, ok": 18,
    "chesapeake, va": 17,
    "miami, fl": 17,
    "milford, ct": 4,
    "pittsburgh, pa": 17,
    "san francisco, ca": 17,
    "san jose, ca": 17,
    "tucson, az": 17,
    "virginia beach, va": 17,
    "albuquerque, nm": 16,
    "gilbert, az": 15,
    "springfield, il": 5,
    "alexandria, va": 12,
    "kansas city, mo": 15,
    "knoxville, tn": 14,
    "lakewood, co": 6,
    "nashville, tn": 15,
    "saint louis, mo": 15,
    "seattle, wa": 15,
    "arlington, tx": 5,
    "aurora, co": 9,
    "boise, id": 14,
    "indianapolis, in": 14,
    "naples, fl": 13,
    "reno, nv": 14,
    "riverside, ca": 13,
    "bismarck, nd": 13,
    "fresno, tx": 1,
    "fresno, ca": 12,
    "lincoln, ri": 1,
    "lincoln, ne": 8,
    "los angeles, ca": 13,
    "mansfield, ma": 2,
    "milwaukee, wi": 13,
    "madison, in": 1,
    "mesa, az": 13,
    "monroe, nc": 4,
    "philadelphia, pa": 13,
    "staten island, ny": 13,
    "washington, dc": 10,
    "baltimore, md": 11,
    "decatur, il": 2,
    "durham, nc": 10,
    "fayetteville, ar": 4,
    "glendale, az": 7,
    "midland, ga": 6,
    "scottsdale, az": 12,
    "woodstock, ga": 6,
    "chandler, az": 8,
    "cincinnati, oh": 11,
    "covington, la": 3,
    "fort myers, fl": 11,
    "greensboro, nc": 10,
    "henderson, tx": 2,
    "henderson, nv": 8,
    "hamilton, mi": 1,
    "lewisburg, pa": 10,
    "littleton, co": 10,
    "mount pleasant, sc": 7,
    "pasadena, md": 3,
    "plano, tx": 11,
    "sanford, nc": 4,
    "wayne, me": 2,
    "ashland, pa": 2,
    "burlington, wi": 1,
    "centerville, ia": 1,
    "clayton, nc": 4,
    "encinitas, ca": 10,
    "fort collins, co": 10,
    "golden, co": 9,
    "hudson, nc": 2,
    "katy, tx": 10,
    "lancaster, sc": 1,
    "newark, nj": 3,
    "parker, co": 9,
    "plymouth, ma": 1,
    "rancho cucamonga, ca": 10,
    "rockville, md": 8,
    "sacramento, ca": 10,
    "tallahassee, fl": 10,
    "the villages, fl": 10,
    "westfield, in": 6,
    "waterloo, ia": 1,
    "west chester, pa": 7,
    "bend, or": 10,
    "amarillo, tx": 9,
    "auburn, ma": 2,
    "bakersfield, ca": 9,
    "boca raton, fl": 9,
    "benton, la": 4,
    "bradenton, fl": 9,
    "dayton, oh": 7,
    "duluth, mn": 7,
    "englewood, co": 6,
    "fort wayne, in": 9,
    "frederick, md": 8,
    "fredericksburg, va": 7,
    "gainesville, fl": 4,
    "grand rapids, mi": 9,
    "green bay, wi": 9,
    "morgantown, wv": 8,
    "pensacola, fl": 9,
    "princeton, ma": 1,
    "peoria, az": 6,
    "saint paul, mn": 9,
    "savannah, ga": 8,
    "seaford, ny": 5,
    "seymour, in": 3,
    "spokane, wa": 9,
    "windsor, co": 3,
    "winston-salem, nc": 0,
    "worcester, ma": 9,
    "york, pa": 5,
    "antioch, tn": 2,
    "athens, ga": 3,
    "corona, ca": 8,
    "fairfield, ca": 3,
    "germantown, md": 6,
    "hillsboro, or": 4,
    "hope mills, nc": 8,
    "hendersonville, nc": 4,
    "holly springs, ms": 2,
    "homer, ak": 5,
    "laurel, md": 6,
    "long beach, ca": 7,
    "manteca, ca": 8,
    "maryville, tn": 7,
    "magnolia, ms": 2,
    "memphis, tn": 8,
    "milton, ma": 1,
    "montgomery, tx": 5,
    "myrtle beach, sc": 8,
    "new market, al": 4,
    "ocala, fl": 8,
    "olathe, ks": 8,
    "oxford, mi": 4,
    "palm beach gardens, fl": 8,
    "roseville, ca": 7,
    "roswell, ga": 7,
    "rockwall, tx": 8,
    "spring, tx": 8,
    "salisbury, ma": 2,
    "santa rosa, ca": 8,
    "sarasota, fl": 8,
    "vancouver, wa": 8,
    "westminster, md": 5,
    "arvada, co": 7,
    "anchorage, ak": 7,
    "birmingham, al": 7,
    "bozeman, mt": 7,
    "conroe, tx": 7,
    "carmel valley, ca": 7,
    "castle rock, co": 7,
    "cedar rapids, ia": 7,
    "cooperstown, ny": 7,
    "dayton, oh": 7,
    "deland, fl": 7,
    "duluth, mn": 7,
    "fort mill, sc": 7,
    "fargo, nd": 7,
    "fredericksburg, va": 7,
    "glen allen, va": 7,
    "glendale, az": 7,
    "grand junction, co": 7,
    "highlands ranch, co": 7,
    "joliet, il": 7,
    "kerrville, tx": 7,
    "longmont, co": 7,
    "long beach, ca": 7,
    "maryville, tn": 7,
    "mount pleasant, sc": 7,
    "minneapolis, mn": 7,
    "naperville, il": 7,
    "neosho, mo": 7,
    "puyallup, wa": 7,
    "queen creek, az": 7,
    "richmond, va": 7,
    "roseville, ca": 7,
    "roswell, ga": 7,
    "saline, mi": 7,
    "sevierville, tn": 7,
    "sterling heights, mi": 7,
    "thornton, co": 7,
    "ventura, ca": 7,
    "wilmington, ma": 7,
    "west chester, pa": 7,
    "clermont, fl": 7,
    "abilene, tx": 6,
    "brick, nj": 6,
    "bentonville, ar": 6,
    "bogart, ga": 6,
    "boston, ma": 6,
    "buffalo, ny": 6,
    "cary, nc": 6,
    "dallas, ga": 6,
    "doylestown, pa": 6,
    "englewood, co": 6,
    "eugene, or": 6,
    "edmond, ok": 6,
    "forney, tx": 6,
    "fuquay varina, nc": 6,
    "germantown, md": 6,
    "glendora, ca": 6,
    "henrico, va": 6,
    "hockessin, de": 6,
    "holly springs, nc": 6,
    "irvine, ca": 6,
    "lakeland, fl": 6,
    "lawrence, ks": 6,
    "league city, tx": 6,
    "mansfield, tx": 6,
    "madison, wi": 6,
    "mckinney, tx": 6,
    "mentor, oh": 6,
    "mount airy, md": 6,
    "norfolk, va": 6,
    "nampa, id": 6,
    "noblesville, in": 6,
    "orland park, il": 6,
    "palm harbor, fl": 6,
    "petaluma, ca": 6,
    "placentia, ca": 6,
    "prescott, az": 6,
    "richardson, tx": 6,
    "roanoke, va": 6,
    "richmond, tx": 6,
    "sanford, me": 6,
    "seminole, fl": 6,
    "sheridan, wy": 6,
    "sparks, nv": 6,
    "traverse city, mi": 6,
    "tulsa, ok": 6,
    "wake forest, nc": 6,
    "woodstock, ga": 6,
    "waterloo, sc": 6,
    "waukesha, wi": 6,
    "wichita, ks": 6,
    "winnsboro, tx": 6,
    "woodbury, mn": 6,
    "cumming, ga": 6,
    "muskegon, mi": 6,
    "arlington, tx": 5,
    "akron, oh": 5,
    "albany, or": 5,
    "alpharetta, ga": 5,
    "asheville, nc": 5,
    "bixby, ok": 5,
    "boulder, co": 5,
    "bethlehem, pa": 5,
    "brenham, tx": 5,
    "broken bow, ok": 5,
    "cape coral, fl": 5,
    "chula vista, ca": 5,
    "crawfordville, fl": 5,
    "cypress, tx": 5,
    "clearwater, fl": 5,
    "columbiana, al": 5,
    "columbus, ga": 5,
    "commack, ny": 5,
    "dublin, ca": 5,
    "decatur, ga": 5,
    "deltona, fl": 5,
    "derry, nh": 5,
    "douglassville, pa": 5,
    "easton, pa": 5,
    "el paso, tx": 5,
    "escondido, ca": 5,
    "fairport, ny": 5,
    "flemington, nj": 5,
    "florence, sc": 5,
    "fort lauderdale, fl": 5,
    "frisco, tx": 5,
    "goodyear, az": 5,
    "gays mills, wi": 5,
    "grafton, wi": 5,
    "green cove springs, fl": 5,
    "greensburg, pa": 5,
    "homer, ak": 5,
    "honolulu, hi": 5,
    "humble, tx": 5,
    "huntington, ny": 5,
    "idaho falls, id": 5,
    "jackson, tn": 5,
    "jacksonville, nc": 5,
    "jenkintown, pa": 5,
    "johnson city, tn": 5,
    "kingwood, tx": 5,
    "lansing, mi": 5,
    "lapeer, mi": 5,
    "lawrenceville, ga": 5,
    "lexington, nc": 5,
    "lubbock, tx": 5,
    "medina, oh": 5,
    "macomb, mi": 5,
    "magnolia, tx": 5,
    "martinsburg, wv": 5,
    "meridian, id": 5,
    "mobile, al": 5,
    "montgomery, tx": 5,
    "morrison, co": 5,
    "napa, ca": 5,
    "new braunfels, tx": 5,
    "new lenox, il": 5,
    "new orleans, la": 5,
    "newark, de": 5,
    "oakland, ca": 5,
    "owasso, ok": 5,
    "overland park, ks": 5,
    "panama city, fl": 5,
    "panama city beach, fl": 5,
    "patchogue, ny": 5,
    "port clinton, oh": 5,
    "paducah, ky": 5,
    "palmetto, fl": 5,
    "pearland, tx": 5,
    "pella, ia": 5,
    "pittsford, ny": 5,
    "port orange, fl": 5,
    "ravenna, oh": 5,
    "roaming shores, oh": 5,
    "richmond hill, ga": 5,
    "riverview, fl": 5,
    "rochester, mn": 5,
    "round rock, tx": 5,
    "saint petersburg, fl": 5,
    "san leandro, ca": 5,
    "southington, ct": 5,
    "spartanburg, sc": 5,
    "surprise, az": 5,
    "santa barbara, ca": 5,
    "seaford, ny": 5,
    "shelton, ct": 5,
    "spring branch, tx": 5,
    "springfield, il": 5,
    "st. louis, mo": 5,
    "stroudsburg, pa": 5,
    "sunapee, nh": 5,
    "swedesboro, nj": 5,
    "syracuse, ny": 5,
    "tomball, tx": 5,
    "topeka, ks": 5,
    "tuscaloosa, al": 5,
    "terre haute, in": 5,
    "tinley park, il": 5,
    "toms river, nj": 5,
    "tulare, ca": 5,
    "vienna, va": 5,
    "walnut creek, ca": 5,
    "wayne, nj": 5,
    "waynesboro, va": 5,
    "west sacramento, ca": 5,
    "westminster, md": 5,
    "yuba city, ca": 5,
    "zachary, la": 5,
    "buford, ga": 5,
    "modesto, ca": 5,
    "winter park, fl": 5,
    "antioch, il": 4,
    "algonquin, il": 4,
    "allen, tx": 4,
    "ames, ia": 4,
    "anaheim, ca": 4,
    "aptos, ca": 4,
    "astoria, ny": 4,
    "bloomington, in": 4,
    "bremerton, wa": 4,
    "brunswick, oh": 4,
    "burke, va": 4,
    "bedford, in": 4,
    "benton, la": 4,
    "billings, mt": 4,
    "birdsboro, pa": 4,
    "bothell, wa": 4,
    "bowie, md": 4,
    "brentwood, tn": 4,
    "bronx, ny": 4,
    "brookfield, wi": 4,
    "canton, oh": 4,
    "celina, tx": 4,
    "chapel hill, nc": 4,
    "claremore, ok": 4,
    "clovis, ca": 4,
    "cocoa, fl": 4,
    "callahan, fl": 4,
    "carmel, in": 4,
    "cedar park, tx": 4,
    "centennial, co": 4,
    "centerville, ma": 4,
    "charleston, sc": 4,
    "charlottesville, va": 4,
    "clarkesville, ga": 4,
    "clayton, nc": 4,
    "clifton, nj": 4,
    "collinsville, ok": 4,
    "columbia, tn": 4,
    "concord, nc": 4,
    "cookeville, tn": 4,
    "corpus christi, tx": 4,
    "corrales, nm": 4,
    "corydon, in": 4,
    "cranston, ri": 4,
    "dekalb, il": 4,
    "delaware, oh": 4,
    "delmar, ny": 4,
    "des moines, ia": 4,
    "dickson, tn": 4,
    "downers grove, il": 4,
    "dripping springs, tx": 4,
    "eustis, fl": 4,
    "east hampton, ny": 4,
    "fenton, mo": 4,
    "fayetteville, ar": 4,
    "flower mound, tx": 4,
    "franklin, oh": 4,
    "friendswood, tx": 4,
    "gainesville, fl": 4,
    "glendale, ca": 4,
    "greenville, sc": 4,
    "greenville, mi": 4,
    "gainesville, ga": 4,
    "gaithersburg, md": 4,
    "garner, nc": 4,
    "gastonia, nc": 4,
    "granada hills, ca": 4,
    "granbury, tx": 4,
    "griffin, ga": 4,
    "harpers ferry, wv": 4,
    "haverhill, ma": 4,
    "hayward, ca": 4,
    "herriman, ut": 4,
    "hillsboro, or": 4,
    "hoboken, nj": 4,
    "huntley, il": 4,
    "hagerstown, md": 4,
    "hamilton, oh": 4,
    "harrisburg, pa": 4,
    "hendersonville, nc": 4,
    "hendersonville, tn": 4,
    "holland, mi": 4,
    "indian trail, nc": 4,
    "jersey city, nj": 4,
    "johnstown, pa": 4,
    "kalamazoo, mi": 4,
    "keatchie, la": 4,
    "kennewick, wa": 4,
    "kingsport, tn": 4,
    "lake elsinore, ca": 4,
    "lebanon, oh": 4,
    "little rock, ar": 4,
    "livonia, mi": 4,
    "los alamos, nm": 4,
    "loveland, co": 4,
    "la mesa, ca": 4,
    "la quinta, ca": 4,
    "lake oswego, or": 4,
    "lake saint louis, mo": 4,
    "lake in the hills, il": 4,
    "lakeway, tx": 4,
    "lancaster, ny": 4,
    "leander, tx": 4,
    "lees summit, mo": 4,
    "lenoir, nc": 4,
    "lewis center, oh": 4,
    "lexington, in": 4,
    "lodi, ca": 4,
    "lynden, wa": 4,
    "methuen, ma": 4,
    "milford, ct": 4,
    "mineola, tx": 4,
    "missoula, mt": 4,
    "mansfield, oh": 4,
    "mcdonough, ga": 4,
    "merritt island, fl": 4,
    "midlothian, va": 4,
    "millington, tn": 4,
    "milton, fl": 4,
    "moab, ut": 4,
    "monroe, nc": 4,
    "mooresville, in": 4,
    "morganton, nc": 4,
    "murfreesboro, tn": 4,
    "newburgh, ny": 4,
    "navarre, fl": 4,
    "new market, al": 4,
    "newnan, ga": 4,
    "norman, ok": 4,
    "novi, mi": 4,
    "ormond beach, fl": 4,
    "owensboro, ky": 4,
    "oconto, wi": 4,
    "old bridge, nj": 4,
    "olympia, wa": 4,
    "oregon city, or": 4,
    "oxford, mi": 4,
    "peachtree city, ga": 4,
    "pleasanton, ca": 4,
    "poughkeepsie, ny": 4,
    "pagosa springs, co": 4,
    "pasadena, ca": 4,
    "pasadena, tx": 4,
    "pasco, wa": 4,
    "pawtucket, ri": 4,
    "pembroke, ma": 4,
    "perrysburg, oh": 4,
    "plant city, fl": 4,
    "port orchard, wa": 4,
    "prescott valley, az": 4,
    "princeton, nj": 4,
    "quincy, ma": 4,
    "redwood city, ca": 4,
    "rohnert park, ca": 4,
    "rigby, id": 4,
    "ripon, wi": 4,
    "rogers, ar": 4,
    "rowlett, tx": 4,
    "rydal, pa": 4,
    "shepherdsville, ky": 4,
    "shreveport, la": 4,
    "stamford, ct": 4,
    "salt lake city, ut": 4,
    "sanford, nc": 4,
    "santa ana, ca": 4,
    "santa cruz, ca": 4,
    "santa fe, nm": 4,
    "sarver, pa": 4,
    "schenectady, ny": 4,
    "sierra vista, az": 4,
    "silver spring, md": 4,
    "south jordan, ut": 4,
    "springfield, mo": 4,
    "summerville, sc": 4,
    "sumter, sc": 4,
    "sun city, az": 4,
    "syosset, ny": 4,
    "taunton, ma": 4,
    "texarkana, tx": 4,
    "toledo, oh": 4,
    "urbana, oh": 4,
    "valparaiso, in": 4,
    "waco, tx": 4,
    "west lafayette, in": 4,
    "winter garden, fl": 4,
    "winter haven, fl": 4,
    "wylie, tx": 4,
    "waldorf, md": 4,
    "wesley chapel, fl": 4,
    "west palm beach, fl": 4,
    "williamsburg, va": 4,
    "woodbridge, va": 4,
    "woodside, ca": 4,
    "yorba linda, ca": 4,
    "queens village, ny": 4,
    "alvin, tx": 3,
    "ann arbor, mi": 3,
    "anna, tx": 3,
    "apopka, fl": 3,
    "argyle, tx": 3,
    "ashdown, ar": 3,
    "auburn, ca": 3,
    "adel, ia": 3,
    "aiken, sc": 3,
    "alameda, ca": 3,
    "aliso viejo, ca": 3,
    "ambler, pa": 3,
    "anacortes, wa": 3,
    "ankeny, ia": 3,
    "annandale, va": 3,
    "apache junction, az": 3,
    "appleton, wi": 3,
    "arlington, wa": 3,
    "arlington, va": 3,
    "ashland, va": 3,
    "ashton, id": 3,
    "atascadero, ca": 3,
    "athens, ga": 3,
    "athens, tx": 3,
    "atwater, ca": 3,
    "ballston lake, ny": 3,
    "barnegat, nj": 3,
    "beaver falls, pa": 3,
    "beavercreek, or": 3,
    "bethalto, il": 3,
    "big lake, mn": 3,
    "blairsville, ga": 3,
    "boynton beach, fl": 3,
    "brownsville, tx": 3,
    "bruceton mills, wv": 3,
    "byron, ga": 3,
    "banner elk, nc": 3,
    "barrington, il": 3,
    "bayonne, nj": 3,
    "beach lake, pa": 3,
    "beaumont, ca": 3,
    "beaverton, or": 3,
    "belleview, fl": 3,
    "bellevue, ne": 3,
    "belton, tx": 3,
    "belton, mo": 3,
    "benton, ar": 3,
    "berlin, md": 3,
    "bethesda, md": 3,
    "bettendorf, ia": 3,
    "bloomfield, ct": 3,
    "bogota, nj": 3,
    "bowling green, ky": 3,
    "bridgeton, nj": 3,
    "broken arrow, ok": 3,
    "broomfield, co": 3,
    "buckeye, az": 3,
    "bulverde, tx": 3,
    "burlington, nc": 3,
    "campbell, ca": 3,
    "carlsbad, ca": 3,
    "carson city, nv": 3,
    "chagrin falls, oh": 3,
    "champaign, il": 3,
    "chesterton, in": 3,
    "clarkston, mi": 3,
    "cordele, ga": 3,
    "canby, or": 3,
    "canfield, oh": 3,
    "canonsburg, pa": 3,
    "canton, mi": 3,
    "canyon country, ca": 3,
    "catonsville, md": 3,
    "centerville, oh": 3,
    "central islip, ny": 3,
    "chandler, ok": 3,
    "chattanooga, tn": 3,
    "cherry hill, nj": 3,
    "chico, ca": 3,
    "chino hills, ca": 3,
    "christiansburg, va": 3,
    "clarksburg, md": 3,
    "clarksville, tn": 3,
    "colonial beach, va": 3,
    "columbia, sc": 3,
    "columbia city, in": 3,
    "columbia falls, mt": 3,
    "conway, sc": 3,
    "conyers, ga": 3,
    "coral springs, fl": 3,
    "corvallis, or": 3,
    "coventry, ri": 3,
    "covina, ca": 3,
    "covington, la": 3,
    "crescent city, ca": 3,
    "dunedin, fl": 3,
    "dacula, ga": 3,
    "dartmouth, ma": 3,
    "dearborn heights, mi": 3,
    "decatur, tx": 3,
    "dickinson, tx": 3,
    "doral, fl": 3,
    "dothan, al": 3,
    "el cajon, ca": 3,
    "elgin, il": 3,
    "elk river, mn": 3,
    "erie, pa": 3,
    "everett, wa": 3,
    "evergreen, co": 3,
    "excelsior, mn": 3,
    "east concord, ny": 3,
    "east greenwich, ri": 3,
    "east islip, ny": 3,
    "eau claire, wi": 3,
    "edison, nj": 3,
    "elizabeth, co": 3,
    "elk grove, ca": 3,
    "ellijay, ga": 3,
    "erie, co": 3,
    "fallbrook, ca": 3,
    "forked river, nj": 3,
    "fort white, fl": 3,
    "fairfax, va": 3,
    "fairfield, ca": 3,
    "fairfield, ct": 3,
    "fairhope, al": 3,
    "fallon, nv": 3,
    "farmington hills, mi": 3,
    "fayetteville, tn": 3,
    "findlay, oh": 3,
    "fisherville, ky": 3,
    "fitchburg, ma": 3,
    "flagstaff, az": 3,
    "florissant, mo": 3,
    "folsom, ca": 3,
    "fond du lac, wi": 3,
    "fontana, ca": 3,
    "forest hills, ny": 3,
    "freehold, nj": 3,
    "georgetown, tx": 3,
    "gig harbor, wa": 3,
    "gillette, wy": 3,
    "gilroy, ca": 3,
    "goshen, in": 3,
    "grand forks, nd": 3,
    "gallatin, tn": 3,
    "geneseo, il": 3,
    "gorham, me": 3,
    "greeneville, tn": 3,
    "greenfield, wi": 3,
    "greenville, nc": 3,
    "grovetown, ga": 3,
    "hampstead, md": 3,
    "hampton, va": 3,
    "haslet, tx": 3,
    "hamilton, nj": 3,
    "happy valley, or": 3,
    "hattiesburg, ms": 3,
    "havertown, pa": 3,
    "healdsburg, ca": 3,
    "herndon, va": 3,
    "hewitt, nj": 3,
    "hickory, nc": 3,
    "hilliard, oh": 3,
    "howell, mi": 3,
    "huffman, tx": 3,
    "iron station, nc": 3,
    "irwin, pa": 3,
    "jackson, mi": 3,
    "jefferson city, mo": 3,
    "jamestown, ri": 3,
    "jesup, ga": 3,
    "juneau, ak": 3,
    "kalispell, mt": 3,
    "kingman, az": 3,
    "kingston, tn": 3,
    "kissimmee, fl": 3,
    "kanab, ut": 3,
    "kenosha, wi": 3,
    "la pine, or": 3,
    "lafayette, la": 3,
    "lake stevens, wa": 3,
    "lakeville, mn": 3,
    "leslie, mi": 3,
    "liberty hill, tx": 3,
    "lone tree, co": 3,
    "long island, me": 3,
    "longview, tx": 3,
    "la verne, ca": 3,
    "lake elmo, mn": 3,
    "lake harmony, pa": 3,
    "lake jackson, tx": 3,
    "lake worth, fl": 3,
    "lakewood, ca": 3,
    "lakewood, oh": 3,
    "lander, wy": 3,
    "lebanon, pa": 3,
    "lehi, ut": 3,
    "lewes, de": 3,
    "longwood, fl": 3,
    "loveland, oh": 3,
    "lufkin, tx": 3,
    "macon, ga": 3,
    "marshville, nc": 3,
    "maspeth, ny": 3,
    "mcgregor, mn": 3,
    "melbourne, fl": 3,
    "metairie, la": 3,
    "milford, oh": 3,
    "missouri valley, ia": 3,
    "monterey, ca": 3,
    "madison, al": 3,
    "madisonville, tn": 3,
    "makawao, hi": 3,
    "manchester, nh": 3,
    "marion, oh": 3,
    "marlborough, ma": 3,
    "massapequa, ny": 3,
    "matawan, nj": 3,
    "matthews, nc": 3,
    "mechanicsburg, pa": 3,
    "medford, or": 3,
    "melville, ny": 3,
    "menasha, wi": 3,
    "middlebury, in": 3,
    "middletown, ny": 3,
    "midland, mi": 3,
    "milford, mi": 3,
    "milford, de": 3,
    "minnetonka, mn": 3,
    "minot, nd": 3,
    "mint hill, nc": 3,
    "mokena, il": 3,
    "monroe, ga": 3,
    "moreno valley, ca": 3,
    "mount juliet, tn": 3,
    "mount laurel, nj": 3,
    "mount prospect, il": 3,
    "murphy, nc": 3,
    "murrieta, ca": 3,
    "natick, ma": 3,
    "new fairfield, ct": 3,
    "norristown, pa": 3,
    "nutley, nj": 3,
    "new brighton, mn": 3,
    "new hartford, ny": 3,
    "new kensington, pa": 3,
    "newark, nj": 3,
    "newport news, va": 3,
    "niceville, fl": 3,
    "norco, ca": 3,
    "north aurora, il": 3,
    "north mankato, mn": 3,
    "north port, fl": 3,
    "northampton, pa": 3,
    "nottingham, md": 3,
    "o fallon, mo": 3,
    "orchard park, ny": 3,
    "oakhurst, tx": 3,
    "oakley, ca": 3,
    "oceanside, ca": 3,
    "ogden, ut": 3,
    "oswego, il": 3,
    "palmerton, pa": 3,
    "pasadena, md": 3,
    "peconic, ny": 3,
    "piedmont, ok": 3,
    "pigeon forge, tn": 3,
    "pittsboro, nc": 3,
    "plainwell, mi": 3,
    "pleasantville, ny": 3,
    "prairie village, ks": 3,
    "priest river, id": 3,
    "paramus, nj": 3,
    "parkville, md": 3,
    "pataskala, oh": 3,
    "pembroke pines, fl": 3,
    "peoria, il": 3,
    "perry hall, md": 3,
    "perryville, mo": 3,
    "plain city, oh": 3,
    "pleasant hill, ia": 3,
    "plover, wi": 3,
    "plymouth, mi": 3,
    "pooler, ga": 3,
    "port saint lucie, fl": 3,
    "potomac, md": 3,
    "pottstown, pa": 3,
    "prattville, al": 3,
    "providence, ri": 3,
    "racine, wi": 3,
    "raeford, nc": 3,
    "rancho santa fe, ca": 3,
    "richmond, ut": 3,
    "rio rancho, nm": 3,
    "rochester, ma": 3,
    "rocky river, oh": 3,
    "romulus, mi": 3,
    "rathdrum, id": 3,
    "reading, ma": 3,
    "redondo beach, ca": 3,
    "ridge, ny": 3,
    "robertsdale, al": 3,
    "rockford, mi": 3,
    "rockland, me": 3,
    "rotterdam, ny": 3,
    "round hill, va": 3,
    "rye, ny": 3,
    "saint augustine, fl": 3,
    "saint george, ut": 3,
    "sandy springs, ga": 3,
    "seguin, tx": 3,
    "springdale, ar": 3,
    "stafford, va": 3,
    "stillwater, mn": 3,
    "saginaw, mi": 3,
    "salisbury, md": 3,
    "san ramon, ca": 3,
    "sandusky, oh": 3,
    "sandy, ut": 3,
    "santa maria, ca": 3,
    "saratoga, ca": 3,
    "saratoga springs, ny": 3,
    "scarborough, me": 3,
    "scotch plains, nj": 3,
    "seaford, de": 3,
    "severna park, md": 3,
    "sewell, nj": 3,
    "seymour, in": 3,
    "shakopee, mn": 3,
    "sherman, tx": 3,
    "sherwood, or": 3,
    "shingle springs, ca": 3,
    "simi valley, ca": 3,
    "simpsonville, sc": 3,
    "sioux falls, sd": 3,
    "slidell, la": 3,
    "smithfield, va": 3,
    "sonora, ca": 3,
    "south elgin, il": 3,
    "spokane valley, wa": 3,
    "spooner, wi": 3,
    "spotsylvania, va": 3,
    "spring creek, nv": 3,
    "spring hill, tn": 3,
    "springtown, tx": 3,
    "star, id": 3,
    "strongsville, oh": 3,
    "suffield, ct": 3,
    "suffolk, va": 3,
    "sugar land, tx": 3,
    "suisun city, ca": 3,
    "sumas, wa": 3,
    "sunnyvale, ca": 3,
    "superior, wi": 3,
    "thompsons station, tn": 3,
    "troy, oh": 3,
    "trussville, al": 3,
    "taylorsville, nc": 3,
    "temecula, ca": 3,
    "the woodlands, tx": 3,
    "thibodaux, la": 3,
    "tremont, il": 3,
    "trenton, mi": 3,
    "turlock, ca": 3,
    "tyler, tx": 3,
    "union, nj": 3,
    "union city, ca": 3,
    "venice, fl": 3,
    "verona, wi": 3,
    "voorheesville, ny": 3,
    "wheeling, wv": 3,
    "wadsworth, oh": 3,
    "washougal, wa": 3,
    "wasilla, ak": 3,
    "wausau, wi": 3,
    "wauwatosa, wi": 3,
    "wayland, mi": 3,
    "wayne, pa": 3,
    "wellesley, ma": 3,
    "wenatchee, wa": 3,
    "wentzville, mo": 3,
    "west chester, oh": 3,
    "west linn, or": 3,
    "west milford, nj": 3,
    "west newbury, ma": 3,
    "west warwick, ri": 3,
    "westerville, oh": 3,
    "westford, ma": 3,
    "white plains, ny": 3,
    "whitmore lake, mi": 3,
    "wichita falls, tx": 3,
    "wickenburg, az": 3,
    "wilbraham, ma": 3,
    "willis, tx": 3,
    "willow spring, nc": 3,
    "wilton, ct": 3,
    "windermere, fl": 3,
    "windsor, co": 3,
    "winterville, nc": 3,
    "woodhaven, mi": 3,
    "woodinville, wa": 3,
    "woods cross, ut": 3,
    "wurtsboro, ny": 3,
    "xenia, oh": 3,
    "yadkinville, nc": 3,
    "yorktown, va": 3,
    "yorktown heights, ny": 3,
    "yuma, az": 3,
    "zephyrhills, fl": 3,
    "zanesville, oh": 3,
    "des plaines, il": 3,
    "east northport, ny": 3,
    "floral park, ny": 3,
    "hingham, ma": 3,
    "loganville, ga": 3,
    "ossining, ny": 3,
    "parkersburg, wv": 3,
    "tacoma, wa": 3,


}
DESTINATION_WEIGHTS = {}
DEFAULT_DEST_WEIGHT = 1.0


@st.cache_data
def load_data(path: str, drop_missing: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    if "ToCity" in df.columns:
        df = df[~df["ToCity"].isin(EXCLUDED_CITIES)]
    for col in ["Cost", "ShippingTimeDays"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if drop_missing:
        df = df.dropna(subset=["Cost", "ShippingTimeDays"])
    df["Destination"] = df["ToCity"].str.strip() + ", " + df["ToState"].str.strip()
    return df


@st.cache_data
def load_lat_long_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def minmax(series: pd.Series) -> pd.Series:
    if series.nunique() <= 1:
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / (series.max() - series.min())

def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    if total == 0:
        return float(np.mean(values)) if len(values) else float("nan")
    return float(np.sum(values * weights) / total)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    total = weights_sorted.sum()
    if total == 0:
        return float(np.median(values_sorted))
    cum_weights = np.cumsum(weights_sorted)
    idx = np.searchsorted(cum_weights, total / 2.0)
    return float(values_sorted[min(idx, len(values_sorted) - 1)])


def nearest_bin(value: float, bins: np.ndarray) -> float:
    if bins.size == 0 or not np.isfinite(value):
        return float("nan")
    idx = int(np.abs(bins - value).argmin())
    return float(bins[idx])


def initial_destination_weights(df: pd.DataFrame, state_counts: dict) -> dict:
    total = sum(state_counts.values())
    if total <= 0 or df.empty:
        return {}
    state_pct = {state: count / total * 100.0 for state, count in state_counts.items()}
    city_counts = df.groupby("ToState")["ToCity"].nunique()
    unique_rows = df[["ToCity", "ToState"]].drop_duplicates().copy()
    unique_rows["Destination"] = (
        unique_rows["ToCity"].astype(str).str.strip()
        + ", "
        + unique_rows["ToState"].astype(str).str.strip()
    )
    unique_rows["StatePct"] = unique_rows["ToState"].map(state_pct)
    unique_rows["CityCount"] = unique_rows["ToState"].map(city_counts)
    valid = unique_rows["StatePct"].notna() & unique_rows["CityCount"].gt(0)
    unique_rows["Weight"] = np.where(
        valid,
        unique_rows["StatePct"] / unique_rows["CityCount"],
        DEFAULT_DEST_WEIGHT,
    )
    return dict(zip(unique_rows["Destination"], unique_rows["Weight"].astype(float)))


def initial_destination_weights_by_city(df: pd.DataFrame, city_counts: dict) -> dict:
    if df.empty:
        return {}
    unique_rows = df[["ToCity", "ToState", "Destination"]].dropna(
        subset=["ToCity", "ToState", "Destination"]
    ).drop_duplicates().copy()
    unique_rows["City"] = unique_rows["ToCity"].astype(str).str.strip()
    unique_rows["State"] = unique_rows["ToState"].astype(str).str.strip()
    unique_rows["Destination"] = unique_rows["Destination"].astype(str).str.strip()
    unique_rows["CityKey"] = unique_rows["City"].str.casefold() + ", " + unique_rows["State"].str.casefold()

    cities_in_df = set(unique_rows["CityKey"].unique())
    city_counts_filtered = {city: count for city, count in city_counts.items() if city in cities_in_df}
    total = sum(city_counts_filtered.values())
    if total <= 0:
        return {}
    city_pct = pd.Series(city_counts_filtered, dtype=float) / total * 100.0
    city_dest_counts = (
        unique_rows.groupby(["City", "State"])["Destination"]
        .nunique()
        .rename("CityDestCount")
        .reset_index()
    )
    unique_rows = unique_rows.merge(city_dest_counts, on=["City", "State"], how="left")
    unique_rows["CityPct"] = unique_rows["CityKey"].map(city_pct)
    valid = unique_rows["CityPct"].notna() & unique_rows["CityDestCount"].gt(0)
    unique_rows["Weight"] = np.where(
        valid,
        unique_rows["CityPct"] / unique_rows["CityDestCount"],
        DEFAULT_DEST_WEIGHT,
    )
    return dict(zip(unique_rows["Destination"], unique_rows["Weight"].astype(float)))


def add_weighted_score(df: pd.DataFrame, cost_weight: float) -> pd.DataFrame:
    df = df.copy()
    df["CostNorm"] = minmax(df["Cost"])
    df["TimeNorm"] = minmax(df["ShippingTimeDays"])
    df["WeightedScore"] = cost_weight * df["CostNorm"] + (1 - cost_weight) * df["TimeNorm"]
    return df


def apply_filters(df: pd.DataFrame, cities, origins) -> pd.DataFrame:
    mask = df["ToCity"].isin(cities) & df["FromAddress"].isin(origins)
    return df[mask].copy()

def destination_weights(destinations, weight_map: dict) -> dict:
    weights = {dest: float(weight_map.get(dest, DEFAULT_DEST_WEIGHT)) for dest in destinations}
    total = sum(weights.values())
    if total == 0:
        equal = 1.0 / max(len(destinations), 1)
        return {dest: equal for dest in destinations}
    return {dest: val / total for dest, val in weights.items()}


def _dict_to_items(weight_map: dict) -> tuple:
    return tuple(sorted(weight_map.items()))


# Computes the same aggregate outputs used by the network builder for a selected origin set.
def compute_built_network_summary(
    built_subset: pd.DataFrame,
    dest_weights: dict,
) -> dict:
    built_best_time = built_subset.groupby("Destination", as_index=False).agg(
        BestTime=("ShippingTimeDays", "min")
    )
    coverage_origin_map = compute_best_origin_map(built_subset)
    built_best_time["Weight"] = built_best_time["Destination"].map(dest_weights).fillna(0.0)

    one_day_df = built_best_time[built_best_time["BestTime"] <= 1.0].copy()
    two_day_df = built_best_time[built_best_time["BestTime"] == 2.0].copy()
    three_day_df = built_best_time[built_best_time["BestTime"] == 3.0].copy()

    weight_rank = (
        pd.Series(dest_weights)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "Destination", 0: "Weight"})
    )
    weight_rank["Rank"] = np.arange(1, len(weight_rank) + 1)
    rank_map = dict(zip(weight_rank["Destination"], weight_rank["Rank"]))

    total_weight = float(built_best_time["Weight"].sum())
    one_day_weight = float(one_day_df["Weight"].sum())
    two_day_coverage_weight = float(
        built_best_time[built_best_time["BestTime"] <= 2.0]["Weight"].sum()
    )
    three_day_coverage_weight = float(
        built_best_time[built_best_time["BestTime"] <= 3.0]["Weight"].sum()
    )
    one_day_coverage = (one_day_weight / total_weight * 100.0) if total_weight else 0.0
    two_day_coverage = (two_day_coverage_weight / total_weight * 100.0) if total_weight else 0.0
    three_day_coverage = (three_day_coverage_weight / total_weight * 100.0) if total_weight else 0.0

    built_avg_time = weighted_mean(
        built_best_time["BestTime"].to_numpy(),
        built_best_time["Weight"].to_numpy(),
    )
    built_best_cost = built_subset.groupby("Destination", as_index=False).agg(
        BestCost=("Cost", "min")
    )
    built_best_cost["Weight"] = built_best_cost["Destination"].map(dest_weights).fillna(0.0)
    built_avg_cost = weighted_mean(
        built_best_cost["BestCost"].to_numpy(),
        built_best_cost["Weight"].to_numpy(),
    )

    def prep_day_display(day_df: pd.DataFrame) -> pd.DataFrame:
        if day_df.empty:
            return day_df
        display_df = day_df[["Destination", "Weight"]].copy()
        display_df["PriorityRank"] = display_df["Destination"].map(rank_map)
        display_df["CoverageFrom"] = display_df["Destination"].map(coverage_origin_map)
        return display_df.sort_values(
            ["PriorityRank", "Destination"],
            ascending=[True, True],
        )[["Destination", "Weight", "PriorityRank", "CoverageFrom"]]

    slow_1 = built_best_time[built_best_time["BestTime"] > 1.0]["Destination"].sort_values().tolist()
    slow_2 = built_best_time[built_best_time["BestTime"] > 2.0]["Destination"].sort_values().tolist()
    slow_3 = built_best_time[built_best_time["BestTime"] > 3.0]["Destination"].sort_values().tolist()

    return {
        "built_best_time": built_best_time,
        "coverage_origin_map": coverage_origin_map,
        "rank_map": rank_map,
        "best_time_map": dict(zip(built_best_time["Destination"], built_best_time["BestTime"])),
        "avg_time": built_avg_time,
        "avg_cost": built_avg_cost,
        "coverage_1_day": one_day_coverage,
        "coverage_2_day": two_day_coverage,
        "coverage_3_day": three_day_coverage,
        "one_day_display": prep_day_display(one_day_df),
        "two_day_display": prep_day_display(two_day_df),
        "three_day_display": prep_day_display(three_day_df),
        "slow_1": slow_1,
        "slow_2": slow_2,
        "slow_3": slow_3,
    }


# Determines which side wins a metric so the better side can be highlighted.
def compare_metric_winners(
    left_value: float,
    right_value: float,
    higher_is_better: bool,
) -> tuple[bool, bool]:
    left_finite = bool(np.isfinite(left_value))
    right_finite = bool(np.isfinite(right_value))
    if not left_finite and not right_finite:
        return False, False
    if left_finite and not right_finite:
        return True, False
    if right_finite and not left_finite:
        return False, True
    if np.isclose(left_value, right_value):
        return True, True
    if higher_is_better:
        return left_value > right_value, right_value > left_value
    return left_value < right_value, right_value < left_value


# Renders a metric value and colors the winner in green.
def render_colored_metric(label: str, value_text: str, highlight: bool) -> None:
    value_color = "#16a34a" if highlight else "#ffffff"
    st.markdown(
        (
            "<div style='padding: 0.1rem 0;'>"
            f"<div style='font-size:0.85rem;color:#6b7280'>{label}</div>"
            f"<div style='font-size:1.25rem;font-weight:600;color:{value_color}'>{value_text}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


# Renders one full network-builder panel so compare mode matches the existing builder details.
def render_network_builder_panel(
    summary: dict | None,
    panel_label: str,
    key_prefix: str,
    highlight_map: dict | None = None,
) -> None:
    highlight_map = highlight_map or {}
    if summary is None:
        st.caption(f"No origins selected for {panel_label}.")
        return

    metric_col_a, metric_col_b = st.columns(2)
    with metric_col_a:
        render_colored_metric(
            f"Built network avg time ({panel_label})",
            f"{summary['avg_time']:.2f}",
            bool(highlight_map.get("avg_time", False)),
        )
    with metric_col_b:
        render_colored_metric(
            f"Built network avg cost ({panel_label})",
            f"{summary['avg_cost']:.2f}",
            bool(highlight_map.get("avg_cost", False)),
        )

    coverage_col_a, coverage_col_b, coverage_col_c = st.columns(3)
    with coverage_col_a:
        render_colored_metric(
            f"1-day coverage (weighted) ({panel_label})",
            f"{summary['coverage_1_day']:.1f}%",
            bool(highlight_map.get("coverage_1_day", False)),
        )
    with coverage_col_b:
        render_colored_metric(
            f"2-day coverage (weighted) ({panel_label})",
            f"{summary['coverage_2_day']:.1f}%",
            bool(highlight_map.get("coverage_2_day", False)),
        )
    with coverage_col_c:
        render_colored_metric(
            f"3-day coverage (weighted) ({panel_label})",
            f"{summary['coverage_3_day']:.1f}%",
            bool(highlight_map.get("coverage_3_day", False)),
        )


# Returns the top priority cities uniquely covered by one network at a given day threshold.
def unique_priority_cities_for_threshold(
    primary_summary: dict | None,
    other_summary: dict | None,
    threshold_days: float,
    limit: int = 5,
) -> pd.DataFrame:
    if primary_summary is None:
        return pd.DataFrame(columns=["Destination", "Weight", "PriorityRank", "CoverageFrom"])

    primary_times = primary_summary["built_best_time"][["Destination", "BestTime", "Weight"]].copy()
    other_map = other_summary["best_time_map"] if other_summary else {}
    primary_times["OtherBestTime"] = primary_times["Destination"].map(other_map)

    unique_mask = (primary_times["BestTime"] <= threshold_days) & (
        primary_times["OtherBestTime"].isna() | (primary_times["OtherBestTime"] > threshold_days)
    )
    unique_df = primary_times[unique_mask].copy()
    if unique_df.empty:
        return pd.DataFrame(columns=["Destination", "Weight", "PriorityRank", "CoverageFrom"])

    unique_df["PriorityRank"] = unique_df["Destination"].map(primary_summary["rank_map"])
    unique_df["CoverageFrom"] = unique_df["Destination"].map(primary_summary["coverage_origin_map"])
    unique_df = unique_df.sort_values(
        ["Weight", "PriorityRank", "Destination"],
        ascending=[False, True, True],
    ).head(limit)
    return unique_df[["Destination", "Weight", "PriorityRank", "CoverageFrom"]]


# Calculates projected monthly profit for a built network using weighted package distribution and day-based profit lifts.
def compute_network_roi_projection(
    summary: dict | None,
    destination_weights_map: dict,
    monthly_packages: int,
    base_revenue_per_package: float,
    base_profit_per_package: float,
    one_day_bonus: float,
    two_day_bonus: float,
    three_day_penalty: float,
) -> dict | None:
    if summary is None:
        return None
    if monthly_packages <= 0:
        return {
            "projected_revenue": 0.0,
            "projected_revenue_yearly": 0.0,
            "projected_profit": 0.0,
            "projected_profit_yearly": 0.0,
            "shipping_uplift_profit": 0.0,
            "avg_profit_per_package": 0.0,
            "one_day_packages": 0.0,
            "two_day_packages": 0.0,
            "three_day_packages": 0.0,
        }

    weight_series = pd.Series(destination_weights_map, dtype=float)
    total_weight = float(weight_series.sum())
    if total_weight <= 0:
        return {
            "projected_revenue": 0.0,
            "projected_revenue_yearly": 0.0,
            "projected_profit": 0.0,
            "projected_profit_yearly": 0.0,
            "shipping_uplift_profit": 0.0,
            "avg_profit_per_package": 0.0,
            "one_day_packages": 0.0,
            "two_day_packages": 0.0,
            "three_day_packages": 0.0,
        }

    best_time_map = summary["best_time_map"]
    dist_df = weight_series.rename("WeightShare").reset_index().rename(columns={"index": "Destination"})
    dist_df["WeightShare"] = dist_df["WeightShare"] / total_weight
    dist_df["Packages"] = dist_df["WeightShare"] * float(monthly_packages)
    dist_df["BestTime"] = dist_df["Destination"].map(best_time_map)
    dist_df["BestTime"] = dist_df["BestTime"].fillna(np.inf)

    one_day_mask = dist_df["BestTime"] <= 1.0
    two_day_mask = (dist_df["BestTime"] > 1.0) & (dist_df["BestTime"] <= 2.0)
    three_day_mask = (dist_df["BestTime"] > 2.0) & (dist_df["BestTime"] <= 3.0)
    dist_df["BonusPerPackage"] = np.where(
        one_day_mask,
        one_day_bonus,
        np.where(two_day_mask, two_day_bonus, np.where(three_day_mask, -three_day_penalty, 0.0)),
    )
    dist_df["RevenueBonusPerPackage"] = np.where(
        one_day_mask,
        one_day_bonus,
        np.where(two_day_mask, two_day_bonus, 0.0),
    )
    dist_df["RevenuePerPackage"] = base_revenue_per_package + dist_df["RevenueBonusPerPackage"]
    dist_df["ProfitPerPackage"] = base_profit_per_package + dist_df["BonusPerPackage"]

    projected_revenue = float(np.sum(dist_df["Packages"] * dist_df["RevenuePerPackage"]))
    projected_revenue_yearly = projected_revenue * 12.0
    projected_profit = float(np.sum(dist_df["Packages"] * dist_df["ProfitPerPackage"]))
    projected_profit_yearly = projected_profit * 12.0
    base_profit = float(np.sum(dist_df["Packages"] * base_profit_per_package))
    shipping_uplift_profit = projected_profit - base_profit
    avg_profit_per_package = projected_profit / float(monthly_packages)
    one_day_packages = float(np.sum(dist_df.loc[one_day_mask, "Packages"]))
    two_day_packages = float(np.sum(dist_df.loc[two_day_mask, "Packages"]))
    three_day_packages = float(np.sum(dist_df.loc[dist_df["BestTime"] <= 3.0, "Packages"]))

    return {
        "projected_revenue": projected_revenue,
        "projected_revenue_yearly": projected_revenue_yearly,
        "projected_profit": projected_profit,
        "projected_profit_yearly": projected_profit_yearly,
        "shipping_uplift_profit": shipping_uplift_profit,
        "avg_profit_per_package": avg_profit_per_package,
        "one_day_packages": one_day_packages,
        "two_day_packages": two_day_packages,
        "three_day_packages": three_day_packages,
    }


def apply_shipping_cost_savings_to_projected_profit(
    roi_projection: dict | None,
    network_summary: dict | None,
    comparison_base_city_avg_cost: float | None,
    monthly_packages: int,
) -> dict | None:
    if roi_projection is None:
        return None

    adjusted_roi = roi_projection.copy()
    shipping_cost_savings_monthly = 0.0
    if (
        network_summary is not None
        and comparison_base_city_avg_cost is not None
        and np.isfinite(comparison_base_city_avg_cost)
        and np.isfinite(network_summary["avg_cost"])
    ):
        shipping_cost_savings_monthly = float(
            (comparison_base_city_avg_cost - network_summary["avg_cost"]) * monthly_packages
        )
    shipping_cost_savings_yearly = shipping_cost_savings_monthly * 12.0
    avg_shipping_saved_per_package = (
        shipping_cost_savings_monthly / float(monthly_packages)
        if monthly_packages > 0
        else 0.0
    )
    adjusted_roi["shipping_cost_savings_vs_base_city_monthly"] = shipping_cost_savings_monthly
    adjusted_roi["shipping_cost_savings_vs_base_city_yearly"] = shipping_cost_savings_yearly
    adjusted_roi["avg_shipping_saved_per_package_vs_base_city"] = avg_shipping_saved_per_package
    adjusted_roi["projected_profit"] = float(
        adjusted_roi["projected_profit"] + shipping_cost_savings_monthly
    )
    adjusted_roi["projected_profit_yearly"] = float(
        adjusted_roi["projected_profit_yearly"] + shipping_cost_savings_yearly
    )
    adjusted_roi["avg_profit_per_package"] = (
        adjusted_roi["projected_profit"] / float(monthly_packages)
        if monthly_packages > 0
        else 0.0
    )
    return adjusted_roi


def recommend_next_city_by_profit(
    full_df: pd.DataFrame,
    selected_origins: list[str],
    origin_options: list[str],
    destination_weights_map: dict,
    monthly_packages: int,
    base_revenue_per_package: float,
    base_profit_per_package: float,
    one_day_bonus: float,
    two_day_bonus: float,
    three_day_penalty: float,
    comparison_base_city_avg_cost: float | None = None,
) -> tuple[str | None, float | None]:
    if not selected_origins:
        return None, None

    remaining_origins = [origin for origin in origin_options if origin not in selected_origins]
    if not remaining_origins:
        return None, None

    base_subset = full_df[full_df["FromAddress"].isin(selected_origins)]
    if base_subset.empty:
        return None, None
    base_summary = compute_built_network_summary(base_subset, destination_weights_map)
    base_roi = compute_network_roi_projection(
        base_summary,
        destination_weights_map,
        monthly_packages,
        base_revenue_per_package,
        base_profit_per_package,
        one_day_bonus,
        two_day_bonus,
        three_day_penalty,
    )
    base_roi = apply_shipping_cost_savings_to_projected_profit(
        base_roi,
        base_summary,
        comparison_base_city_avg_cost,
        monthly_packages,
    )
    if base_roi is None:
        return None, None

    best_origin = None
    best_profit_lift = float("-inf")
    for candidate_origin in remaining_origins:
        candidate_subset = full_df[
            full_df["FromAddress"].isin(selected_origins + [candidate_origin])
        ]
        if candidate_subset.empty:
            continue
        candidate_summary = compute_built_network_summary(
            candidate_subset,
            destination_weights_map,
        )
        candidate_roi = compute_network_roi_projection(
            candidate_summary,
            destination_weights_map,
            monthly_packages,
            base_revenue_per_package,
            base_profit_per_package,
            one_day_bonus,
            two_day_bonus,
            three_day_penalty,
        )
        candidate_roi = apply_shipping_cost_savings_to_projected_profit(
            candidate_roi,
            candidate_summary,
            comparison_base_city_avg_cost,
            monthly_packages,
        )
        if candidate_roi is None:
            continue

        profit_lift = float(candidate_roi["projected_profit"] - base_roi["projected_profit"])
        if (
            best_origin is None
            or profit_lift > best_profit_lift
            or (np.isclose(profit_lift, best_profit_lift) and candidate_origin < best_origin)
        ):
            best_origin = candidate_origin
            best_profit_lift = profit_lift

    if best_origin is None:
        return None, None
    return best_origin, best_profit_lift


def render_top_combos(
    df: pd.DataFrame,
    origin_list: list,
    full_origin_list: list | None,
    dest_in_view: list,
    dest_weights: dict,
    build_cost_weight: float,
    avg_cost: float,
    avg_time: float,
    major_weight_map: dict,
    key_prefix: str,
    show_day_percentages: bool = False,
    max_k: int = 10,
    baseline_origin: str | None = None,
    required_origin: str | None = None,
) -> None:
    st.markdown("Recommended starting two (synergy pairs)")
    show_key = f"{key_prefix}_show_top_combos"
    if show_key not in st.session_state:
        st.session_state[show_key] = False
    if st.button("Clear combo cache", key=f"{key_prefix}_clear_combo_cache"):
        compute_pair_df_cached.clear()
        compute_top_k_combos_cached.clear()
        compute_top_k_combos_with_required_cached.clear()
        st.caption("Combo cache cleared.")
    if st.button("Compute top combos", key=f"{key_prefix}_compute_top_combos"):
        st.session_state[show_key] = True

    if not st.session_state[show_key]:
        st.caption('Top combos are hidden until you click "Compute top combos".')
        return

    dest_weights_items = _dict_to_items(dest_weights)
    origin_list_local = list(origin_list)
    small_origin_list = list(full_origin_list or origin_list_local)
    if required_origin:
        if required_origin in small_origin_list and required_origin not in origin_list_local:
            origin_list_local.append(required_origin)
        if required_origin not in small_origin_list:
            small_origin_list.append(required_origin)
    required_small = (required_origin,) if required_origin and required_origin in small_origin_list else tuple()
    required_local = (required_origin,) if required_origin and required_origin in origin_list_local else tuple()
    small_cost_mat, small_time_mat = build_origin_destination_matrices(
        df,
        tuple(small_origin_list),
        tuple(dest_in_view),
    )
    cost_mat, time_mat = build_origin_destination_matrices(
        df,
        tuple(origin_list_local),
        tuple(dest_in_view),
    )
    weights_vec = np.array([dest_weights.get(dest, 0.0) for dest in dest_in_view], dtype=float)
    if weights_vec.sum() == 0:
        weights_vec = np.ones(len(dest_in_view), dtype=float)

    def combo_avg_cost_time(combo_indices):
        best_cost = np.nanmin(cost_mat[combo_indices, :], axis=0)
        best_time = np.nanmin(time_mat[combo_indices, :], axis=0)
        valid_mask = ~np.isnan(best_cost) & ~np.isnan(best_time)
        if not valid_mask.any():
            return float("nan"), float("nan")
        weights = weights_vec[valid_mask]
        if weights.sum() == 0:
            weights = np.ones(len(weights), dtype=float)
        avg_cost_local = float(np.sum(best_cost[valid_mask] * weights) / weights.sum())
        avg_time_local = float(np.sum(best_time[valid_mask] * weights) / weights.sum())
        return avg_cost_local, avg_time_local

    def combo_weighted_total(combo_indices):
        avg_cost_local, avg_time_local = combo_avg_cost_time(combo_indices)
        if not np.isfinite(avg_cost_local) or not np.isfinite(avg_time_local):
            return float("nan"), avg_cost_local, avg_time_local
        weighted_total = build_cost_weight * avg_cost_local + (1 - build_cost_weight) * avg_time_local
        return weighted_total, avg_cost_local, avg_time_local

    def small_combo_avg_cost_time(combo_indices):
        best_cost = np.nanmin(small_cost_mat[combo_indices, :], axis=0)
        best_time = np.nanmin(small_time_mat[combo_indices, :], axis=0)
        valid_mask = ~np.isnan(best_cost) & ~np.isnan(best_time)
        if not valid_mask.any():
            return float("nan"), float("nan")
        weights = weights_vec[valid_mask]
        if weights.sum() == 0:
            weights = np.ones(len(weights), dtype=float)
        avg_cost_local = float(np.sum(best_cost[valid_mask] * weights) / weights.sum())
        avg_time_local = float(np.sum(best_time[valid_mask] * weights) / weights.sum())
        return avg_cost_local, avg_time_local

    def small_best_combo_indices(k_size: int):
        best_combo = None
        best_two_day_pct = float("-inf")
        best_weighted = float("inf")
        if required_small and k_size < len(required_small):
            return None
        for combo in combinations(range(len(small_origin_list)), k_size):
            if required_small:
                required_idx = small_origin_list.index(required_small[0])
                if required_idx not in combo:
                    continue
            avg_cost_local, avg_time_local = small_combo_avg_cost_time(list(combo))
            if not np.isfinite(avg_cost_local) or not np.isfinite(avg_time_local):
                continue
            weighted_total = build_cost_weight * avg_cost_local + (1 - build_cost_weight) * avg_time_local
            _, two_day_pct, _ = small_combo_day_percentages(list(combo))
            if (
                two_day_pct > best_two_day_pct
                or (np.isclose(two_day_pct, best_two_day_pct) and weighted_total < best_weighted)
            ):
                best_two_day_pct = two_day_pct
                best_weighted = weighted_total
                best_combo = list(combo)
        return best_combo

    def small_combo_time_improvements(combo_indices, prev_indices):
        if not prev_indices:
            return 0, 0
        best_time = np.nanmin(small_time_mat[combo_indices, :], axis=0)
        prev_time = np.nanmin(small_time_mat[prev_indices, :], axis=0)
        valid = ~np.isnan(best_time) & ~np.isnan(prev_time)
        if not valid.any():
            return 0, 0
        three_to_two = np.sum((prev_time[valid] > 2.0) & (best_time[valid] <= 2.0) & (best_time[valid] > 1.0))
        two_to_one = np.sum((prev_time[valid] > 1.0) & (best_time[valid] <= 1.0))
        return int(three_to_two), int(two_to_one)

    def small_combo_improved_cities(prev_indices, combo_indices):
        if not prev_indices:
            return []
        prev_time = np.nanmin(small_time_mat[prev_indices, :], axis=0)
        best_time = np.nanmin(small_time_mat[combo_indices, :], axis=0)
        improved = []
        for i, (p, b) in enumerate(zip(prev_time, best_time)):
            if np.isfinite(p) and np.isfinite(b) and b < p:
                improved.append(dest_in_view[i])
        return improved

    def small_combo_not_covered(combo_indices, threshold: float):
        best_time = np.nanmin(small_time_mat[combo_indices, :], axis=0)
        not_covered = []
        for i, t in enumerate(best_time):
            if np.isfinite(t) and t > threshold:
                not_covered.append(dest_in_view[i])
        return not_covered

    def small_combo_day_percentages(combo_indices):
        best_time = np.nanmin(small_time_mat[combo_indices, :], axis=0)
        valid = np.isfinite(best_time)
        if not valid.any():
            return 0.0, 0.0, 0.0
        valid_weights = weights_vec[valid]
        if valid_weights.sum() == 0:
            valid_weights = np.ones(len(valid_weights), dtype=float)
        total_weight = float(np.sum(valid_weights))
        one_day_weight = float(np.sum(valid_weights[best_time[valid] <= 1.0]))
        two_day_weight = float(np.sum(valid_weights[best_time[valid] <= 2.0]))
        three_day_weight = float(np.sum(valid_weights[best_time[valid] <= 3.0]))
        return (
            one_day_weight / total_weight * 100.0,
            two_day_weight / total_weight * 100.0,
            three_day_weight / total_weight * 100.0,
        )

    def best_combo_indices(k_size: int):
        best_combo = None
        best_two_day_pct = float("-inf")
        best_weighted = float("inf")
        if required_local and k_size < len(required_local):
            return None
        for combo in combinations(range(len(origin_list_local)), k_size):
            if required_local:
                required_idx = origin_list_local.index(required_local[0])
                if required_idx not in combo:
                    continue
            weighted_total, _, _ = combo_weighted_total(list(combo))
            if not np.isfinite(weighted_total):
                continue
            _, two_day_pct, _ = combo_day_percentages(list(combo))
            if (
                two_day_pct > best_two_day_pct
                or (np.isclose(two_day_pct, best_two_day_pct) and weighted_total < best_weighted)
            ):
                best_two_day_pct = two_day_pct
                best_weighted = weighted_total
                best_combo = list(combo)
        return best_combo

    def combo_time_improvements(combo_indices, prev_indices):
        if not prev_indices:
            return 0, 0
        best_time = np.nanmin(time_mat[combo_indices, :], axis=0)
        prev_time = np.nanmin(time_mat[prev_indices, :], axis=0)
        valid = ~np.isnan(best_time) & ~np.isnan(prev_time)
        if not valid.any():
            return 0, 0
        three_to_two = np.sum((prev_time[valid] > 2.0) & (best_time[valid] <= 2.0) & (best_time[valid] > 1.0))
        two_to_one = np.sum((prev_time[valid] > 1.0) & (best_time[valid] <= 1.0))
        return int(three_to_two), int(two_to_one)

    def combo_improved_cities(prev_indices, combo_indices):
        if not prev_indices:
            return []
        prev_time = np.nanmin(time_mat[prev_indices, :], axis=0)
        best_time = np.nanmin(time_mat[combo_indices, :], axis=0)
        improved = []
        for i, (p, b) in enumerate(zip(prev_time, best_time)):
            if np.isfinite(p) and np.isfinite(b) and b < p:
                improved.append(dest_in_view[i])
        return improved

    def combo_not_covered(combo_indices, threshold: float):
        best_time = np.nanmin(time_mat[combo_indices, :], axis=0)
        not_covered = []
        for i, t in enumerate(best_time):
            if np.isfinite(t) and t > threshold:
                not_covered.append(dest_in_view[i])
        return not_covered

    def format_not_covered(destinations):
        if not destinations:
            return "None"
        parts = []
        for dest in destinations:
            weight = dest_weights.get(dest, 0.0) * 100.0
            parts.append(f"{dest} ({weight:.1f}%)")
        return ", ".join(parts)

    def combo_day_percentages(combo_indices):
        best_time = np.nanmin(time_mat[combo_indices, :], axis=0)
        valid = np.isfinite(best_time)
        if not valid.any():
            return 0.0, 0.0, 0.0
        valid_weights = weights_vec[valid]
        if valid_weights.sum() == 0:
            valid_weights = np.ones(len(valid_weights), dtype=float)
        total_weight = float(np.sum(valid_weights))
        one_day_weight = float(np.sum(valid_weights[best_time[valid] <= 1.0]))
        two_day_weight = float(np.sum(valid_weights[best_time[valid] <= 2.0]))
        three_day_weight = float(np.sum(valid_weights[best_time[valid] <= 3.0]))
        return (
            one_day_weight / total_weight * 100.0,
            two_day_weight / total_weight * 100.0,
            three_day_weight / total_weight * 100.0,
        )

    def combo_indices_from_list(combo_list):
        if not combo_list:
            return None
        combo_name = combo_list[0][0]
        combo_names = combo_name.split(" + ")
        return [origin_list_local.index(name) for name in combo_names]

    def expand_from_best(prev_list, k_size):
        if not prev_list:
            return []
        best_name = prev_list[0][0]
        best_origins = best_name.split(" + ")
        if len(best_origins) != k_size - 1:
            return []
        prev_indices = [origin_list_local.index(name) for name in best_origins]
        _, prev_two_day_pct, _ = combo_day_percentages(prev_indices)
        results = []
        for origin in origin_list_local:
            if origin in best_origins:
                continue
            combo = best_origins + [origin]
            combo_indices = [origin_list_local.index(name) for name in combo]
            weighted_total, _, _ = combo_weighted_total(combo_indices)
            _, two_day_pct, _ = combo_day_percentages(combo_indices)
            two_day_gain = two_day_pct - prev_two_day_pct
            if np.isfinite(weighted_total):
                # Primary ranking: largest 2-day coverage gain. Tiebreaker: lower weighted total.
                results.append((" + ".join(combo), weighted_total, two_day_gain))
        results.sort(key=lambda x: (-x[2], x[1], x[0]))
        if len(results) <= 5:
            return [(name, weighted_total) for name, weighted_total, _ in results]
        top_results = results[:5]
        return [(name, weighted_total) for name, weighted_total, _ in top_results]

    pair_df = compute_pair_df_cached(
        df,
        tuple(small_origin_list),
        tuple(dest_in_view),
        build_cost_weight,
        _dict_to_items(dest_weights),
        _dict_to_items(major_weight_map),
    )
    if required_small:
        required_name = required_small[0]
        pair_df = pair_df[pair_df["Pair"].str.contains(required_name, regex=False)]

    if not pair_df.empty:
        best_value = float(pair_df["TwoDayCoveragePct"].iloc[0])
        top_mask = np.isclose(pair_df["TwoDayCoveragePct"], best_value)
        tied_pairs = pair_df[top_mask]["Pair"].tolist()
        if len(tied_pairs) > 5:
            top_pairs = pair_df[top_mask][["Pair", "WeightedTotal"]].copy()
        else:
            top_pairs = pair_df.head(5)[["Pair", "WeightedTotal"]].copy()
        pair_labels = [
            f"{pair} ({weighted_total:.2f})"
            for pair, weighted_total in zip(top_pairs["Pair"], top_pairs["WeightedTotal"])
        ]
        selected_pair = st.selectbox(
            "Top 5 pairs (2-day priority)",
            pair_labels,
            index=0,
            key=f"{key_prefix}_top5_pairs",
        )
        selected_pair_name = selected_pair.rsplit(" (", 1)[0]
        selected_row = pair_df[pair_df["Pair"] == selected_pair_name].iloc[0]
        combo_indices = [small_origin_list.index(name) for name in selected_pair_name.split(" + ")]
        combo_avg_cost, combo_avg_time = small_combo_avg_cost_time(combo_indices)
        cost_delta = avg_cost - combo_avg_cost if np.isfinite(combo_avg_cost) else float("nan")
        time_delta = avg_time - combo_avg_time if np.isfinite(combo_avg_time) else float("nan")
        baseline_indices = None
        if baseline_origin and baseline_origin in small_origin_list and (
            not required_small or baseline_origin == required_small[0]
        ):
            baseline_indices = [small_origin_list.index(baseline_origin)]
        prev_combo = baseline_indices or small_best_combo_indices(1)
        if prev_combo:
            prev_cost, prev_time = small_combo_avg_cost_time(prev_combo)
            prev_cost_delta = prev_cost - combo_avg_cost
            prev_time_delta = prev_time - combo_avg_time
        else:
            prev_cost_delta = float("nan")
            prev_time_delta = float("nan")
        move_3_to_2, move_2_to_1 = small_combo_time_improvements(combo_indices, prev_combo or [])
        st.caption(
            f"Weighted total: {selected_row['WeightedTotal']:.2f} - Avg cost: {combo_avg_cost:.2f} "
            f"(delta {cost_delta:.2f}) - Avg time: {combo_avg_time:.2f} (delta {time_delta:.2f}) - "
            f"Major coverage: {selected_row['MajorCoveragePct']:.1f}%"
        )
        st.caption(f"Delta vs best 1 origin: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
        st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
        pair_improved = small_combo_improved_cities(prev_combo or [], combo_indices)
        st.selectbox(
            "Improved destinations vs best 1 origin (pair)",
            pair_improved or ["None"],
            key=f"{key_prefix}_pair_no1_day",
        )
        if show_day_percentages:
            pct_1, pct_2, pct_3 = small_combo_day_percentages(combo_indices)
            st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
            not_one_day = small_combo_not_covered(combo_indices, 1.0)
            not_two_day = small_combo_not_covered(combo_indices, 2.0)
            st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
            st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
    else:
        st.caption("Not enough origins to calculate pairs.")

    st.markdown("Top 5 trios")
    trio_list = (
        compute_top_k_combos_with_required_cached(
            df,
            tuple(small_origin_list),
            tuple(dest_in_view),
            required_small,
            3,
            build_cost_weight,
            dest_weights_items,
        ) if len(small_origin_list) >= 3 else []
    )
    if trio_list:
        trio_labels = [f"{name} ({value:.2f})" for name, value in trio_list]
        selected_trio = st.selectbox(
            "Top 5 trios (2-day priority)",
            trio_labels,
            index=0,
            key=f"{key_prefix}_top5_trios",
        )
        trio_name = selected_trio.rsplit(" (", 1)[0]
        trio_indices = [small_origin_list.index(name) for name in trio_name.split(" + ")]
        trio_avg_cost, trio_avg_time = small_combo_avg_cost_time(trio_indices)
        prev_combo = small_best_combo_indices(2)
        if prev_combo:
            prev_cost, prev_time = small_combo_avg_cost_time(prev_combo)
            prev_cost_delta = prev_cost - trio_avg_cost
            prev_time_delta = prev_time - trio_avg_time
        else:
            prev_cost_delta = float("nan")
            prev_time_delta = float("nan")
        move_3_to_2, move_2_to_1 = small_combo_time_improvements(trio_indices, prev_combo or [])
        st.caption(
            f"Avg cost: {trio_avg_cost:.2f} (delta {avg_cost - trio_avg_cost:.2f}) - "
            f"Avg time: {trio_avg_time:.2f} (delta {avg_time - trio_avg_time:.2f})"
        )
        st.caption(f"Delta vs best 2 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
        st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
        trio_improved = small_combo_improved_cities(prev_combo or [], trio_indices)
        st.selectbox(
            "Improved destinations vs best 2 origins (trio)",
            trio_improved or ["None"],
            key=f"{key_prefix}_trio_no1_day",
        )
        if show_day_percentages:
            pct_1, pct_2, pct_3 = small_combo_day_percentages(trio_indices)
            st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
            not_one_day = small_combo_not_covered(trio_indices, 1.0)
            not_two_day = small_combo_not_covered(trio_indices, 2.0)
            st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
            st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
    else:
        st.caption("Not enough origins to calculate trios.")

    st.markdown("Top 5 quads")
    quad_list = (
        compute_top_k_combos_with_required_cached(
            df,
            tuple(origin_list_local),
            tuple(dest_in_view),
            required_local,
            4,
            build_cost_weight,
            dest_weights_items,
        ) if len(origin_list_local) >= 4 else []
    )
    if quad_list:
        quad_labels = [f"{name} ({value:.2f})" for name, value in quad_list]
        selected_quad = st.selectbox(
            "Top 5 quads (2-day priority)",
            quad_labels,
            index=0,
            key=f"{key_prefix}_top5_quads",
        )
        quad_name = selected_quad.rsplit(" (", 1)[0]
        quad_indices = [origin_list_local.index(name) for name in quad_name.split(" + ")]
        quad_avg_cost, quad_avg_time = combo_avg_cost_time(quad_indices)
        prev_combo = best_combo_indices(3)
        if prev_combo:
            prev_cost, prev_time = combo_avg_cost_time(prev_combo)
            prev_cost_delta = prev_cost - quad_avg_cost
            prev_time_delta = prev_time - quad_avg_time
        else:
            prev_cost_delta = float("nan")
            prev_time_delta = float("nan")
        move_3_to_2, move_2_to_1 = combo_time_improvements(quad_indices, prev_combo or [])
        st.caption(
            f"Avg cost: {quad_avg_cost:.2f} (delta {avg_cost - quad_avg_cost:.2f}) - "
            f"Avg time: {quad_avg_time:.2f} (delta {avg_time - quad_avg_time:.2f})"
        )
        st.caption(f"Delta vs best 3 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
        st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
        quad_improved = combo_improved_cities(prev_combo or [], quad_indices)
        st.selectbox(
            "Improved destinations vs best 3 origins (quad)",
            quad_improved or ["None"],
            key=f"{key_prefix}_quad_no1_day",
        )
        if show_day_percentages:
            pct_1, pct_2, pct_3 = combo_day_percentages(quad_indices)
            st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
            not_one_day = combo_not_covered(quad_indices, 1.0)
            not_two_day = combo_not_covered(quad_indices, 2.0)
            st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
            st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
    else:
        st.caption("Not enough origins to calculate quads.")

    st.markdown("Top 5 (5 locations)")
    five_list = (
        compute_top_k_combos_with_required_cached(
            df,
            tuple(origin_list_local),
            tuple(dest_in_view),
            required_local,
            5,
            build_cost_weight,
            dest_weights_items,
        ) if len(origin_list_local) >= 5 else []
    )
    common_origins = []
    if five_list:
        five_labels = [f"{name} ({value:.2f})" for name, value in five_list]
        selected_five = st.selectbox(
            "Top 5 (5 locations) (2-day priority)",
            five_labels,
            index=0,
            key=f"{key_prefix}_top5_fives",
        )
        five_name = selected_five.rsplit(" (", 1)[0]
        five_indices = [origin_list_local.index(name) for name in five_name.split(" + ")]
        five_avg_cost, five_avg_time = combo_avg_cost_time(five_indices)
        prev_combo = best_combo_indices(4)
        if prev_combo:
            prev_cost, prev_time = combo_avg_cost_time(prev_combo)
            prev_cost_delta = prev_cost - five_avg_cost
            prev_time_delta = prev_time - five_avg_time
        else:
            prev_cost_delta = float("nan")
            prev_time_delta = float("nan")
        move_3_to_2, move_2_to_1 = combo_time_improvements(five_indices, prev_combo or [])
        st.caption(
            f"Avg cost: {five_avg_cost:.2f} (delta {avg_cost - five_avg_cost:.2f}) - "
            f"Avg time: {five_avg_time:.2f} (delta {avg_time - five_avg_time:.2f})"
        )
        st.caption(f"Delta vs best 4 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
        st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
        five_origin_sets = [set(name.split(" + ")) for name, _ in five_list]
        common_origins = sorted(set.intersection(*five_origin_sets)) if five_origin_sets else []
        if common_origins:
            st.caption(f"Auto-included origins for 6/7: {', '.join(common_origins)}")
        five_improved = combo_improved_cities(prev_combo or [], five_indices)
        st.selectbox(
            "Improved destinations vs best 4 origins (5 locations)",
            five_improved or ["None"],
            key=f"{key_prefix}_five_no1_day",
        )
        if show_day_percentages:
            pct_1, pct_2, pct_3 = combo_day_percentages(five_indices)
            st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
            not_one_day = combo_not_covered(five_indices, 1.0)
            not_two_day = combo_not_covered(five_indices, 2.0)
            st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
            st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
    else:
        st.caption("Not enough origins to calculate 5-location combos.")

    st.markdown("Top 5 (6 locations)")
    required_origins = tuple(common_origins) if common_origins else tuple()
    if required_local and required_local[0] not in required_origins:
        required_origins = required_origins + required_local
    six_list = (
        compute_top_k_combos_with_required_cached(
            df,
            tuple(origin_list_local),
            tuple(dest_in_view),
            required_origins,
            6,
            build_cost_weight,
            dest_weights_items,
        ) if len(origin_list_local) >= 6 else []
    )
    common_origins6 = []
    if six_list:
        six_labels = [f"{name} ({value:.2f})" for name, value in six_list]
        selected_six = st.selectbox(
            "Top 5 (6 locations) (2-day priority)",
            six_labels,
            index=0,
            key=f"{key_prefix}_top5_sixes",
        )
        six_name = selected_six.rsplit(" (", 1)[0]
        six_indices = [origin_list_local.index(name) for name in six_name.split(" + ")]
        six_avg_cost, six_avg_time = combo_avg_cost_time(six_indices)
        prev_combo = best_combo_indices(5)
        if prev_combo:
            prev_cost, prev_time = combo_avg_cost_time(prev_combo)
            prev_cost_delta = prev_cost - six_avg_cost
            prev_time_delta = prev_time - six_avg_time
        else:
            prev_cost_delta = float("nan")
            prev_time_delta = float("nan")
        move_3_to_2, move_2_to_1 = combo_time_improvements(six_indices, prev_combo or [])
        st.caption(
            f"Avg cost: {six_avg_cost:.2f} (delta {avg_cost - six_avg_cost:.2f}) - "
            f"Avg time: {six_avg_time:.2f} (delta {avg_time - six_avg_time:.2f})"
        )
        st.caption(f"Delta vs best 5 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
        st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
        six_origin_sets = [set(name.split(" + ")) for name, _ in six_list]
        common_origins6 = sorted(set.intersection(*six_origin_sets)) if six_origin_sets else []
        if common_origins6:
            st.caption(f"Auto-included origins for 7/8: {', '.join(common_origins6)}")
        six_improved = combo_improved_cities(prev_combo or [], six_indices)
        st.selectbox(
            "Improved destinations vs best 5 origins (6 locations)",
            six_improved or ["None"],
            key=f"{key_prefix}_six_no1_day",
        )
        if show_day_percentages:
            pct_1, pct_2, pct_3 = combo_day_percentages(six_indices)
            st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
            not_one_day = combo_not_covered(six_indices, 1.0)
            not_two_day = combo_not_covered(six_indices, 2.0)
            st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
            st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
    else:
        st.caption("Not enough origins to calculate 6-location combos.")

    st.markdown("Top 5 (7 locations)")
    st.caption("From 7+ locations, candidates are ranked by the biggest increase in 2-day coverage (weighted total used as tiebreaker).")
    seven_list = (expand_from_best(six_list, 7) if len(origin_list_local) >= 7 else [])
    if seven_list:
        seven_labels = [f"{name} ({value:.2f})" for name, value in seven_list]
        selected_seven = st.selectbox(
            "Top 5 (7 locations) (2-day priority)",
            seven_labels,
            index=0,
            key=f"{key_prefix}_top5_sevens",
        )
        seven_name = selected_seven.rsplit(" (", 1)[0]
        seven_indices = [origin_list_local.index(name) for name in seven_name.split(" + ")]
        seven_avg_cost, seven_avg_time = combo_avg_cost_time(seven_indices)
        prev_combo = combo_indices_from_list(six_list)
        if prev_combo:
            prev_cost, prev_time = combo_avg_cost_time(prev_combo)
            prev_cost_delta = prev_cost - seven_avg_cost
            prev_time_delta = prev_time - seven_avg_time
        else:
            prev_cost_delta = float("nan")
            prev_time_delta = float("nan")
        move_3_to_2, move_2_to_1 = combo_time_improvements(seven_indices, prev_combo or [])
        st.caption(
            f"Avg cost: {seven_avg_cost:.2f} (delta {avg_cost - seven_avg_cost:.2f}) - "
            f"Avg time: {seven_avg_time:.2f} (delta {avg_time - seven_avg_time:.2f})"
        )
        st.caption(f"Delta vs best 6 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
        st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
        seven_improved = combo_improved_cities(prev_combo or [], seven_indices)
        st.selectbox(
            "Improved destinations vs best 6 origins (7 locations)",
            seven_improved or ["None"],
            key=f"{key_prefix}_seven_no1_day",
        )
        if show_day_percentages:
            pct_1, pct_2, pct_3 = combo_day_percentages(seven_indices)
            st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
            not_one_day = combo_not_covered(seven_indices, 1.0)
            not_two_day = combo_not_covered(seven_indices, 2.0)
            st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
            st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
    else:
        st.caption("Not enough origins to calculate 7-location combos.")

    st.markdown("Top 5 (8 locations)")
    eight_list = (expand_from_best(seven_list, 8) if len(origin_list_local) >= 8 else [])
    if eight_list:
        eight_labels = [f"{name} ({value:.2f})" for name, value in eight_list]
        selected_eight = st.selectbox(
            "Top 5 (8 locations) (2-day priority)",
            eight_labels,
            index=0,
            key=f"{key_prefix}_top5_eights",
        )
        eight_name = selected_eight.rsplit(" (", 1)[0]
        eight_indices = [origin_list_local.index(name) for name in eight_name.split(" + ")]
        eight_avg_cost, eight_avg_time = combo_avg_cost_time(eight_indices)
        prev_combo = combo_indices_from_list(seven_list)
        if prev_combo:
            prev_cost, prev_time = combo_avg_cost_time(prev_combo)
            prev_cost_delta = prev_cost - eight_avg_cost
            prev_time_delta = prev_time - eight_avg_time
        else:
            prev_cost_delta = float("nan")
            prev_time_delta = float("nan")
        move_3_to_2, move_2_to_1 = combo_time_improvements(eight_indices, prev_combo or [])
        st.caption(
            f"Avg cost: {eight_avg_cost:.2f} (delta {avg_cost - eight_avg_cost:.2f}) - "
            f"Avg time: {eight_avg_time:.2f} (delta {avg_time - eight_avg_time:.2f})"
        )
        st.caption(f"Delta vs best 7 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
        st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
        eight_improved = combo_improved_cities(prev_combo or [], eight_indices)
        st.selectbox(
            "Improved destinations vs best 7 origins (8 locations)",
            eight_improved or ["None"],
            key=f"{key_prefix}_eight_no1_day",
        )
        if show_day_percentages:
            pct_1, pct_2, pct_3 = combo_day_percentages(eight_indices)
            st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
            not_one_day = combo_not_covered(eight_indices, 1.0)
            not_two_day = combo_not_covered(eight_indices, 2.0)
            st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
            st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
    else:
        st.caption("Not enough origins to calculate 8-location combos.")

    st.markdown("Top 5 (9 locations)")
    nine_list = (expand_from_best(eight_list, 9) if len(origin_list_local) >= 9 else [])
    if nine_list:
        nine_labels = [f"{name} ({value:.2f})" for name, value in nine_list]
        selected_nine = st.selectbox(
            "Top 5 (9 locations) (2-day priority)",
            nine_labels,
            index=0,
            key=f"{key_prefix}_top5_nines",
        )
        nine_name = selected_nine.rsplit(" (", 1)[0]
        nine_indices = [origin_list_local.index(name) for name in nine_name.split(" + ")]
        nine_avg_cost, nine_avg_time = combo_avg_cost_time(nine_indices)
        prev_combo = combo_indices_from_list(eight_list)
        if prev_combo:
            prev_cost, prev_time = combo_avg_cost_time(prev_combo)
            prev_cost_delta = prev_cost - nine_avg_cost
            prev_time_delta = prev_time - nine_avg_time
        else:
            prev_cost_delta = float("nan")
            prev_time_delta = float("nan")
        move_3_to_2, move_2_to_1 = combo_time_improvements(nine_indices, prev_combo or [])
        st.caption(
            f"Avg cost: {nine_avg_cost:.2f} (delta {avg_cost - nine_avg_cost:.2f}) - "
            f"Avg time: {nine_avg_time:.2f} (delta {avg_time - nine_avg_time:.2f})"
        )
        st.caption(f"Delta vs best 8 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
        st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
        nine_improved = combo_improved_cities(prev_combo or [], nine_indices)
        st.selectbox(
            "Improved destinations vs best 8 origins (9 locations)",
            nine_improved or ["None"],
            key=f"{key_prefix}_nine_no1_day",
        )
        if show_day_percentages:
            pct_1, pct_2, pct_3 = combo_day_percentages(nine_indices)
            st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
            not_one_day = combo_not_covered(nine_indices, 1.0)
            not_two_day = combo_not_covered(nine_indices, 2.0)
            st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")
            st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
    else:
        st.caption("Not enough origins to calculate 9-location combos.")

    if max_k >= 10:
        st.markdown("Top 5 (10 locations)")
        ten_list = (expand_from_best(nine_list, 10) if len(origin_list_local) >= 10 else [])
        if ten_list:
            ten_labels = [f"{name} ({value:.2f})" for name, value in ten_list]
            selected_ten = st.selectbox(
                "Top 5 (10 locations) (2-day priority)",
                ten_labels,
                index=0,
                key=f"{key_prefix}_top5_tens",
            )
            ten_name = selected_ten.rsplit(" (", 1)[0]
            ten_indices = [origin_list_local.index(name) for name in ten_name.split(" + ")]
            ten_avg_cost, ten_avg_time = combo_avg_cost_time(ten_indices)
            prev_combo = combo_indices_from_list(nine_list)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - ten_avg_cost
                prev_time_delta = prev_time - ten_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(ten_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {ten_avg_cost:.2f} (delta {avg_cost - ten_avg_cost:.2f}) - "
                f"Avg time: {ten_avg_time:.2f} (delta {avg_time - ten_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 9 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            ten_improved = combo_improved_cities(prev_combo or [], ten_indices)
            st.selectbox(
                "Improved destinations vs best 9 origins (10 locations)",
                ten_improved or ["None"],
                key=f"{key_prefix}_ten_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2, pct_3 = combo_day_percentages(ten_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
                not_one_day = combo_not_covered(ten_indices, 1.0)

                not_two_day = combo_not_covered(ten_indices, 2.0)

                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")

                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 10-location combos.")

    if max_k >= 11:
        st.markdown("Top 5 (11 locations)")
        eleven_list = (expand_from_best(ten_list, 11) if len(origin_list_local) >= 11 else [])
        if eleven_list:
            eleven_labels = [f"{name} ({value:.2f})" for name, value in eleven_list]
            selected_eleven = st.selectbox(
                "Top 5 (11 locations) (2-day priority)",
                eleven_labels,
                index=0,
                key=f"{key_prefix}_top5_elevens",
            )
            eleven_name = selected_eleven.rsplit(" (", 1)[0]
            eleven_indices = [origin_list_local.index(name) for name in eleven_name.split(" + ")]
            eleven_avg_cost, eleven_avg_time = combo_avg_cost_time(eleven_indices)
            prev_combo = combo_indices_from_list(ten_list)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - eleven_avg_cost
                prev_time_delta = prev_time - eleven_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(eleven_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {eleven_avg_cost:.2f} (delta {avg_cost - eleven_avg_cost:.2f}) - "
                f"Avg time: {eleven_avg_time:.2f} (delta {avg_time - eleven_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 10 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            eleven_improved = combo_improved_cities(prev_combo or [], eleven_indices)
            st.selectbox(
                "Improved destinations vs best 10 origins (11 locations)",
                eleven_improved or ["None"],
                key=f"{key_prefix}_eleven_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2, pct_3 = combo_day_percentages(eleven_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
                not_one_day = combo_not_covered(eleven_indices, 1.0)

                not_two_day = combo_not_covered(eleven_indices, 2.0)

                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")

                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 11-location combos.")

    if max_k >= 12:
        st.markdown("Top 5 (12 locations)")
        twelve_list = (expand_from_best(eleven_list, 12) if len(origin_list_local) >= 12 else [])
        if twelve_list:
            twelve_labels = [f"{name} ({value:.2f})" for name, value in twelve_list]
            selected_twelve = st.selectbox(
                "Top 5 (12 locations) (2-day priority)",
                twelve_labels,
                index=0,
                key=f"{key_prefix}_top5_twelves",
            )
            twelve_name = selected_twelve.rsplit(" (", 1)[0]
            twelve_indices = [origin_list_local.index(name) for name in twelve_name.split(" + ")]
            twelve_avg_cost, twelve_avg_time = combo_avg_cost_time(twelve_indices)
            prev_combo = combo_indices_from_list(eleven_list)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - twelve_avg_cost
                prev_time_delta = prev_time - twelve_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(twelve_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {twelve_avg_cost:.2f} (delta {avg_cost - twelve_avg_cost:.2f}) - "
                f"Avg time: {twelve_avg_time:.2f} (delta {avg_time - twelve_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 11 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            twelve_improved = combo_improved_cities(prev_combo or [], twelve_indices)
            st.selectbox(
                "Improved destinations vs best 11 origins (12 locations)",
                twelve_improved or ["None"],
                key=f"{key_prefix}_twelve_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2, pct_3 = combo_day_percentages(twelve_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
                not_one_day = combo_not_covered(twelve_indices, 1.0)

                not_two_day = combo_not_covered(twelve_indices, 2.0)

                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")

                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 12-location combos.")

    if max_k >= 13:
        st.markdown("Top 5 (13 locations)")
        thirteen_list = (expand_from_best(twelve_list, 13) if len(origin_list_local) >= 13 else [])
        if thirteen_list:
            thirteen_labels = [f"{name} ({value:.2f})" for name, value in thirteen_list]
            selected_thirteen = st.selectbox(
                "Top 5 (13 locations) (2-day priority)",
                thirteen_labels,
                index=0,
                key=f"{key_prefix}_top5_thirteens",
            )
            thirteen_name = selected_thirteen.rsplit(" (", 1)[0]
            thirteen_indices = [origin_list_local.index(name) for name in thirteen_name.split(" + ")]
            thirteen_avg_cost, thirteen_avg_time = combo_avg_cost_time(thirteen_indices)
            prev_combo = combo_indices_from_list(twelve_list)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - thirteen_avg_cost
                prev_time_delta = prev_time - thirteen_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(thirteen_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {thirteen_avg_cost:.2f} (delta {avg_cost - thirteen_avg_cost:.2f}) - "
                f"Avg time: {thirteen_avg_time:.2f} (delta {avg_time - thirteen_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 12 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            thirteen_improved = combo_improved_cities(prev_combo or [], thirteen_indices)
            st.selectbox(
                "Improved destinations vs best 12 origins (13 locations)",
                thirteen_improved or ["None"],
                key=f"{key_prefix}_thirteen_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2, pct_3 = combo_day_percentages(thirteen_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
                not_one_day = combo_not_covered(thirteen_indices, 1.0)

                not_two_day = combo_not_covered(thirteen_indices, 2.0)

                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")

                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 13-location combos.")

    if max_k >= 14:
        st.markdown("Top 5 (14 locations)")
        fourteen_list = (expand_from_best(thirteen_list, 14) if len(origin_list_local) >= 14 else [])
        if fourteen_list:
            fourteen_labels = [f"{name} ({value:.2f})" for name, value in fourteen_list]
            selected_fourteen = st.selectbox(
                "Top 5 (14 locations) (2-day priority)",
                fourteen_labels,
                index=0,
                key=f"{key_prefix}_top5_fourteens",
            )
            fourteen_name = selected_fourteen.rsplit(" (", 1)[0]
            fourteen_indices = [origin_list_local.index(name) for name in fourteen_name.split(" + ")]
            fourteen_avg_cost, fourteen_avg_time = combo_avg_cost_time(fourteen_indices)
            prev_combo = combo_indices_from_list(thirteen_list)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - fourteen_avg_cost
                prev_time_delta = prev_time - fourteen_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(fourteen_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {fourteen_avg_cost:.2f} (delta {avg_cost - fourteen_avg_cost:.2f}) - "
                f"Avg time: {fourteen_avg_time:.2f} (delta {avg_time - fourteen_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 13 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            fourteen_improved = combo_improved_cities(prev_combo or [], fourteen_indices)
            st.selectbox(
                "Improved destinations vs best 13 origins (14 locations)",
                fourteen_improved or ["None"],
                key=f"{key_prefix}_fourteen_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2, pct_3 = combo_day_percentages(fourteen_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
                not_one_day = combo_not_covered(fourteen_indices, 1.0)

                not_two_day = combo_not_covered(fourteen_indices, 2.0)

                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")

                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 14-location combos.")

    if max_k >= 15:
        st.markdown("Top 5 (15 locations)")
        fifteen_list = (expand_from_best(fourteen_list, 15) if len(origin_list_local) >= 15 else [])
        if fifteen_list:
            fifteen_labels = [f"{name} ({value:.2f})" for name, value in fifteen_list]
            selected_fifteen = st.selectbox(
                "Top 5 (15 locations) (2-day priority)",
                fifteen_labels,
                index=0,
                key=f"{key_prefix}_top5_fifteens",
            )
            fifteen_name = selected_fifteen.rsplit(" (", 1)[0]
            fifteen_indices = [origin_list_local.index(name) for name in fifteen_name.split(" + ")]
            fifteen_avg_cost, fifteen_avg_time = combo_avg_cost_time(fifteen_indices)
            prev_combo = combo_indices_from_list(fourteen_list)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - fifteen_avg_cost
                prev_time_delta = prev_time - fifteen_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(fifteen_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {fifteen_avg_cost:.2f} (delta {avg_cost - fifteen_avg_cost:.2f}) - "
                f"Avg time: {fifteen_avg_time:.2f} (delta {avg_time - fifteen_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 14 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            fifteen_improved = combo_improved_cities(prev_combo or [], fifteen_indices)
            st.selectbox(
                "Improved destinations vs best 14 origins (15 locations)",
                fifteen_improved or ["None"],
                key=f"{key_prefix}_fifteen_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2, pct_3 = combo_day_percentages(fifteen_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
                not_one_day = combo_not_covered(fifteen_indices, 1.0)

                not_two_day = combo_not_covered(fifteen_indices, 2.0)

                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")

                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 15-location combos.")

    if max_k >= 16:
        st.markdown("Top 5 (16 locations)")
        sixteen_list = (expand_from_best(fifteen_list, 16) if len(origin_list_local) >= 16 else [])
        if sixteen_list:
            sixteen_labels = [f"{name} ({value:.2f})" for name, value in sixteen_list]
            selected_sixteen = st.selectbox(
                "Top 5 (16 locations) (2-day priority)",
                sixteen_labels,
                index=0,
                key=f"{key_prefix}_top5_sixteens",
            )
            sixteen_name = selected_sixteen.rsplit(" (", 1)[0]
            sixteen_indices = [origin_list_local.index(name) for name in sixteen_name.split(" + ")]
            sixteen_avg_cost, sixteen_avg_time = combo_avg_cost_time(sixteen_indices)
            prev_combo = combo_indices_from_list(fifteen_list)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - sixteen_avg_cost
                prev_time_delta = prev_time - sixteen_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(sixteen_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {sixteen_avg_cost:.2f} (delta {avg_cost - sixteen_avg_cost:.2f}) - "
                f"Avg time: {sixteen_avg_time:.2f} (delta {avg_time - sixteen_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 15 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            sixteen_improved = combo_improved_cities(prev_combo or [], sixteen_indices)
            st.selectbox(
                "Improved destinations vs best 15 origins (16 locations)",
                sixteen_improved or ["None"],
                key=f"{key_prefix}_sixteen_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2, pct_3 = combo_day_percentages(sixteen_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
                not_one_day = combo_not_covered(sixteen_indices, 1.0)

                not_two_day = combo_not_covered(sixteen_indices, 2.0)

                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")

                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 16-location combos.")

    if max_k >= 17:
        st.markdown("Top 5 (17 locations)")
        seventeen_list = (expand_from_best(sixteen_list, 17) if len(origin_list_local) >= 17 else [])
        if seventeen_list:
            seventeen_labels = [f"{name} ({value:.2f})" for name, value in seventeen_list]
            selected_seventeen = st.selectbox(
                "Top 5 (17 locations) (2-day priority)",
                seventeen_labels,
                index=0,
                key=f"{key_prefix}_top5_seventeens",
            )
            seventeen_name = selected_seventeen.rsplit(" (", 1)[0]
            seventeen_indices = [origin_list_local.index(name) for name in seventeen_name.split(" + ")]
            seventeen_avg_cost, seventeen_avg_time = combo_avg_cost_time(seventeen_indices)
            prev_combo = combo_indices_from_list(sixteen_list)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - seventeen_avg_cost
                prev_time_delta = prev_time - seventeen_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(seventeen_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {seventeen_avg_cost:.2f} (delta {avg_cost - seventeen_avg_cost:.2f}) - "
                f"Avg time: {seventeen_avg_time:.2f} (delta {avg_time - seventeen_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 16 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            seventeen_improved = combo_improved_cities(prev_combo or [], seventeen_indices)
            st.selectbox(
                "Improved destinations vs best 16 origins (17 locations)",
                seventeen_improved or ["None"],
                key=f"{key_prefix}_seventeen_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2, pct_3 = combo_day_percentages(seventeen_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
                not_one_day = combo_not_covered(seventeen_indices, 1.0)

                not_two_day = combo_not_covered(seventeen_indices, 2.0)

                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")

                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 17-location combos.")

    if max_k >= 18:
        st.markdown("Top 5 (18 locations)")
        eighteen_list = (expand_from_best(seventeen_list, 18) if len(origin_list_local) >= 18 else [])
        if eighteen_list:
            eighteen_labels = [f"{name} ({value:.2f})" for name, value in eighteen_list]
            selected_eighteen = st.selectbox(
                "Top 5 (18 locations) (2-day priority)",
                eighteen_labels,
                index=0,
                key=f"{key_prefix}_top5_eighteens",
            )
            eighteen_name = selected_eighteen.rsplit(" (", 1)[0]
            eighteen_indices = [origin_list_local.index(name) for name in eighteen_name.split(" + ")]
            eighteen_avg_cost, eighteen_avg_time = combo_avg_cost_time(eighteen_indices)
            prev_combo = combo_indices_from_list(seventeen_list)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - eighteen_avg_cost
                prev_time_delta = prev_time - eighteen_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(eighteen_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {eighteen_avg_cost:.2f} (delta {avg_cost - eighteen_avg_cost:.2f}) - "
                f"Avg time: {eighteen_avg_time:.2f} (delta {avg_time - eighteen_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 17 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            eighteen_improved = combo_improved_cities(prev_combo or [], eighteen_indices)
            st.selectbox(
                "Improved destinations vs best 17 origins (18 locations)",
                eighteen_improved or ["None"],
                key=f"{key_prefix}_eighteen_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2, pct_3 = combo_day_percentages(eighteen_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
                not_one_day = combo_not_covered(eighteen_indices, 1.0)

                not_two_day = combo_not_covered(eighteen_indices, 2.0)

                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")

                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 18-location combos.")

    if max_k >= 19:
        st.markdown("Top 5 (19 locations)")
        nineteen_list = (expand_from_best(eighteen_list, 19) if len(origin_list_local) >= 19 else [])
        if nineteen_list:
            nineteen_labels = [f"{name} ({value:.2f})" for name, value in nineteen_list]
            selected_nineteen = st.selectbox(
                "Top 5 (19 locations) (2-day priority)",
                nineteen_labels,
                index=0,
                key=f"{key_prefix}_top5_nineteens",
            )
            nineteen_name = selected_nineteen.rsplit(" (", 1)[0]
            nineteen_indices = [origin_list_local.index(name) for name in nineteen_name.split(" + ")]
            nineteen_avg_cost, nineteen_avg_time = combo_avg_cost_time(nineteen_indices)
            prev_combo = combo_indices_from_list(eighteen_list)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - nineteen_avg_cost
                prev_time_delta = prev_time - nineteen_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(nineteen_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {nineteen_avg_cost:.2f} (delta {avg_cost - nineteen_avg_cost:.2f}) - "
                f"Avg time: {nineteen_avg_time:.2f} (delta {avg_time - nineteen_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 18 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            nineteen_improved = combo_improved_cities(prev_combo or [], nineteen_indices)
            st.selectbox(
                "Improved destinations vs best 18 origins (19 locations)",
                nineteen_improved or ["None"],
                key=f"{key_prefix}_nineteen_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2, pct_3 = combo_day_percentages(nineteen_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
                not_one_day = combo_not_covered(nineteen_indices, 1.0)

                not_two_day = combo_not_covered(nineteen_indices, 2.0)

                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")

                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 19-location combos.")

    if max_k >= 20:
        st.markdown("Top 5 (20 locations)")
        twenty_list = (expand_from_best(nineteen_list, 20) if len(origin_list_local) >= 20 else [])
        if twenty_list:
            twenty_labels = [f"{name} ({value:.2f})" for name, value in twenty_list]
            selected_twenty = st.selectbox(
                "Top 5 (20 locations) (2-day priority)",
                twenty_labels,
                index=0,
                key=f"{key_prefix}_top5_twenties",
            )
            twenty_name = selected_twenty.rsplit(" (", 1)[0]
            twenty_indices = [origin_list_local.index(name) for name in twenty_name.split(" + ")]
            twenty_avg_cost, twenty_avg_time = combo_avg_cost_time(twenty_indices)
            prev_combo = combo_indices_from_list(nineteen_list)
            if prev_combo:
                prev_cost, prev_time = combo_avg_cost_time(prev_combo)
                prev_cost_delta = prev_cost - twenty_avg_cost
                prev_time_delta = prev_time - twenty_avg_time
            else:
                prev_cost_delta = float("nan")
                prev_time_delta = float("nan")
            move_3_to_2, move_2_to_1 = combo_time_improvements(twenty_indices, prev_combo or [])
            st.caption(
                f"Avg cost: {twenty_avg_cost:.2f} (delta {avg_cost - twenty_avg_cost:.2f}) - "
                f"Avg time: {twenty_avg_time:.2f} (delta {avg_time - twenty_avg_time:.2f})"
            )
            st.caption(f"Delta vs best 19 origins: cost {prev_cost_delta:.2f}, time {prev_time_delta:.2f}")
            st.caption(f"Moves 3->2 day: {move_3_to_2} - Moves 2->1 day: {move_2_to_1}")
            twenty_improved = combo_improved_cities(prev_combo or [], twenty_indices)
            st.selectbox(
                "Improved destinations vs best 19 origins (20 locations)",
                twenty_improved or ["None"],
                key=f"{key_prefix}_twenty_no1_day",
            )
            if show_day_percentages:
                pct_1, pct_2, pct_3 = combo_day_percentages(twenty_indices)
                st.caption(f"1-day coverage: {pct_1:.1f}% | 2-day coverage: {pct_2:.1f}% | 3-day coverage: {pct_3:.1f}%")
                not_one_day = combo_not_covered(twenty_indices, 1.0)

                not_two_day = combo_not_covered(twenty_indices, 2.0)

                st.caption(f"Not covered in 1 day: {format_not_covered(not_one_day)}")

                st.caption(f"Not covered in 2 days: {format_not_covered(not_two_day)}")
        else:
            st.caption("Not enough origins to calculate 20-location combos.")


@st.cache_data
def compute_pair_df_cached(
    df: pd.DataFrame,
    origin_list: tuple,
    dest_list: tuple,
    build_cost_weight: float,
    dest_weights_items: tuple,
    major_weight_items: tuple,
) -> pd.DataFrame:
    dest_weights = dict(dest_weights_items)
    major_weight_map = dict(major_weight_items)
    total_major_weight = sum(major_weight_map.values())

    cost_mat, time_mat = build_origin_destination_matrices(df, origin_list, dest_list)
    weights_vec = np.array([dest_weights.get(dest, 0.0) for dest in dest_list], dtype=float)
    if weights_vec.sum() == 0:
        weights_vec = np.ones(len(dest_list), dtype=float)

    major_dest_indices = [i for i, dest in enumerate(dest_list) if dest in major_weight_map]
    pair_rows = []
    for i, j in combinations(range(len(origin_list)), 2):
        best_cost = np.nanmin(cost_mat[[i, j], :], axis=0)
        best_time = np.nanmin(time_mat[[i, j], :], axis=0)
        valid_mask = ~np.isnan(best_cost) & ~np.isnan(best_time)
        valid_time = np.isfinite(best_time)
        if valid_mask.any():
            weights = weights_vec[valid_mask]
            if weights.sum() == 0:
                weights = np.ones(len(weights), dtype=float)
            weighted = build_cost_weight * best_cost[valid_mask] + (1 - build_cost_weight) * best_time[valid_mask]
            weighted_total = float(np.sum(weighted * (weights / weights.sum())))
        else:
            weighted_total = float("nan")
        if valid_time.any():
            valid_weights = weights_vec[valid_time]
            if valid_weights.sum() == 0:
                valid_weights = np.ones(len(valid_weights), dtype=float)
            total_weight = float(np.sum(valid_weights))
            two_day_weight = float(np.sum(valid_weights[best_time[valid_time] <= 2.0]))
            two_day_coverage_pct = two_day_weight / total_weight * 100.0
        else:
            two_day_coverage_pct = 0.0

        pair_cost = float(np.nansum(best_cost))
        pair_time = float(np.nansum(best_time))

        if major_weight_map and total_major_weight:
            coverage_weight = 0.0
            for idx in major_dest_indices:
                if not np.isnan(best_time[idx]) and best_time[idx] <= 1.0:
                    coverage_weight += major_weight_map.get(dest_list[idx], 0.0)
            major_coverage_pct = coverage_weight / total_major_weight * 100.0
        else:
            major_coverage_pct = 0.0

        pair_rows.append({
            "Pair": f"{origin_list[i]} + {origin_list[j]}",
            "WeightedTotal": weighted_total,
            "TwoDayCoveragePct": two_day_coverage_pct,
            "PairCost": pair_cost,
            "PairTime": pair_time,
            "MajorCoveragePct": major_coverage_pct,
        })
    return pd.DataFrame(pair_rows).sort_values(
        by=["TwoDayCoveragePct", "WeightedTotal", "Pair"],
        ascending=[False, True, True],
    )


@st.cache_data
def compute_top_k_combos_cached(
    df: pd.DataFrame,
    origin_list: tuple,
    dest_list: tuple,
    k: int,
    build_cost_weight: float,
    dest_weights_items: tuple,
    limit: int = 5,
) -> list[tuple[str, float]]:
    dest_weights = dict(dest_weights_items)
    cost_mat, time_mat = build_origin_destination_matrices(df, origin_list, dest_list)
    weights_vec = np.array([dest_weights.get(dest, 0.0) for dest in dest_list], dtype=float)
    if weights_vec.sum() == 0:
        weights_vec = np.ones(len(dest_list), dtype=float)

    combos = []
    for combo in combinations(range(len(origin_list)), k):
        best_cost = np.nanmin(cost_mat[list(combo), :], axis=0)
        best_time = np.nanmin(time_mat[list(combo), :], axis=0)
        valid_mask = ~np.isnan(best_cost) & ~np.isnan(best_time)
        valid_time = np.isfinite(best_time)
        if valid_mask.any():
            weights = weights_vec[valid_mask]
            if weights.sum() == 0:
                weights = np.ones(len(weights), dtype=float)
            weighted = build_cost_weight * best_cost[valid_mask] + (1 - build_cost_weight) * best_time[valid_mask]
            total = float(np.sum(weighted * (weights / weights.sum())))
        else:
            total = float("nan")
        if valid_time.any():
            valid_weights = weights_vec[valid_time]
            if valid_weights.sum() == 0:
                valid_weights = np.ones(len(valid_weights), dtype=float)
            total_weight = float(np.sum(valid_weights))
            two_day_weight = float(np.sum(valid_weights[best_time[valid_time] <= 2.0]))
            two_day_coverage_pct = two_day_weight / total_weight * 100.0
        else:
            two_day_coverage_pct = 0.0
        combos.append((" + ".join(origin_list[idx] for idx in combo), total, two_day_coverage_pct))
    combos.sort(key=lambda item: (-item[2], item[1] if np.isfinite(item[1]) else float("inf"), item[0]))
    if len(combos) <= limit:
        return [(name, total) for name, total, _ in combos]
    cutoff_two_day_pct = combos[limit - 1][2]
    cutoff_weighted_total = combos[limit - 1][1]
    if not np.isfinite(cutoff_weighted_total):
        return [(name, total) for name, total, _ in combos[:limit]]
    return [
        (name, total)
        for name, total, two_day_pct in combos
        if two_day_pct > cutoff_two_day_pct
        or (np.isclose(two_day_pct, cutoff_two_day_pct) and total <= cutoff_weighted_total)
    ]


@st.cache_data
def compute_top_k_combos_with_required_cached(
    df: pd.DataFrame,
    origin_list: tuple,
    dest_list: tuple,
    required_origins: tuple,
    k: int,
    build_cost_weight: float,
    dest_weights_items: tuple,
    limit: int = 5,
) -> list[tuple[str, float]]:
    if k < len(required_origins):
        return []
    dest_weights = dict(dest_weights_items)
    cost_mat, time_mat = build_origin_destination_matrices(df, origin_list, dest_list)
    weights_vec = np.array([dest_weights.get(dest, 0.0) for dest in dest_list], dtype=float)
    if weights_vec.sum() == 0:
        weights_vec = np.ones(len(dest_list), dtype=float)

    origin_index = {origin: idx for idx, origin in enumerate(origin_list)}
    required_indices = [origin_index[o] for o in required_origins if o in origin_index]
    remaining_indices = [i for i in range(len(origin_list)) if i not in required_indices]
    choose_n = k - len(required_indices)

    combos = []
    for combo in combinations(remaining_indices, choose_n):
        combo_indices = list(required_indices) + list(combo)
        best_cost = np.nanmin(cost_mat[combo_indices, :], axis=0)
        best_time = np.nanmin(time_mat[combo_indices, :], axis=0)
        valid_mask = ~np.isnan(best_cost) & ~np.isnan(best_time)
        valid_time = np.isfinite(best_time)
        if valid_mask.any():
            weights = weights_vec[valid_mask]
            if weights.sum() == 0:
                weights = np.ones(len(weights), dtype=float)
            weighted = build_cost_weight * best_cost[valid_mask] + (1 - build_cost_weight) * best_time[valid_mask]
            total = float(np.sum(weighted * (weights / weights.sum())))
        else:
            total = float("nan")
        if valid_time.any():
            valid_weights = weights_vec[valid_time]
            if valid_weights.sum() == 0:
                valid_weights = np.ones(len(valid_weights), dtype=float)
            total_weight = float(np.sum(valid_weights))
            two_day_weight = float(np.sum(valid_weights[best_time[valid_time] <= 2.0]))
            two_day_coverage_pct = two_day_weight / total_weight * 100.0
        else:
            two_day_coverage_pct = 0.0
        combos.append((" + ".join(origin_list[idx] for idx in combo_indices), total, two_day_coverage_pct))
    combos.sort(key=lambda item: (-item[2], item[1] if np.isfinite(item[1]) else float("inf"), item[0]))
    if len(combos) <= limit:
        return [(name, total) for name, total, _ in combos]
    cutoff_two_day_pct = combos[limit - 1][2]
    cutoff_weighted_total = combos[limit - 1][1]
    if not np.isfinite(cutoff_weighted_total):
        return [(name, total) for name, total, _ in combos[:limit]]
    return [
        (name, total)
        for name, total, two_day_pct in combos
        if two_day_pct > cutoff_two_day_pct
        or (np.isclose(two_day_pct, cutoff_two_day_pct) and total <= cutoff_weighted_total)
    ]


@st.cache_data
def build_origin_destination_matrices(
    df: pd.DataFrame,
    origin_list: tuple,
    dest_list: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    pl_df = pl.from_pandas(df)
    grouped = pl_df.group_by(["FromAddress", "Destination"]).agg(
        pl.col("Cost").min().alias("MinCost"),
        pl.col("ShippingTimeDays").min().alias("MinTime"),
    )

    origin_index = {origin: idx for idx, origin in enumerate(origin_list)}
    dest_index = {dest: idx for idx, dest in enumerate(dest_list)}
    cost_mat = np.full((len(origin_list), len(dest_list)), np.nan, dtype=float)
    time_mat = np.full((len(origin_list), len(dest_list)), np.nan, dtype=float)
    for row in grouped.iter_rows(named=True):
        o_idx = origin_index.get(row["FromAddress"])
        d_idx = dest_index.get(row["Destination"])
        if o_idx is None or d_idx is None:
            continue
        cost_mat[o_idx, d_idx] = row["MinCost"]
        time_mat[o_idx, d_idx] = row["MinTime"]
    return cost_mat, time_mat


@st.cache_data
def compute_network_cost_stats(
    df: pd.DataFrame,
    network_origins: tuple,
    comparison_origin: str,
    weight_map: dict,
) -> dict:
    if not network_origins or not comparison_origin:
        return {
            "destination_count": 0,
            "network_avg_cost": float("nan"),
            "comparison_avg_cost": float("nan"),
            "savings_per_sign": float("nan"),
        }

    network_subset = df[df["FromAddress"].isin(network_origins)]
    comparison_subset = df[df["FromAddress"] == comparison_origin]
    if network_subset.empty or comparison_subset.empty:
        return {
            "destination_count": 0,
            "network_avg_cost": float("nan"),
            "comparison_avg_cost": float("nan"),
            "savings_per_sign": float("nan"),
        }

    network_by_dest = network_subset.groupby("Destination", as_index=False).agg(
        NetworkCost=("Cost", "min")
    )
    comparison_by_dest = comparison_subset.groupby("Destination", as_index=False).agg(
        ComparisonCost=("Cost", "mean")
    )
    merged = network_by_dest.merge(comparison_by_dest, on="Destination", how="inner")
    if merged.empty:
        return {
            "destination_count": 0,
            "network_avg_cost": float("nan"),
            "comparison_avg_cost": float("nan"),
            "savings_per_sign": float("nan"),
        }

    weight_lookup = destination_weights(merged["Destination"].tolist(), weight_map)
    weights = merged["Destination"].map(weight_lookup).fillna(0.0).to_numpy()
    weight_total = float(np.sum(weights))
    if weight_total == 0:
        weights = np.ones(len(merged))
        weight_total = float(len(merged))

    merged["SavingsPerSign"] = merged["ComparisonCost"] - merged["NetworkCost"]
    network_avg_cost = float(np.sum(merged["NetworkCost"].to_numpy() * weights) / weight_total)
    comparison_avg_cost = float(np.sum(merged["ComparisonCost"].to_numpy() * weights) / weight_total)
    return {
        "destination_count": int(len(merged)),
        "network_avg_cost": network_avg_cost,
        "comparison_avg_cost": comparison_avg_cost,
        "savings_per_sign": comparison_avg_cost - network_avg_cost,
    }


def weighted_origin_stats(savings_df: pd.DataFrame, dest_weights: dict) -> pd.DataFrame:
    if savings_df.empty:
        return pd.DataFrame(
            columns=[
                "FromAddress",
                "MeanSavingsPerSign",
                "MedianSavingsPerSign",
                "MeanTimeSavingsPerPlace",
                "MedianTimeSavingsPerPlace",
            ]
        )
    stats_df = savings_df.copy()
    stats_df["Weight"] = stats_df["Destination"].map(dest_weights).fillna(0.0)
    stats_df["WeightedSavings"] = stats_df["SavingsPerSign"] * stats_df["Weight"]
    stats_df["WeightedTime"] = stats_df["TimeSavingsPerPlace"] * stats_df["Weight"]

    means = stats_df.groupby("FromAddress", as_index=False).agg(
        WeightSum=("Weight", "sum"),
        WeightedSavingsSum=("WeightedSavings", "sum"),
        WeightedTimeSum=("WeightedTime", "sum"),
        FallbackMeanSavings=("SavingsPerSign", "mean"),
        FallbackMeanTime=("TimeSavingsPerPlace", "mean"),
    )
    means["MeanSavingsPerSign"] = np.where(
        means["WeightSum"] > 0,
        means["WeightedSavingsSum"] / means["WeightSum"],
        means["FallbackMeanSavings"],
    )
    means["MeanTimeSavingsPerPlace"] = np.where(
        means["WeightSum"] > 0,
        means["WeightedTimeSum"] / means["WeightSum"],
        means["FallbackMeanTime"],
    )

    median_rows = []
    for origin, group in stats_df.groupby("FromAddress", sort=False):
        weights = group["Weight"].to_numpy()
        if weights.sum() == 0:
            weights = np.ones(len(group))
        median_rows.append(
            {
                "FromAddress": origin,
                "MedianSavingsPerSign": weighted_median(
                    group["SavingsPerSign"].to_numpy(),
                    weights,
                ),
                "MedianTimeSavingsPerPlace": weighted_median(
                    group["TimeSavingsPerPlace"].to_numpy(),
                    weights,
                ),
            }
        )
    medians = pd.DataFrame(median_rows)

    merged = means.merge(medians, on="FromAddress", how="left")
    return merged[
        [
            "FromAddress",
            "MeanSavingsPerSign",
            "MedianSavingsPerSign",
            "MeanTimeSavingsPerPlace",
            "MedianTimeSavingsPerPlace",
        ]
    ]


def best_total(df: pd.DataFrame, origins, metric: str) -> float:
    subset = df[df["FromAddress"].isin(origins)]
    if subset.empty:
        return float("nan")
    return subset.groupby("Destination")[metric].min().sum()


def best_total_weighted(df: pd.DataFrame, origins, cost_weight: float, dest_weights: dict) -> float:
    subset = df[df["FromAddress"].isin(origins)]
    if subset.empty:
        return float("nan")
    best = subset.groupby("Destination", as_index=False).agg(
        BestCost=("Cost", "min"),
        BestTime=("ShippingTimeDays", "min"),
    )
    best["Weighted"] = cost_weight * best["BestCost"] + (1 - cost_weight) * best["BestTime"]
    weights = best["Destination"].map(dest_weights).fillna(0.0).to_numpy()
    if weights.sum() == 0:
        weights = np.ones(len(best))
    return float(np.sum(best["Weighted"].to_numpy() * (weights / weights.sum())))


def greedy_diminishing_returns(df: pd.DataFrame, baseline_origin: str, metric: str) -> pd.DataFrame:
    origins = sorted(df["FromAddress"].unique())
    if baseline_origin not in origins:
        return pd.DataFrame()
    remaining = [o for o in origins if o != baseline_origin]
    selected = [baseline_origin]
    baseline_total = best_total(df, selected, metric)
    current_total = baseline_total
    rows = [{
        "ShopCount": 1,
        "AddedOrigin": baseline_origin,
        "IncrementalSavings": 0.0,
        "CumulativeSavings": 0.0,
    }]
    # Greedy add: pick the origin that maximizes improvement each step.
    while remaining:
        best_origin = None
        best_improvement = -np.inf
        best_next_total = None
        for origin in remaining:
            total = best_total(df, selected + [origin], metric)
            improvement = current_total - total
            if improvement > best_improvement:
                best_improvement = improvement
                best_origin = origin
                best_next_total = total
        if best_origin is None:
            break
        selected.append(best_origin)
        remaining.remove(best_origin)
        current_total = best_next_total
        rows.append({
            "ShopCount": len(selected),
            "AddedOrigin": best_origin,
            "IncrementalSavings": best_improvement,
            "CumulativeSavings": baseline_total - current_total,
        })
    return pd.DataFrame(rows)


def savings_vs_baseline(df: pd.DataFrame, baseline_origin: str) -> pd.DataFrame:
    baseline = df[df["FromAddress"] == baseline_origin]
    baseline_dest = baseline.groupby("Destination", as_index=False).agg(BaselineCost=("Cost", "mean"))
    baseline_time = baseline.groupby("Destination", as_index=False).agg(BaselineTime=("ShippingTimeDays", "mean"))
    other = df.merge(baseline_dest, on="Destination", how="left").merge(baseline_time, on="Destination", how="left")
    other["SavingsPerSign"] = other["BaselineCost"] - other["Cost"]
    other["TimeSavingsPerPlace"] = other["BaselineTime"] - other["ShippingTimeDays"]
    return other


def greedy_savings_per_sign(
    df: pd.DataFrame,
    baseline_origin: str,
    build_cost_weight: float,
    dest_weights: dict,
) -> pd.DataFrame:
    origins = sorted(df["FromAddress"].unique())
    if baseline_origin not in origins:
        return pd.DataFrame()
    remaining = [o for o in origins if o != baseline_origin]
    selected = [baseline_origin]
    baseline_total = best_total_weighted(df, selected, build_cost_weight, dest_weights)
    current_total = baseline_total
    rows = [{
        "ShopCount": 1,
        "AddedOrigin": baseline_origin,
        "IncrementalSavingsPerSign_Mean": 0.0,
        "IncrementalSavingsPerSign_Median": 0.0,
        "IncrementalTimeSavingsPerPlace_Mean": 0.0,
        "IncrementalTimeSavingsPerPlace_Median": 0.0,
        "IncrementalWeightedSavings_Mean": 0.0,
        "IncrementalWeightedSavings_Median": 0.0,
        "CumulativeSavingsPerSign_Mean": 0.0,
        "CumulativeSavingsPerSign_Median": 0.0,
        "CumulativeTimeSavingsPerPlace_Mean": 0.0,
        "CumulativeTimeSavingsPerPlace_Median": 0.0,
        "CumulativeWeightedSavings_Mean": 0.0,
        "CumulativeWeightedSavings_Median": 0.0,
    }]
    while remaining:
        best_origin = None
        best_improvement = -np.inf
        for origin in remaining:
            total = best_total_weighted(df, selected + [origin], build_cost_weight, dest_weights)
            improvement = current_total - total
            if improvement > best_improvement:
                best_improvement = improvement
                best_origin = origin
        if best_origin is None:
            break
        selected.append(best_origin)
        remaining.remove(best_origin)
        current_total = best_total_weighted(df, selected, build_cost_weight, dest_weights)
        baseline = df[df["FromAddress"] == baseline_origin].groupby("Destination", as_index=False).agg(
            BaselineCost=("Cost", "mean"),
            BaselineTime=("ShippingTimeDays", "mean"),
        )
        best = df[df["FromAddress"].isin(selected)].groupby("Destination", as_index=False).agg(
            BestCost=("Cost", "min"),
            BestTime=("ShippingTimeDays", "min"),
        )
        merged = best.merge(baseline, on="Destination", how="left")
        merged["SavingsPerSign"] = merged["BaselineCost"] - merged["BestCost"]
        merged["TimeSavingsPerPlace"] = merged["BaselineTime"] - merged["BestTime"]
        merged["CostSavingsNorm"] = minmax(merged["SavingsPerSign"])
        merged["TimeSavingsNorm"] = minmax(merged["TimeSavingsPerPlace"])
        merged["WeightedSavings"] = (
            build_cost_weight * merged["CostSavingsNorm"] + (1 - build_cost_weight) * merged["TimeSavingsNorm"]
        )
        weights = merged["Destination"].map(dest_weights).fillna(0.0).to_numpy()
        if weights.sum() == 0:
            weights = np.ones(len(merged))
        mean_savings = weighted_mean(merged["SavingsPerSign"].to_numpy(), weights)
        median_savings = weighted_median(merged["SavingsPerSign"].to_numpy(), weights)
        mean_time = weighted_mean(merged["TimeSavingsPerPlace"].to_numpy(), weights)
        median_time = weighted_median(merged["TimeSavingsPerPlace"].to_numpy(), weights)
        mean_weighted = weighted_mean(merged["WeightedSavings"].to_numpy(), weights)
        median_weighted = weighted_median(merged["WeightedSavings"].to_numpy(), weights)
        prev_best = df[df["FromAddress"].isin(selected[:-1])].groupby("Destination", as_index=False).agg(
            BestCost=("Cost", "min"),
            BestTime=("ShippingTimeDays", "min"),
        )
        prev_merged = prev_best.merge(baseline, on="Destination", how="left")
        prev_merged["SavingsPerSign"] = prev_merged["BaselineCost"] - prev_merged["BestCost"]
        prev_merged["TimeSavingsPerPlace"] = prev_merged["BaselineTime"] - prev_merged["BestTime"]
        prev_merged["CostSavingsNorm"] = minmax(prev_merged["SavingsPerSign"])
        prev_merged["TimeSavingsNorm"] = minmax(prev_merged["TimeSavingsPerPlace"])
        prev_merged["WeightedSavings"] = (
            build_cost_weight * prev_merged["CostSavingsNorm"] + (1 - build_cost_weight) * prev_merged["TimeSavingsNorm"]
        )
        prev_weights = prev_merged["Destination"].map(dest_weights).fillna(0.0).to_numpy()
        if prev_weights.sum() == 0:
            prev_weights = np.ones(len(prev_merged))
        prev_mean = weighted_mean(prev_merged["SavingsPerSign"].to_numpy(), prev_weights)
        prev_median = weighted_median(prev_merged["SavingsPerSign"].to_numpy(), prev_weights)
        prev_mean_time = weighted_mean(prev_merged["TimeSavingsPerPlace"].to_numpy(), prev_weights)
        prev_median_time = weighted_median(prev_merged["TimeSavingsPerPlace"].to_numpy(), prev_weights)
        prev_mean_weighted = weighted_mean(prev_merged["WeightedSavings"].to_numpy(), prev_weights)
        prev_median_weighted = weighted_median(prev_merged["WeightedSavings"].to_numpy(), prev_weights)
        rows.append({
            "ShopCount": len(selected),
            "AddedOrigin": best_origin,
            "IncrementalSavingsPerSign_Mean": mean_savings - prev_mean,
            "IncrementalSavingsPerSign_Median": median_savings - prev_median,
            "IncrementalTimeSavingsPerPlace_Mean": mean_time - prev_mean_time,
            "IncrementalTimeSavingsPerPlace_Median": median_time - prev_median_time,
            "IncrementalWeightedSavings_Mean": mean_weighted - prev_mean_weighted,
            "IncrementalWeightedSavings_Median": median_weighted - prev_median_weighted,
            "CumulativeSavingsPerSign_Mean": mean_savings,
            "CumulativeSavingsPerSign_Median": median_savings,
            "CumulativeTimeSavingsPerPlace_Mean": mean_time,
            "CumulativeTimeSavingsPerPlace_Median": median_time,
            "CumulativeWeightedSavings_Mean": mean_weighted,
            "CumulativeWeightedSavings_Median": median_weighted,
        })
    return pd.DataFrame(rows)


st.set_page_config(page_title="Shipping Cost Dashboard", layout="wide")
st.title("Shipping Cost Dashboard")
tab_regionals_v3, tab_compare_networks, tab_roi, tab_weights = st.tabs(
    ["Shipping Summary Regionals v3", "Compare Shipping Networks", "ROI", "Destination Weights"]
)

with tab_regionals_v3:
    st.subheader("Shipping Summary Regionals v3")
    st.caption("Top combos for the regional v3 data (page3.csv).")
    regionals_v3_path = Path("page3.csv")
    if regionals_v3_path.exists():
        regionals_v3_df = load_data(str(regionals_v3_path))
        if regionals_v3_df.empty:
            st.info("No data available in page3.csv.")
        else:
            st.metric("Rows", f"{len(regionals_v3_df):,}")
            st.markdown("Data sanity checks")
            null_counts = regionals_v3_df.isna().sum()
            nulls_df = (
                null_counts[null_counts > 0]
                .sort_values(ascending=False)
                .rename("NullCount")
                .reset_index()
                .rename(columns={"index": "Column"})
            )
            if nulls_df.empty:
                st.caption("Null values: none")
            else:
                st.dataframe(nulls_df, use_container_width=True)

            unique_from = sorted(regionals_v3_df["FromAddress"].unique())
            unique_to = sorted(regionals_v3_df["Destination"].unique())
            total_possible_pairs = len(unique_from) * len(unique_to)
            actual_pairs = regionals_v3_df[["FromAddress", "Destination"]].drop_duplicates()
            missing_pairs = total_possible_pairs - len(actual_pairs)
            coverage_pct = (len(actual_pairs) / total_possible_pairs * 100.0) if total_possible_pairs else 0.0
            st.caption(
                f"From/To coverage: {len(actual_pairs):,} of {total_possible_pairs:,} "
                f"pairs ({coverage_pct:.1f}%), missing {missing_pairs:,}"
            )
            if missing_pairs == 0:
                st.caption("All FromAddress entries have routes to all Destination entries.")
            else:
                missing_from = [
                    from_city
                    for from_city in unique_from
                    if len(actual_pairs[actual_pairs["FromAddress"] == from_city]) != len(unique_to)
                ]
                if missing_from:
                    st.caption(f"From cities missing destinations: {len(missing_from)}")
                    st.selectbox(
                        "Example FromAddress missing destinations",
                        missing_from,
                        index=0,
                        key="v3_missing_from_example",
                    )
                    selected_missing_from = st.selectbox(
                        "Show missing destinations for",
                        missing_from,
                        index=0,
                        key="v3_missing_from_detail",
                    )
                    present = set(
                        actual_pairs[actual_pairs["FromAddress"] == selected_missing_from]["Destination"]
                    )
                    missing_destinations = [dest for dest in unique_to if dest not in present]
                    st.selectbox(
                        "Missing destinations for selected FromAddress",
                        missing_destinations or ["None"],
                        key="v3_missing_from_destinations",
                    )
                else:
                    st.caption("From cities missing destinations: 0")

            duplicates = regionals_v3_df.duplicated(
                subset=["FromAddress", "ToCity", "ToState", "PackageSize", "Service"],
                keep=False,
            )
            dup_count = int(duplicates.sum())
            st.caption(f"Duplicate rows (same from/to/size/service): {dup_count:,}")

            lat_long_path = Path("lat_long_from_to.csv")
            if lat_long_path.exists():
                    lat_long_df = load_lat_long_data(str(lat_long_path))
                    lat_long_df["direction"] = lat_long_df["direction"].astype(str).str.upper()
                    lat_long_df["city_key"] = (
                        lat_long_df["city"].astype(str).str.strip().str.casefold()
                        + "||"
                        + lat_long_df["state"].astype(str).str.strip().str.casefold()
                    )
                    from_keys = set(lat_long_df[lat_long_df["direction"] == "FROM"]["city_key"])
                    map_points = lat_long_df.dropna(subset=["lat", "lng"]).copy()
                    map_points["direction"] = np.where(
                        map_points["city_key"].isin(from_keys),
                        "FROM",
                        map_points["direction"],
                    )
                    if not map_points.empty:
                        st.markdown("Origin vs destination map")
                        weight_keys = set(V3_CITY_COUNTS.keys())
                        map_points["weight_key"] = (
                            map_points["city"].astype(str).str.strip().str.casefold()
                            + ", "
                            + map_points["state"].astype(str).str.strip().str.casefold()
                        )
                        map_points["weight_value"] = map_points["weight_key"].map(V3_CITY_COUNTS)
                        to_mask = map_points["direction"] == "TO"
                        to_weights = map_points.loc[to_mask, "weight_value"].dropna()
                        if not to_weights.empty:
                            min_w = float(to_weights.min())
                            max_w = float(to_weights.max())
                        else:
                            min_w = max_w = 0.0

                        def weight_to_color(val: float):
                            if max_w <= min_w:
                                return [217, 83, 79, 190]
                            t = (val - min_w) / (max_w - min_w)
                            r = int(255 * t + 245 * (1 - t))
                            g = int(80 * t + 140 * (1 - t))
                            b = int(60 * t + 60 * (1 - t))
                            return [r, g, b, 190]

                        def color_for_row(row):
                            if row["direction"] == "FROM":
                                return [27, 124, 212, 190]
                            val = row.get("weight_value")
                            if not np.isfinite(val):
                                return [180, 180, 180, 150]
                            return weight_to_color(float(val))

                        map_points["color_rgba"] = map_points.apply(color_for_row, axis=1)
                        view_state = pdk.ViewState(
                            latitude=float(map_points["lat"].mean()),
                            longitude=float(map_points["lng"].mean()),
                            zoom=3.5,
                        )
                        scatter = pdk.Layer(
                            "ScatterplotLayer",
                            data=map_points,
                            get_position="[lng, lat]",
                            get_fill_color="color_rgba",
                            get_radius=35000,
                            pickable=True,
                        )
                        st.pydeck_chart(
                            pdk.Deck(
                                layers=[scatter],
                                initial_view_state=view_state,
                                tooltip={
                                    "text": "{direction} - {name}\n{city}, {state} {postal_code}\nweight: {weight_value}"
                                },
                            )
                        )
                    else:
                        st.caption("No valid lat/long rows found in lat_long_from_to.csv.")
            else:
                st.caption("lat_long_from_to.csv not found for the map.")
            origin_list_all_v3 = sorted(regionals_v3_df["FromAddress"].unique())
            if not origin_list_all_v3:
                st.info("No origins available in shipping_summary_regionals_v3.csv.")
            else:
                default_baseline_v3 = (
                    "Harrisburg" if "Harrisburg" in origin_list_all_v3 else origin_list_all_v3[0]
                )
                baseline_origin_v3 = st.selectbox(
                    "Baseline origin (v3)",
                    origin_list_all_v3,
                    index=origin_list_all_v3.index(default_baseline_v3),
                    key="v3_baseline_origin",
                )
                cost_weight_v3 = st.slider(
                    "Cost weight (Time weight = 1 - Cost) (v3)",
                    0.0,
                    1.0,
                    0.0,
                    0.05,
                    key="v3_cost_weight",
                )
                build_cost_weight_v3 = cost_weight_v3

                dest_in_view_v3 = sorted(regionals_v3_df["Destination"].unique())
                if st.session_state.get("destination_weights_v3_version") != "city_counts_v1":
                    st.session_state.destination_weights_v3 = initial_destination_weights_by_city(
                        regionals_v3_df,
                        V3_CITY_COUNTS,
                    )
                    st.session_state.destination_weights_v3_version = "city_counts_v1"

                dest_weights_v3 = destination_weights(
                    dest_in_view_v3,
                    st.session_state.destination_weights_v3,
                )
                weight_keys = set(V3_CITY_COUNTS.keys())
                unique_dest_v3 = regionals_v3_df[["ToCity", "ToState"]].dropna().drop_duplicates().copy()
                unique_dest_v3["City"] = unique_dest_v3["ToCity"].astype(str).str.strip()
                unique_dest_v3["State"] = unique_dest_v3["ToState"].astype(str).str.strip()
                unique_dest_v3["WeightKey"] = (
                    unique_dest_v3["City"].str.casefold() + ", " + unique_dest_v3["State"].str.casefold()
                )
                missing_weight_keys = (
                    unique_dest_v3.loc[~unique_dest_v3["WeightKey"].isin(weight_keys), ["City", "State"]]
                    .assign(MissingDestination=lambda d: d["City"] + ", " + d["State"])["MissingDestination"]
                    .sort_values()
                    .tolist()
                )
                if missing_weight_keys:
                    st.warning(
                        f"Missing weights for {len(missing_weight_keys)} destinations.",
                        icon="⚠️",
                    )
                    st.text(", ".join(sorted(missing_weight_keys)))
                else:
                    st.caption("All destinations in page3.csv have weights.")

                avg_time_by_origin_v3 = (
                    regionals_v3_df.groupby(["FromAddress", "Destination"], as_index=False)
                    .agg(AvgTime=("ShippingTimeDays", "mean"))
                )
                avg_time_by_origin_v3["Weight"] = avg_time_by_origin_v3["Destination"].map(
                    dest_weights_v3
                ).fillna(0.0)
                avg_time_by_origin_v3["WeightedTime"] = (
                    avg_time_by_origin_v3["AvgTime"] * avg_time_by_origin_v3["Weight"]
                )
                weighted_avg_time_by_origin_v3 = (
                    avg_time_by_origin_v3.groupby("FromAddress", as_index=False)
                    .agg(
                        WeightedTimeSum=("WeightedTime", "sum"),
                        WeightSum=("Weight", "sum"),
                        FallbackMeanTime=("AvgTime", "mean"),
                    )
                )
                weighted_avg_time_by_origin_v3["WeightedAvgTime"] = np.where(
                    weighted_avg_time_by_origin_v3["WeightSum"] > 0,
                    weighted_avg_time_by_origin_v3["WeightedTimeSum"]
                    / weighted_avg_time_by_origin_v3["WeightSum"],
                    weighted_avg_time_by_origin_v3["FallbackMeanTime"],
                )
                weighted_avg_time_by_origin_v3 = weighted_avg_time_by_origin_v3[
                    ["FromAddress", "WeightedAvgTime"]
                ]
                if not weighted_avg_time_by_origin_v3.empty:
                    best_time_origin = weighted_avg_time_by_origin_v3.sort_values(
                        "WeightedAvgTime"
                    ).iloc[0]
                    st.metric(
                        "Best origin by weighted avg time (page3)",
                        best_time_origin["FromAddress"],
                        f"{best_time_origin['WeightedAvgTime']:.2f} days",
                    )

                st.markdown("Build network (page3)")
                built_network_origins_v3 = st.multiselect(
                    "Included origins (page3)",
                    sorted(regionals_v3_df["FromAddress"].unique()),
                    default=[],
                    key="page3_built_origins",
                )
                built_subset_v3 = regionals_v3_df[
                    regionals_v3_df["FromAddress"].isin(built_network_origins_v3)
                ]
                if built_subset_v3.empty:
                    st.caption("No origins selected for the page3 network.")
                else:
                    built_best_time_v3 = built_subset_v3.groupby("Destination", as_index=False).agg(
                        BestTime=("ShippingTimeDays", "min")
                    )
                    coverage_origin_map_v3 = compute_best_origin_map(built_subset_v3)
                    built_best_time_v3["Weight"] = built_best_time_v3["Destination"].map(
                        dest_weights_v3
                    ).fillna(0.0)
                    one_day_v3 = built_best_time_v3[built_best_time_v3["BestTime"] <= 1.0].copy()
                    two_day_v3 = built_best_time_v3[built_best_time_v3["BestTime"] == 2.0].copy()
                    three_day_v3 = built_best_time_v3[built_best_time_v3["BestTime"] == 3.0].copy()
                    weight_rank_v3 = (
                        pd.Series(dest_weights_v3)
                        .sort_values(ascending=False)
                        .reset_index()
                        .rename(columns={"index": "Destination", 0: "Weight"})
                    )
                    weight_rank_v3["Rank"] = np.arange(1, len(weight_rank_v3) + 1)
                    rank_map_v3 = dict(zip(weight_rank_v3["Destination"], weight_rank_v3["Rank"]))
                    total_weight_v3 = float(built_best_time_v3["Weight"].sum())
                    one_day_weight_v3 = float(one_day_v3["Weight"].sum())
                    two_day_weight_v3 = float(two_day_v3["Weight"].sum())
                    three_day_weight_v3 = float(three_day_v3["Weight"].sum())
                    two_day_coverage_weight_v3 = float(
                        built_best_time_v3[built_best_time_v3["BestTime"] <= 2.0]["Weight"].sum()
                    )
                    one_day_coverage_v3 = (
                        one_day_weight_v3 / total_weight_v3 * 100.0
                    ) if total_weight_v3 else 0.0
                    two_day_coverage_v3 = (
                        two_day_coverage_weight_v3 / total_weight_v3 * 100.0
                    ) if total_weight_v3 else 0.0
                    three_day_coverage_v3 = (
                        three_day_weight_v3 / total_weight_v3 * 100.0
                    ) if total_weight_v3 else 0.0
                    built_avg_time_v3 = weighted_mean(
                        built_best_time_v3["BestTime"].to_numpy(),
                        built_best_time_v3["Weight"].to_numpy(),
                    )
                    built_best_cost_v3 = built_subset_v3.groupby("Destination", as_index=False).agg(
                        BestCost=("Cost", "min")
                    )
                    built_best_cost_v3["Weight"] = built_best_cost_v3["Destination"].map(
                        dest_weights_v3
                    ).fillna(0.0)
                    built_avg_cost_v3 = weighted_mean(
                        built_best_cost_v3["BestCost"].to_numpy(),
                        built_best_cost_v3["Weight"].to_numpy(),
                    )
                    col_a, col_b = st.columns(2)
                    col_a.metric("Built network avg time (page3)", f"{built_avg_time_v3:.2f}")
                    col_b.metric("Built network avg cost (page3)", f"{built_avg_cost_v3:.2f}")
                    coverage_col_a, coverage_col_b, coverage_col_c = st.columns(3)
                    coverage_col_a.metric(
                        "1-day coverage (weighted) (page3)",
                        f"{one_day_coverage_v3:.1f}%",
                    )
                    coverage_col_b.metric(
                        "2-day coverage (weighted) (page3)",
                        f"{two_day_coverage_v3:.1f}%",
                    )
                    coverage_col_c.metric(
                        "3-day coverage (weighted) (page3)",
                        f"{three_day_coverage_v3:.1f}%",
                    )
                    origin_order_share_v3 = (
                        built_best_time_v3[["Destination", "Weight"]]
                        .assign(CoverageFrom=lambda d: d["Destination"].map(coverage_origin_map_v3))
                        .assign(
                            OriginList=lambda d: d["CoverageFrom"].apply(
                                lambda val: [
                                    origin.strip()
                                    for origin in str(val).split(",")
                                    if origin and origin.strip()
                                ]
                                if pd.notna(val)
                                else []
                            )
                        )
                    )
                    origin_order_share_v3["OriginCount"] = origin_order_share_v3["OriginList"].apply(len)
                    origin_order_share_v3 = origin_order_share_v3[
                        origin_order_share_v3["OriginCount"] > 0
                    ].copy()
                    origin_order_share_v3 = origin_order_share_v3.explode("OriginList")
                    origin_order_share_v3["OrderWeight"] = (
                        origin_order_share_v3["Weight"] / origin_order_share_v3["OriginCount"]
                    )
                    origin_order_share_v3 = (
                        origin_order_share_v3.groupby("OriginList", as_index=False)
                        .agg(OrderWeight=("OrderWeight", "sum"))
                        .rename(columns={"OriginList": "Origin"})
                    )
                    origin_order_share_v3 = (
                        pd.DataFrame({"Origin": sorted(set(built_network_origins_v3))})
                        .merge(origin_order_share_v3, on="Origin", how="left")
                        .fillna({"OrderWeight": 0.0})
                    )
                    origin_order_share_v3["OrderSharePct"] = np.where(
                        total_weight_v3 > 0,
                        origin_order_share_v3["OrderWeight"] / total_weight_v3 * 100.0,
                        0.0,
                    )
                    origin_order_share_v3 = origin_order_share_v3.sort_values(
                        ["OrderSharePct", "Origin"],
                        ascending=[False, True],
                    )
                    origin_order_share_display_v3 = origin_order_share_v3[
                        ["Origin", "OrderSharePct"]
                    ].copy()
                    origin_order_share_display_v3["OrderSharePct"] = origin_order_share_display_v3[
                        "OrderSharePct"
                    ].map(lambda x: f"{x:.1f}%")
                    st.markdown("Projected order share by included origin (page3)")
                    st.caption(
                        "Based on destination weights and the best-time origin per destination. "
                        "If multiple origins tie for best time on a destination, its share is split evenly across them."
                    )
                    st.dataframe(origin_order_share_display_v3, use_container_width=True)
                    st.markdown("1-day shipping cities (page3)")
                    if one_day_v3.empty:
                        st.caption("No destinations with 1-day shipping in the page3 network.")
                    else:
                        one_day_v3_display = one_day_v3[["Destination", "Weight"]].copy()
                        one_day_v3_display["PriorityRank"] = one_day_v3_display["Destination"].map(
                            rank_map_v3
                        )
                        one_day_v3_display["CoverageFrom"] = one_day_v3_display["Destination"].map(
                            coverage_origin_map_v3
                        )
                        one_day_v3_display = one_day_v3_display.sort_values(
                            ["PriorityRank", "Destination"],
                            ascending=[True, True],
                        )[["Destination", "Weight", "PriorityRank", "CoverageFrom"]]
                        st.dataframe(one_day_v3_display, use_container_width=True)
                    st.markdown("2-day shipping cities (page3)")
                    if two_day_v3.empty:
                        st.caption("No destinations with 2-day shipping in the page3 network.")
                    else:
                        two_day_v3_display = two_day_v3[["Destination", "Weight"]].copy()
                        two_day_v3_display["PriorityRank"] = two_day_v3_display["Destination"].map(
                            rank_map_v3
                        )
                        two_day_v3_display["CoverageFrom"] = two_day_v3_display["Destination"].map(
                            coverage_origin_map_v3
                        )
                        two_day_v3_display = two_day_v3_display.sort_values(
                            ["PriorityRank", "Destination"],
                            ascending=[True, True],
                        )[["Destination", "Weight", "PriorityRank", "CoverageFrom"]]
                        st.dataframe(two_day_v3_display, use_container_width=True)
                    st.markdown("3-day shipping cities (page3)")
                    if three_day_v3.empty:
                        st.caption("No destinations with 3-day shipping in the page3 network.")
                    else:
                        three_day_v3_display = three_day_v3[["Destination", "Weight"]].copy()
                        three_day_v3_display["PriorityRank"] = three_day_v3_display["Destination"].map(
                            rank_map_v3
                        )
                        three_day_v3_display["CoverageFrom"] = three_day_v3_display["Destination"].map(
                            coverage_origin_map_v3
                        )
                        three_day_v3_display = three_day_v3_display.sort_values(
                            ["PriorityRank", "Destination"],
                            ascending=[True, True],
                        )[["Destination", "Weight", "PriorityRank", "CoverageFrom"]]
                        st.dataframe(three_day_v3_display, use_container_width=True)
                    slow_1_v3 = built_best_time_v3[built_best_time_v3["BestTime"] > 1.0][
                        "Destination"
                    ].sort_values().tolist()
                    slow_2_v3 = built_best_time_v3[built_best_time_v3["BestTime"] > 2.0][
                        "Destination"
                    ].sort_values().tolist()
                    slow_3_v3 = built_best_time_v3[built_best_time_v3["BestTime"] > 3.0][
                        "Destination"
                    ].sort_values().tolist()
                    st.markdown("Cities without 1-day shipping (page3)")
                    if slow_1_v3:
                        st.selectbox("Destinations > 1 day (page3)", slow_1_v3, key="page3_slow_1")
                    else:
                        st.caption("All destinations have 1-day shipping in the page3 network.")
                    st.markdown("Cities without 2-day shipping (page3)")
                    if slow_2_v3:
                        st.selectbox("Destinations > 2 days (page3)", slow_2_v3, key="page3_slow_2")
                    else:
                        st.caption("All destinations have 2-day shipping in the page3 network.")
                    st.markdown("Cities without 3-day shipping (page3)")
                    if slow_3_v3:
                        st.selectbox("Destinations > 3 days (page3)", slow_3_v3, key="page3_slow_3")
                    else:
                        st.caption("All destinations have 3-day shipping in the page3 network.")

                if baseline_origin_v3 not in regionals_v3_df["FromAddress"].unique():
                    baseline_origin_v3 = origin_list_all_v3[0]

                baseline_time_by_dest_v3 = regionals_v3_df[
                    regionals_v3_df["FromAddress"] == baseline_origin_v3
                ].groupby("Destination", as_index=False).agg(AvgTime=("ShippingTimeDays", "mean"))
                baseline_time_by_dest_v3["Weight"] = baseline_time_by_dest_v3["Destination"].map(
                    dest_weights_v3
                ).fillna(0.0)
                avg_time_v3 = weighted_mean(
                    baseline_time_by_dest_v3["AvgTime"].to_numpy(),
                    baseline_time_by_dest_v3["Weight"].to_numpy(),
                )
                baseline_cost_by_dest_v3 = regionals_v3_df[
                    regionals_v3_df["FromAddress"] == baseline_origin_v3
                ].groupby("Destination", as_index=False).agg(AvgCost=("Cost", "mean"))
                baseline_cost_by_dest_v3["Weight"] = baseline_cost_by_dest_v3["Destination"].map(
                    dest_weights_v3
                ).fillna(0.0)
                avg_cost_v3 = weighted_mean(
                    baseline_cost_by_dest_v3["AvgCost"].to_numpy(),
                    baseline_cost_by_dest_v3["Weight"].to_numpy(),
                )

                one_day_origins_v3 = sorted(
                    regionals_v3_df[regionals_v3_df["ShippingTimeDays"] <= 1.0][
                        "FromAddress"
                    ].unique().tolist()
                )
                avg_time_by_origin_v3 = regionals_v3_df.groupby("FromAddress", as_index=False).agg(
                    AvgTime=("ShippingTimeDays", "mean")
                )
                baseline_avg_time_v3 = float(
                    avg_time_by_origin_v3[
                        avg_time_by_origin_v3["FromAddress"] == baseline_origin_v3
                    ]["AvgTime"].iloc[0]
                ) if baseline_origin_v3 in avg_time_by_origin_v3["FromAddress"].values else float("nan")
                avg_time_by_origin_v3["TimeReductionVsBaseline"] = (
                    baseline_avg_time_v3 - avg_time_by_origin_v3["AvgTime"]
                )
                top_time_reducers_v3 = (
                    avg_time_by_origin_v3.sort_values("TimeReductionVsBaseline", ascending=False)
                    .head(10)["FromAddress"]
                    .tolist()
                )
                origin_candidates_v3 = sorted(
                    {baseline_origin_v3} | set(one_day_origins_v3) | set(top_time_reducers_v3)
                )
                origin_list_v3 = [o for o in origin_candidates_v3 if o in origin_list_all_v3]

                major_destinations_v3 = [
                    dest
                    for dest in dest_in_view_v3
                    if float(st.session_state.destination_weights_v3.get(dest, DEFAULT_DEST_WEIGHT)) > 1.0
                ]
                major_weight_map_v3 = {
                    dest: float(st.session_state.destination_weights_v3.get(dest, DEFAULT_DEST_WEIGHT))
                    for dest in major_destinations_v3
                }

                include_boise_v3 = st.checkbox(
                    "Include Boise in all combos (page3)",
                    value=False,
                    key="v3_include_boise",
                )
                render_top_combos(
                    regionals_v3_df,
                    origin_list_v3,
                    origin_list_all_v3,
                    dest_in_view_v3,
                    dest_weights_v3,
                    build_cost_weight_v3,
                    avg_cost_v3,
                    avg_time_v3,
                    major_weight_map_v3,
                    "regionals_v3",
                    show_day_percentages=True,
                    max_k=20,
                    baseline_origin=baseline_origin_v3,
                    required_origin="Boise" if include_boise_v3 else None,
                )
    else:
        st.info("page3.csv not found in the app folder.")

with tab_compare_networks:
    st.subheader("Compare Shipping Networks")
    st.caption(
        "Build two networks side by side, highlight the better side for each metric, and review unique priority-city coverage."
    )

    compare_page3_path = Path("page3.csv")
    if compare_page3_path.exists():
        compare_df = load_data(str(compare_page3_path))
        if compare_df.empty:
            st.info("No data available in page3.csv.")
        else:
            origin_options = sorted(compare_df["FromAddress"].unique())
            dest_in_view_compare = sorted(compare_df["Destination"].unique())

            if st.session_state.get("destination_weights_v3_version") != "city_counts_v1":
                st.session_state.destination_weights_v3 = initial_destination_weights_by_city(
                    compare_df,
                    V3_CITY_COUNTS,
                )
                st.session_state.destination_weights_v3_version = "city_counts_v1"

            dest_weights_compare = destination_weights(
                dest_in_view_compare,
                st.session_state.destination_weights_v3,
            )

            st.markdown("### ROI Assumptions")
            roi_col_1, roi_col_2 = st.columns(2)
            with roi_col_1:
                monthly_packages = int(
                    st.slider(
                        "Packages sold per month",
                        min_value=0,
                        max_value=30000,
                        value=10000,
                        step=100,
                        key="compare_monthly_packages",
                    )
                )
                base_profit_per_package = float(
                    st.slider(
                        "Base profit per package ($)",
                        min_value=0.0,
                        max_value=50.0,
                        value=10.0,
                        step=0.5,
                        key="compare_base_profit_per_package",
                    )
                )
                base_revenue_per_package = float(
                    st.slider(
                        "Revenue per package ($)",
                        min_value=0.0,
                        max_value=200.0,
                        value=10.0,
                        step=0.5,
                        key="compare_base_revenue_per_package",
                    )
                )
            with roi_col_2:
                one_day_bonus = float(
                    st.slider(
                        "Extra profit for 1-day shipping ($/package)",
                        min_value=0.0,
                        max_value=20.0,
                        value=2.0,
                        step=0.25,
                        key="compare_one_day_bonus",
                    )
                )
                two_day_bonus = float(
                    st.slider(
                        "Extra profit for 2-day shipping ($/package)",
                        min_value=0.0,
                        max_value=20.0,
                        value=1.0,
                        step=0.25,
                        key="compare_two_day_bonus",
                    )
                )
                three_day_penalty = float(
                    st.slider(
                        "Profit loss for 3-day shipping ($/package)",
                        min_value=0.0,
                        max_value=20.0,
                        value=0.0,
                        step=0.25,
                        key="compare_three_day_penalty",
                    )
                )
            st.caption(
                "Revenue uses: revenue per package + 1-day/2-day boosts. "
                "3-day penalty affects profit only."
            )
            default_base_city = "Harrisburg" if "Harrisburg" in origin_options else origin_options[0]
            comparison_base_city = st.selectbox(
                "Comparison base city for projected profit shipping savings",
                origin_options,
                index=origin_options.index(default_base_city),
                key="compare_roi_base_city",
            )
            st.caption(
                "Projected monthly profit includes shipping-cost savings vs this base city: "
                "(base city avg cost - network avg cost) x packages/month."
            )

            selector_col_a, selector_col_b = st.columns(2)
            with selector_col_a:
                st.markdown("### Network A")
                network_a_origins = st.multiselect(
                    "Included origins (Network A)",
                    origin_options,
                    default=[],
                    key="compare_network_a_origins",
                )
            with selector_col_b:
                st.markdown("### Network B")
                network_b_origins = st.multiselect(
                    "Included origins (Network B)",
                    origin_options,
                    default=[],
                    key="compare_network_b_origins",
                )

            comparison_base_subset = compare_df[compare_df["FromAddress"] == comparison_base_city]
            comparison_base_summary = (
                compute_built_network_summary(comparison_base_subset, dest_weights_compare)
                if not comparison_base_subset.empty
                else None
            )
            comparison_base_city_avg_cost = (
                float(comparison_base_summary["avg_cost"])
                if comparison_base_summary is not None and np.isfinite(comparison_base_summary["avg_cost"])
                else None
            )

            subset_a = compare_df[compare_df["FromAddress"].isin(network_a_origins)]
            subset_b = compare_df[compare_df["FromAddress"].isin(network_b_origins)]
            summary_a = (
                compute_built_network_summary(subset_a, dest_weights_compare)
                if not subset_a.empty
                else None
            )
            summary_b = (
                compute_built_network_summary(subset_b, dest_weights_compare)
                if not subset_b.empty
                else None
            )
            roi_a = apply_shipping_cost_savings_to_projected_profit(
                compute_network_roi_projection(
                    summary_a,
                    dest_weights_compare,
                    monthly_packages,
                    base_revenue_per_package,
                    base_profit_per_package,
                    one_day_bonus,
                    two_day_bonus,
                    three_day_penalty,
                ),
                summary_a,
                comparison_base_city_avg_cost,
                monthly_packages,
            )
            roi_b = apply_shipping_cost_savings_to_projected_profit(
                compute_network_roi_projection(
                    summary_b,
                    dest_weights_compare,
                    monthly_packages,
                    base_revenue_per_package,
                    base_profit_per_package,
                    one_day_bonus,
                    two_day_bonus,
                    three_day_penalty,
                ),
                summary_b,
                comparison_base_city_avg_cost,
                monthly_packages,
            )
            recommended_city_a, recommended_lift_a = recommend_next_city_by_profit(
                compare_df,
                network_a_origins,
                origin_options,
                dest_weights_compare,
                monthly_packages,
                base_revenue_per_package,
                base_profit_per_package,
                one_day_bonus,
                two_day_bonus,
                three_day_penalty,
                comparison_base_city_avg_cost=comparison_base_city_avg_cost,
            )
            recommended_city_b, recommended_lift_b = recommend_next_city_by_profit(
                compare_df,
                network_b_origins,
                origin_options,
                dest_weights_compare,
                monthly_packages,
                base_revenue_per_package,
                base_profit_per_package,
                one_day_bonus,
                two_day_bonus,
                three_day_penalty,
                comparison_base_city_avg_cost=comparison_base_city_avg_cost,
            )

            highlights_a = {}
            highlights_b = {}
            roi_highlights_a = {}
            roi_highlights_b = {}
            shipping_saved_vs_other_monthly_a = float("nan")
            shipping_saved_vs_other_monthly_b = float("nan")
            if summary_a is not None and summary_b is not None:
                time_a, time_b = compare_metric_winners(
                    summary_a["avg_time"],
                    summary_b["avg_time"],
                    higher_is_better=False,
                )
                cost_a, cost_b = compare_metric_winners(
                    summary_a["avg_cost"],
                    summary_b["avg_cost"],
                    higher_is_better=False,
                )
                cov1_a, cov1_b = compare_metric_winners(
                    summary_a["coverage_1_day"],
                    summary_b["coverage_1_day"],
                    higher_is_better=True,
                )
                cov2_a, cov2_b = compare_metric_winners(
                    summary_a["coverage_2_day"],
                    summary_b["coverage_2_day"],
                    higher_is_better=True,
                )
                cov3_a, cov3_b = compare_metric_winners(
                    summary_a["coverage_3_day"],
                    summary_b["coverage_3_day"],
                    higher_is_better=True,
                )
                highlights_a = {
                    "avg_time": time_a,
                    "avg_cost": cost_a,
                    "coverage_1_day": cov1_a,
                    "coverage_2_day": cov2_a,
                    "coverage_3_day": cov3_a,
                }
                highlights_b = {
                    "avg_time": time_b,
                    "avg_cost": cost_b,
                    "coverage_1_day": cov1_b,
                    "coverage_2_day": cov2_b,
                    "coverage_3_day": cov3_b,
                }
                shipping_saved_vs_other_monthly_a = float(
                    max(summary_b["avg_cost"] - summary_a["avg_cost"], 0.0) * monthly_packages
                )
                shipping_saved_vs_other_monthly_b = float(
                    max(summary_a["avg_cost"] - summary_b["avg_cost"], 0.0) * monthly_packages
                )
            if roi_a is not None and roi_b is not None:
                ship_saved_a, ship_saved_b = compare_metric_winners(
                    shipping_saved_vs_other_monthly_a,
                    shipping_saved_vs_other_monthly_b,
                    higher_is_better=True,
                )
                revenue_a, revenue_b = compare_metric_winners(
                    roi_a["projected_revenue"],
                    roi_b["projected_revenue"],
                    higher_is_better=True,
                )
                yearly_revenue_a, yearly_revenue_b = compare_metric_winners(
                    roi_a["projected_revenue_yearly"],
                    roi_b["projected_revenue_yearly"],
                    higher_is_better=True,
                )
                profit_a, profit_b = compare_metric_winners(
                    roi_a["projected_profit"],
                    roi_b["projected_profit"],
                    higher_is_better=True,
                )
                yearly_profit_a, yearly_profit_b = compare_metric_winners(
                    roi_a["projected_profit_yearly"],
                    roi_b["projected_profit_yearly"],
                    higher_is_better=True,
                )
                uplift_a, uplift_b = compare_metric_winners(
                    roi_a["shipping_uplift_profit"],
                    roi_b["shipping_uplift_profit"],
                    higher_is_better=True,
                )
                avg_ship_saved_pkg_a, avg_ship_saved_pkg_b = compare_metric_winners(
                    roi_a["avg_shipping_saved_per_package_vs_base_city"],
                    roi_b["avg_shipping_saved_per_package_vs_base_city"],
                    higher_is_better=True,
                )
                avg_pp_a, avg_pp_b = compare_metric_winners(
                    roi_a["avg_profit_per_package"],
                    roi_b["avg_profit_per_package"],
                    higher_is_better=True,
                )
                one_day_pkg_a, one_day_pkg_b = compare_metric_winners(
                    roi_a["one_day_packages"],
                    roi_b["one_day_packages"],
                    higher_is_better=True,
                )
                two_day_pkg_a, two_day_pkg_b = compare_metric_winners(
                    roi_a["two_day_packages"],
                    roi_b["two_day_packages"],
                    higher_is_better=True,
                )
                three_day_pkg_a, three_day_pkg_b = compare_metric_winners(
                    roi_a["three_day_packages"],
                    roi_b["three_day_packages"],
                    higher_is_better=True,
                )
                roi_highlights_a = {
                    "shipping_saved_vs_other_monthly": ship_saved_a,
                    "projected_revenue": revenue_a,
                    "projected_revenue_yearly": yearly_revenue_a,
                    "projected_profit": profit_a,
                    "projected_profit_yearly": yearly_profit_a,
                    "shipping_uplift_profit": uplift_a,
                    "avg_shipping_saved_per_package_vs_base_city": avg_ship_saved_pkg_a,
                    "avg_profit_per_package": avg_pp_a,
                    "one_day_packages": one_day_pkg_a,
                    "two_day_packages": two_day_pkg_a,
                    "three_day_packages": three_day_pkg_a,
                }
                roi_highlights_b = {
                    "shipping_saved_vs_other_monthly": ship_saved_b,
                    "projected_revenue": revenue_b,
                    "projected_revenue_yearly": yearly_revenue_b,
                    "projected_profit": profit_b,
                    "projected_profit_yearly": yearly_profit_b,
                    "shipping_uplift_profit": uplift_b,
                    "avg_shipping_saved_per_package_vs_base_city": avg_ship_saved_pkg_b,
                    "avg_profit_per_package": avg_pp_b,
                    "one_day_packages": one_day_pkg_b,
                    "two_day_packages": two_day_pkg_b,
                    "three_day_packages": three_day_pkg_b,
                }

            panel_col_a, panel_col_b = st.columns(2)
            with panel_col_a:
                st.markdown("### Network A Results")
                render_network_builder_panel(
                    summary_a,
                    "Network A",
                    "compare_network_a",
                    highlights_a,
                )
                st.markdown("### ROI Analysis (Network A)")
                if roi_a is None:
                    st.caption("No ROI projection yet for Network A.")
                else:
                    render_colored_metric(
                        "Projected monthly revenue (Network A)",
                        f"${roi_a['projected_revenue']:,.2f}",
                        bool(roi_highlights_a.get("projected_revenue", False)),
                    )
                    render_colored_metric(
                        "Projected monthly profit (Network A)",
                        f"${roi_a['projected_profit']:,.2f}",
                        bool(roi_highlights_a.get("projected_profit", False)),
                    )
                    render_colored_metric(
                        "Shipping-speed uplift profit (Network A)",
                        f"${roi_a['shipping_uplift_profit']:,.2f}",
                        bool(roi_highlights_a.get("shipping_uplift_profit", False)),
                    )
                    render_colored_metric(
                        "Shipping cost saved vs Network B (monthly)",
                        (
                            f"${shipping_saved_vs_other_monthly_a:,.2f}"
                            if np.isfinite(shipping_saved_vs_other_monthly_a)
                            else "N/A"
                        ),
                        bool(roi_highlights_a.get("shipping_saved_vs_other_monthly", False)),
                    )
                    render_colored_metric(
                        "Avg shipping saved per package (Network A)",
                        f"${roi_a['avg_shipping_saved_per_package_vs_base_city']:.2f}",
                        bool(roi_highlights_a.get("avg_shipping_saved_per_package_vs_base_city", False)),
                    )
                    render_colored_metric(
                        "Avg profit per package (Network A)",
                        f"${roi_a['avg_profit_per_package']:.2f}",
                        bool(roi_highlights_a.get("avg_profit_per_package", False)),
                    )
                    render_colored_metric(
                        "Projected 1-day packages (Network A)",
                        f"{roi_a['one_day_packages']:,.0f}",
                        bool(roi_highlights_a.get("one_day_packages", False)),
                    )
                    render_colored_metric(
                        "Projected 2-day packages (Network A)",
                        f"{roi_a['two_day_packages']:,.0f}",
                        bool(roi_highlights_a.get("two_day_packages", False)),
                    )
                    render_colored_metric(
                        "Projected 3-day packages (Network A)",
                        f"{roi_a['three_day_packages']:,.0f}",
                        bool(roi_highlights_a.get("three_day_packages", False)),
                    )
                    render_colored_metric(
                        "Projected yearly profit (Network A)",
                        f"${roi_a['projected_profit_yearly']:,.2f}",
                        bool(roi_highlights_a.get("projected_profit_yearly", False)),
                    )
                    render_colored_metric(
                        "Projected yearly revenue (Network A)",
                        f"${roi_a['projected_revenue_yearly']:,.2f}",
                        bool(roi_highlights_a.get("projected_revenue_yearly", False)),
                    )
                st.markdown("### Recommended Next City (Network A)")
                if not network_a_origins:
                    st.info("Select at least one origin in Network A to see a recommendation.")
                elif recommended_city_a is None or recommended_lift_a is None:
                    st.info("No additional city available for recommendation.")
                else:
                    lift_label_a = (
                        f"+${recommended_lift_a:,.2f}"
                        if recommended_lift_a >= 0
                        else f"-${abs(recommended_lift_a):,.2f}"
                    )
                    st.success(
                        f"Recommended next city: {recommended_city_a} "
                        f"({lift_label_a}/month projected profit vs current network)."
                    )
            with panel_col_b:
                st.markdown("### Network B Results")
                render_network_builder_panel(
                    summary_b,
                    "Network B",
                    "compare_network_b",
                    highlights_b,
                )
                st.markdown("### ROI Analysis (Network B)")
                if roi_b is None:
                    st.caption("No ROI projection yet for Network B.")
                else:
                    render_colored_metric(
                        "Projected monthly revenue (Network B)",
                        f"${roi_b['projected_revenue']:,.2f}",
                        bool(roi_highlights_b.get("projected_revenue", False)),
                    )
                    render_colored_metric(
                        "Projected monthly profit (Network B)",
                        f"${roi_b['projected_profit']:,.2f}",
                        bool(roi_highlights_b.get("projected_profit", False)),
                    )
                    render_colored_metric(
                        "Shipping-speed uplift profit (Network B)",
                        f"${roi_b['shipping_uplift_profit']:,.2f}",
                        bool(roi_highlights_b.get("shipping_uplift_profit", False)),
                    )
                    render_colored_metric(
                        "Shipping cost saved vs Network A (monthly)",
                        (
                            f"${shipping_saved_vs_other_monthly_b:,.2f}"
                            if np.isfinite(shipping_saved_vs_other_monthly_b)
                            else "N/A"
                        ),
                        bool(roi_highlights_b.get("shipping_saved_vs_other_monthly", False)),
                    )
                    render_colored_metric(
                        "Avg shipping saved per package (Network B)",
                        f"${roi_b['avg_shipping_saved_per_package_vs_base_city']:.2f}",
                        bool(roi_highlights_b.get("avg_shipping_saved_per_package_vs_base_city", False)),
                    )
                    render_colored_metric(
                        "Avg profit per package (Network B)",
                        f"${roi_b['avg_profit_per_package']:.2f}",
                        bool(roi_highlights_b.get("avg_profit_per_package", False)),
                    )
                    render_colored_metric(
                        "Projected 1-day packages (Network B)",
                        f"{roi_b['one_day_packages']:,.0f}",
                        bool(roi_highlights_b.get("one_day_packages", False)),
                    )
                    render_colored_metric(
                        "Projected 2-day packages (Network B)",
                        f"{roi_b['two_day_packages']:,.0f}",
                        bool(roi_highlights_b.get("two_day_packages", False)),
                    )
                    render_colored_metric(
                        "Projected 3-day packages (Network B)",
                        f"{roi_b['three_day_packages']:,.0f}",
                        bool(roi_highlights_b.get("three_day_packages", False)),
                    )
                    render_colored_metric(
                        "Projected yearly profit (Network B)",
                        f"${roi_b['projected_profit_yearly']:,.2f}",
                        bool(roi_highlights_b.get("projected_profit_yearly", False)),
                    )
                    render_colored_metric(
                        "Projected yearly revenue (Network B)",
                        f"${roi_b['projected_revenue_yearly']:,.2f}",
                        bool(roi_highlights_b.get("projected_revenue_yearly", False)),
                    )
                st.markdown("### Recommended Next City (Network B)")
                if not network_b_origins:
                    st.info("Select at least one origin in Network B to see a recommendation.")
                elif recommended_city_b is None or recommended_lift_b is None:
                    st.info("No additional city available for recommendation.")
                else:
                    lift_label_b = (
                        f"+${recommended_lift_b:,.2f}"
                        if recommended_lift_b >= 0
                        else f"-${abs(recommended_lift_b):,.2f}"
                    )
                    st.success(
                        f"Recommended next city: {recommended_city_b} "
                        f"({lift_label_b}/month projected profit vs current network)."
                    )

            st.markdown("### Unique Priority Cities By Coverage Threshold")
            if summary_a is None and summary_b is None:
                st.caption("Select origins in at least one network to compare unique coverage.")
            else:
                for threshold, label in [(1.0, "1-day"), (2.0, "2-day"), (3.0, "3-day")]:
                    st.markdown(f"#### {label} Unique Coverage")
                    unique_a = unique_priority_cities_for_threshold(
                        summary_a,
                        summary_b,
                        threshold,
                        limit=5,
                    )
                    unique_b = unique_priority_cities_for_threshold(
                        summary_b,
                        summary_a,
                        threshold,
                        limit=5,
                    )
                    unique_col_a, unique_col_b = st.columns(2)
                    with unique_col_a:
                        st.markdown("Network A cities not covered by Network B")
                        if unique_a.empty:
                            st.caption(f"No unique {label} cities for Network A.")
                        else:
                            st.dataframe(unique_a, use_container_width=True)
                    with unique_col_b:
                        st.markdown("Network B cities not covered by Network A")
                        if unique_b.empty:
                            st.caption(f"No unique {label} cities for Network B.")
                        else:
                            st.dataframe(unique_b, use_container_width=True)

            st.markdown("### Projected Savings Vs Comparison City")
            demand_increase_pct = float(
                st.slider(
                    "Demand shift from comparison city to built networks (%)",
                    min_value=0.0,
                    max_value=200.0,
                    value=0.0,
                    step=1.0,
                    key="compare_demand_increase_pct",
                )
            )
            default_compare_city = (
                comparison_base_city if comparison_base_city in origin_options else origin_options[0]
            )
            comparison_city = st.selectbox(
                "Comparison city for savings analysis",
                origin_options,
                index=origin_options.index(default_compare_city),
                key="compare_savings_city",
            )

            comparison_subset = compare_df[compare_df["FromAddress"] == comparison_city]
            comparison_summary = (
                compute_built_network_summary(comparison_subset, dest_weights_compare)
                if not comparison_subset.empty
                else None
            )
            if comparison_summary is None:
                st.caption("Comparison city has no usable data for savings analysis.")
            else:
                built_network_monthly_packages = int(monthly_packages)
                demand_shift_multiplier = 1.0 + (demand_increase_pct / 100.0)
                comparison_city_monthly_packages = int(
                    round(built_network_monthly_packages / demand_shift_multiplier)
                )
                comparison_city_roi = apply_shipping_cost_savings_to_projected_profit(
                    compute_network_roi_projection(
                        comparison_summary,
                        dest_weights_compare,
                        comparison_city_monthly_packages,
                        base_revenue_per_package,
                        base_profit_per_package,
                        one_day_bonus,
                        two_day_bonus,
                        three_day_penalty,
                    ),
                    comparison_summary,
                    comparison_base_city_avg_cost,
                    comparison_city_monthly_packages,
                )
                network_a_roi_vs_city = (
                    apply_shipping_cost_savings_to_projected_profit(
                        compute_network_roi_projection(
                            summary_a,
                            dest_weights_compare,
                            built_network_monthly_packages,
                            base_revenue_per_package,
                            base_profit_per_package,
                            one_day_bonus,
                            two_day_bonus,
                            three_day_penalty,
                        ),
                        summary_a,
                        comparison_base_city_avg_cost,
                        built_network_monthly_packages,
                    ) if summary_a is not None else None
                )
                network_b_roi_vs_city = (
                    apply_shipping_cost_savings_to_projected_profit(
                        compute_network_roi_projection(
                            summary_b,
                            dest_weights_compare,
                            built_network_monthly_packages,
                            base_revenue_per_package,
                            base_profit_per_package,
                            one_day_bonus,
                            two_day_bonus,
                            three_day_penalty,
                        ),
                        summary_b,
                        comparison_base_city_avg_cost,
                        built_network_monthly_packages,
                    ) if summary_b is not None else None
                )

                savings_rows = []
                for network_name, network_roi in [
                    ("Network A", network_a_roi_vs_city),
                    ("Network B", network_b_roi_vs_city),
                ]:
                    if network_roi is None:
                        savings_rows.append(
                            {
                                "Network": network_name,
                                "ProjectedSavedMonthlyVsCity": float("nan"),
                                "ProjectedSavedYearlyVsCity": float("nan"),
                            }
                        )
                        continue
                    monthly_saved = network_roi["projected_profit"] - comparison_city_roi["projected_profit"]
                    yearly_saved = network_roi["projected_profit_yearly"] - comparison_city_roi["projected_profit_yearly"]
                    savings_rows.append(
                        {
                            "Network": network_name,
                            "ProjectedSavedMonthlyVsCity": monthly_saved,
                            "ProjectedSavedYearlyVsCity": yearly_saved,
                        }
                    )

                savings_df = pd.DataFrame(savings_rows)
                savings_display_df = savings_df.copy()
                for col in ["ProjectedSavedMonthlyVsCity", "ProjectedSavedYearlyVsCity"]:
                    savings_display_df[col] = savings_display_df[col].apply(
                        lambda val: f"${val:,.2f}" if np.isfinite(val) else "N/A"
                    )
                st.caption(
                    "Calculation: ProjectedSavedMonthlyVsCity = built network projected monthly profit "
                    "- comparison city projected monthly profit. "
                    "ProjectedSavedYearlyVsCity = built network projected yearly profit "
                    "- comparison city projected yearly profit."
                )
                st.caption(
                    "Demand-shift calculation: comparison city packages = built network packages / "
                    "(1 + demand shift % / 100). Example: 100% shift means the comparison city is at 50% of network volume."
                )
                st.caption(
                    f"Built networks use {built_network_monthly_packages:,} packages/month. "
                    f"{comparison_city} uses {comparison_city_monthly_packages:,} packages/month after demand shift."
                )
                st.dataframe(savings_display_df, use_container_width=True)
    else:
        st.info("page3.csv not found in the app folder.")

with tab_roi:
    st.subheader("ROI")
    st.caption(
        "Estimate monthly shipping savings for a built network vs a comparison location."
    )

    small_path = Path("small.csv")
    large_path = Path("page3.csv")
    if not small_path.exists():
        st.info("small.csv not found in the app folder.")
    elif not large_path.exists():
        st.info("page3.csv not found in the app folder.")
    else:
        small_df = load_data(str(small_path))
        large_df = load_data(str(large_path))
        if small_df.empty:
            st.info("No data available in small.csv.")
        elif large_df.empty:
            st.info("No data available in page3.csv.")
        else:
            small_weight_map = initial_destination_weights_by_city(small_df, V3_CITY_COUNTS)
            large_weight_map = initial_destination_weights_by_city(large_df, V3_CITY_COUNTS)
            small_origins = set(small_df["FromAddress"].dropna().unique().tolist())
            large_origins = set(large_df["FromAddress"].dropna().unique().tolist())
            origin_options = sorted(small_origins & large_origins)
            if not origin_options:
                st.warning(
                    "No shared origins between small.csv and page3.csv. ROI requires shared origins."
                )
            else:
                st.caption("Small signs are sourced from small.csv. Large signs are sourced from page3.csv.")
                comparison_origin = st.selectbox(
                    "Comparison location",
                    origin_options,
                    index=0,
                    key="roi_comparison_origin",
                )
                built_network_origins = st.multiselect(
                    "Built network origins",
                    origin_options,
                    default=[],
                    key="roi_built_origins",
                )

                cost_col1, cost_col2 = st.columns(2)
                built_monthly_cost = float(
                    cost_col1.number_input(
                        "Built network monthly cost",
                        min_value=0.0,
                        value=18000.0,
                        step=100.0,
                        key="roi_built_monthly_cost",
                    )
                )
                comparison_monthly_cost = float(
                    cost_col2.number_input(
                        "Comparison location monthly cost",
                        min_value=0.0,
                        value=6000.0,
                        step=100.0,
                        key="roi_comparison_monthly_cost",
                    )
                )

                count_col1, count_col2 = st.columns(2)
                small_sign_count = int(
                    count_col1.number_input(
                        "Small signs per month",
                        min_value=0,
                        value=0,
                        step=1,
                        key="roi_small_sign_count",
                    )
                )
                large_sign_count = int(
                    count_col2.number_input(
                        "Large signs per month",
                        min_value=0,
                        value=0,
                        step=1,
                        key="roi_large_sign_count",
                    )
                )

                if not built_network_origins:
                    st.caption("Select at least one built network origin to calculate ROI.")
                else:
                    network_tuple = tuple(sorted(set(built_network_origins)))
                    small_stats = compute_network_cost_stats(
                        small_df,
                        network_tuple,
                        comparison_origin,
                        small_weight_map,
                    )
                    large_stats = compute_network_cost_stats(
                        large_df,
                        network_tuple,
                        comparison_origin,
                        large_weight_map,
                    )

                    if small_stats["destination_count"] == 0:
                        st.warning("No overlapping destinations found for small-sign ROI calculation.")
                    if large_stats["destination_count"] == 0:
                        st.warning("No overlapping destinations found for large-sign ROI calculation.")

                    small_savings_per_sign = (
                        small_stats["savings_per_sign"]
                        if np.isfinite(small_stats["savings_per_sign"])
                        else 0.0
                    )
                    large_savings_per_sign = (
                        large_stats["savings_per_sign"]
                        if np.isfinite(large_stats["savings_per_sign"])
                        else 0.0
                    )

                    total_small_savings = small_sign_count * small_savings_per_sign
                    total_large_savings = large_sign_count * large_savings_per_sign
                    total_sign_savings = total_small_savings + total_large_savings
                    built_minus_sign_savings = built_monthly_cost - total_sign_savings
                    fixed_monthly_gap = built_monthly_cost - comparison_monthly_cost
                    net_monthly_after_savings = fixed_monthly_gap - total_sign_savings
                    total_sign_count = small_sign_count + large_sign_count
                    blended_savings_per_sign = (
                        total_sign_savings / total_sign_count if total_sign_count > 0 else float("nan")
                    )
                    signs_to_cover_built_monthly = (
                        fixed_monthly_gap / blended_savings_per_sign
                        if np.isfinite(blended_savings_per_sign) and blended_savings_per_sign > 0
                        else float("nan")
                    )

                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    metric_col1.metric("Small savings per sign", f"${small_savings_per_sign:.2f}")
                    metric_col2.metric("Large savings per sign", f"${large_savings_per_sign:.2f}")
                    metric_col3.metric("Total sign savings / month", f"${total_sign_savings:,.2f}")
                    metric_col4.metric("Built monthly - sign savings", f"${built_minus_sign_savings:,.2f}")

                    metric_col5, metric_col6, metric_col7 = st.columns(3)
                    metric_col5.metric("Fixed monthly gap", f"${fixed_monthly_gap:,.2f}")
                    metric_col6.metric(
                        "Net monthly after sign savings",
                        f"${net_monthly_after_savings:,.2f}",
                    )
                    metric_col7.metric(
                        "Signs to cover monthly gap",
                        (
                            f"{int(np.ceil(signs_to_cover_built_monthly)):,}"
                            if np.isfinite(signs_to_cover_built_monthly)
                            else "N/A"
                        ),
                    )

                    st.caption(
                        "Net monthly after sign savings = (Built network monthly cost - Comparison monthly cost) "
                        "- Total sign savings."
                    )

                    details_df = pd.DataFrame(
                        [
                            {
                                "SignType": "Small",
                                "SignsPerMonth": small_sign_count,
                                "SavingsPerSign": small_savings_per_sign,
                                "TotalSavings": total_small_savings,
                                "MatchedDestinations": small_stats["destination_count"],
                            },
                            {
                                "SignType": "Large",
                                "SignsPerMonth": large_sign_count,
                                "SavingsPerSign": large_savings_per_sign,
                                "TotalSavings": total_large_savings,
                                "MatchedDestinations": large_stats["destination_count"],
                            },
                        ]
                    )
                    st.dataframe(details_df, use_container_width=True)

with tab_weights:
    st.subheader("Destination Weights (page3)")
    page3_path = Path("page3.csv")
    if page3_path.exists():
        page3_df = load_data(str(page3_path))
        if page3_df.empty:
            st.info("No data available in page3.csv.")
        else:
            total_weight = sum(V3_CITY_COUNTS.values())
            weights_df = page3_df[["ToCity", "ToState"]].dropna().drop_duplicates().copy()
            weights_df["City"] = weights_df["ToCity"].astype(str).str.strip()
            weights_df["State"] = weights_df["ToState"].astype(str).str.strip()
            weights_df["Key"] = weights_df["City"].str.casefold() + ", " + weights_df["State"].str.casefold()
            weights_df["Weight"] = weights_df["Key"].map(V3_CITY_COUNTS).fillna(0.0)
            weights_df["Percent"] = (
                weights_df["Weight"] / total_weight * 100.0
            ) if total_weight else 0.0
            weights_df = weights_df[["City", "State", "Weight", "Percent"]].sort_values(
                ["Weight", "City", "State"],
                ascending=[False, True, True],
            )
            st.metric("Destinations", f"{len(weights_df):,}")
            st.dataframe(weights_df, use_container_width=True)
    else:
        st.info("page3.csv not found in the app folder.")
