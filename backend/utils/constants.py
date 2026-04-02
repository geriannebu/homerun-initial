# ---------------------------------------------------------------------------
# Towns & Flat Types
# ---------------------------------------------------------------------------
TOWNS = [
    "Ang Mo Kio", "Bedok", "Bishan", "Bukit Batok", "Bukit Merah",
    "Choa Chu Kang", "Clementi", "Hougang", "Jurong West",
    "Kallang/Whampoa", "Pasir Ris", "Queenstown", "Sembawang",
    "Sengkang", "Tampines", "Toa Payoh", "Woodlands", "Yishun",
]

FLAT_TYPES = ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"]

# ---------------------------------------------------------------------------
# School Options
# ---------------------------------------------------------------------------
SCHOOL_OPTIONS = ["Any", "Primary schools", "Secondary schools", "Both"]

# ---------------------------------------------------------------------------
# Amenity Labels
# ---------------------------------------------------------------------------
AMENITY_LABELS = {
    "train": "MRT stations",
    "bus": "Bus stops",
    "polyclinic": "Hospitals / polyclinics",
    "primary_school": "Schools",
    "hawker": "Hawker centres",
    "mall": "Shopping malls",
    "supermarket": "Supermarkets",
}

# ---------------------------------------------------------------------------
# Amenity Colors (RGBA)
# ---------------------------------------------------------------------------
AMENITY_COLORS = {
    "town": [31, 119, 180, 180],
    "anchor": [148, 103, 189, 220],
    "train": [214, 39, 40, 160],
    "bus": [255, 127, 14, 140],
    "healthcare": [44, 160, 44, 150],      # polyclinics
    "schools": [23, 190, 207, 150],        # primary schools
    "hawker": [140, 86, 75, 150],
    "retail": [227, 119, 194, 150],        # malls
    "supermarket": [255, 215, 0, 150],     # supermarkets
}

# ---------------------------------------------------------------------------
# Amenity Keys (for scoring / iteration)
# ---------------------------------------------------------------------------
AMENITY_KEYS = [
    "train",
    "bus",
    "primary_school",
    "hawker",
    "mall",
    "polyclinic",
    "supermarket",
]

# ---------------------------------------------------------------------------
# Town Coordinates
# ---------------------------------------------------------------------------
TOWN_COORDS = {
    "Ang Mo Kio": (1.3691, 103.8454),
    "Bedok": (1.3236, 103.9273),
    "Bishan": (1.3500, 103.8485),
    "Bukit Batok": (1.3496, 103.7528),
    "Bukit Merah": (1.2773, 103.8195),
    "Choa Chu Kang": (1.3854, 103.7443),
    "Clementi": (1.3151, 103.7651),
    "Hougang": (1.3612, 103.8863),
    "Jurong West": (1.3404, 103.7068),
    "Kallang/Whampoa": (1.3123, 103.8660),
    "Pasir Ris": (1.3731, 103.9493),
    "Queenstown": (1.2942, 103.7860),
    "Sembawang": (1.4491, 103.8201),
    "Sengkang": (1.3909, 103.8952),
    "Tampines": (1.3496, 103.9568),
    "Toa Payoh": (1.3343, 103.8563),
    "Woodlands": (1.4360, 103.7865),
    "Yishun": (1.4294, 103.8354),
}

# ---------------------------------------------------------------------------
# Amenity Mapping (to column prefixes / scoring)
# ---------------------------------------------------------------------------
AMENITY_MAPPING = {
    "train":          "train",          # walk_train_min1, walk_train_avg_mins
    "bus":            "bus",            # walk_bus_min1, walk_bus_avg_mins
    "polyclinic":     "polyclinic",     # walk_polyclinic_min1, walk_polyclinic_avg_mins
    "primary_school": "primary_school", # walk_primary_school_min1, walk_primary_school_avg_mins
    "hawker":         "hawker",         # walk_hawker_min1, walk_hawker_avg_mins
    "mall":           "mall",           # walk_mall_min1, walk_mall_avg_mins
    "supermarket":    "supermarket",    # walk_supermarket_min1, walk_supermarket_avg_mins
}