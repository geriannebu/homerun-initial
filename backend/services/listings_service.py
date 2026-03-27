import numpy as np
import pandas as pd

from backend.utils.constants import TOWNS
from backend.utils.scoring import classify_listing
from backend.schemas.inputs import UserInputs


def mock_active_listings(inputs: UserInputs) -> pd.DataFrame:
    base = {
        "2 ROOM": 320000,
        "3 ROOM": 410000,
        "4 ROOM": 560000,
        "5 ROOM": 700000,
        "EXECUTIVE": 850000,
    }

    mul = 1.0
    if inputs.town in {"Bishan", "Queenstown", "Bukit Merah", "Kallang/Whampoa", "Toa Payoh"}:
        mul = 1.18
    elif inputs.town in {"Yishun", "Woodlands", "Jurong West", "Choa Chu Kang", "Sembawang"}:
        mul = 0.92

    pred = (
        base[inputs.flat_type] * mul
        + max(0, (inputs.floor_area_sqm - 70) * 3800)
        - max(0, (2026 - inputs.lease_commence_year) * 2500)
        + 70000 * sum(inputs.amenity_weights.values()) / len(inputs.amenity_weights)
        + min(len(inputs.landmark_postals), 2) * 15000
    )
    pred = max(pred, 180000)

    town_pool = [inputs.town] if inputs.town else list(np.random.choice(TOWNS, 5, replace=False))

    rows = []
    for i in range(8):
        town = str(np.random.choice(town_pool))
        asking = pred * np.random.uniform(0.88, 1.18)

        rows.append({
            "listing_id": f"LST-{1000+i}",
            "town": town,
            "flat_type": inputs.flat_type,
            "floor_area_sqm": round(inputs.floor_area_sqm + np.random.uniform(-8, 8), 1),
            "storey_range": str(np.random.choice(["04 TO 06", "07 TO 09", "10 TO 12", "13 TO 15"])),
            "asking_price": round(asking),
            "predicted_price": round(pred * np.random.uniform(0.97, 1.03)),
            "recent_median_transacted": round(pred * np.random.uniform(0.94, 1.06)),
            "listing_url": f"https://example-property-site.com/listing/{1000+i}",
        })

    df = pd.DataFrame(rows)
    df["asking_vs_predicted_pct"] = (
        (df["asking_price"] - df["predicted_price"]) / df["predicted_price"] * 100
    ).round(1)
    df["valuation_label"] = df.apply(classify_listing, axis=1)

    return df