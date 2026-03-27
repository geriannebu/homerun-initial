from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class UserInputs:
    budget: int
    flat_type: str
    floor_area_sqm: float
    remaining_lease_years: int          # NEW: replaces lease_commence_year
    town: Optional[str]
    school_scope: str
    amenity_weights: Dict[str, float]   # normalised 0-1 weights derived from rank
    amenity_rank: List[str]             # NEW: hard-ranked list, index 0 = top priority
    landmark_postals: List[str]

    @property
    def lease_commence_year(self) -> int:
        """Back-compat shim so existing backend services still work."""
        return max(1966, 2025 - self.remaining_lease_years)

