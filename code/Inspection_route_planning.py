"""]
BONUS 5 - Optimized Route using Greedy Nearest Neighbor

Description:
It automatically identifies
the facility type from the "DatasetEN" column, converts coordinates to
the Hong Kong 1980 Grid System, and generates an optimized inspection
route using the Greedy Nearest Neighbor algorithm with local grid distance.
"""

import os
import csv
import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple

import geopy.distance as geo   # Used only for initial radius filtering


# ====================== CORE CLASSES ======================

class FacilityType(Enum):
    """Enumeration of all possible facility types based on DatasetEN field."""
    HOSPITAL = "Hospital Authority Hospital/Institution List"
    CLINIC_DH = "Clinics / Health Centres under Department of Health"
    CLINIC_CAP343 = "Clinics registered under Cap 343"
    DAY_CARE_elderly = "Day Care Centres for the Elderly"
    RESIDENTIAL_CARE_elderly = "Location of Residential Care Homes for the Elderly in Hong Kong"
    PRIVATE_CAP633 = "Private healthcare facilities under Cap 633"
    UNKNOWN = "Unknown"

    @classmethod
    def from_string(cls, s: str):
        """Convert DatasetEN string to corresponding FacilityType enum."""
        s = s.strip()
        for member in cls:
            if member.value == s:
                return member
        return cls.UNKNOWN


@dataclass
class FacilityData:
    """Structured data container for one facility record."""
    name: str
    latitude: float
    longitude: float
    dataset_type: FacilityType
    original_row: dict
    easting: Optional[float] = None      # HK1980 Easting
    northing: Optional[float] = None     # HK1980 Northing


# ====================== MAIN FACILITY CLASS ======================

class facility:
    """Main class responsible for loading facility data and computing optimized route."""

    # EPSG Codes
    WGS84_EPSG = 4326
    HK1980_GRID_EPSG = 2326

    def __init__(self, input_file: str):
        """Initialize by loading CSV data and preparing coordinate systems."""
        self.file_path = input_file
        self.facilities: List[FacilityData] = []
        self.dataset_name: Optional[str] = None
        self.facility_type: FacilityType = FacilityType.UNKNOWN

        self.transformer_to_hk = None
        self.transformer_to_wgs84 = None

        self._init_transformers()
        self._load_data(input_file)
        self._add_hk1980_coordinates()

    def _init_transformers(self):
        """Set up pyproj transformers for conversion between WGS84 and HK1980 Grid."""
        try:
            from pyproj import Transformer
            self.transformer_to_hk = Transformer.from_crs(
                f"EPSG:{self.WGS84_EPSG}",
                f"EPSG:{self.HK1980_GRID_EPSG}",
                always_xy=True)
            self.transformer_to_wgs84 = Transformer.from_crs(
                f"EPSG:{self.HK1980_GRID_EPSG}",
                f"EPSG:{self.WGS84_EPSG}",
                always_xy=True)
            print("Coordinate transformation system (HK1980 Grid) initialized successfully.")
        except ImportError:
            print("Warning: pyproj library not installed. HK1980 conversion disabled.")
        except Exception as e:
            print(f"Warning: Transformer initialization failed: {e}")

    def _load_data(self, input_file: str):
        """Load facility records from CSV and automatically detect type from DatasetEN."""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File not found: {input_file}")

        with open(input_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            dataset_types_set = set()

            for row in reader:
                try:
                    lat = float(row['Latitude'])
                    lon = float(row['Longitude'])
                except (ValueError, KeyError):
                    continue

                dtype_str = row.get('DatasetEN', '').strip()
                dataset_types_set.add(dtype_str)

                fac_data = FacilityData(
                    name=row.get('NameEN', '').strip(),
                    latitude=lat,
                    longitude=lon,
                    dataset_type=FacilityType.from_string(dtype_str),
                    original_row=dict(row)
                )
                self.facilities.append(fac_data)

        # Automatically determine facility type from DatasetEN
        self.dataset_name = next(iter(dataset_types_set)) if len(dataset_types_set) == 1 else "Facilities"
        self.facility_type = FacilityType.from_string(self.dataset_name)
        print(f"Successfully loaded {len(self.facilities)} facilities.")
        print(f"Facility Type being searched: {self.facility_type.name}\n")

    def _add_hk1980_coordinates(self):
        """Pre-convert all facility locations to HK1980 Grid for accurate distance calculation."""
        if not self.transformer_to_hk:
            print("HK1980 Grid conversion skipped.")
            return

        print("Converting all coordinates to HK1980 Grid system...")
        for fac in self.facilities:
            try:
                fac.easting, fac.northing = self.transformer_to_hk.transform(
                    fac.longitude, fac.latitude)
            except:
                fac.easting = fac.northing = None

    def _wgs84_to_hk1980(self, longitude: float, latitude: float) -> Tuple[Optional[float], Optional[float]]:
        """Convert a WGS84 coordinate pair to HK1980 Grid (Easting, Northing)."""
        if self.transformer_to_hk:
            try:
                return self.transformer_to_hk.transform(longitude, latitude)
            except:
                return None, None
        return None, None

    def _hk1980_to_wgs84(self, easting: float, northing: float) -> Tuple[Optional[float], Optional[float]]:
        """Convert HK1980 Grid coordinates back to WGS84."""
        if self.transformer_to_wgs84:
            try:
                return self.transformer_to_wgs84.transform(easting, northing)
            except:
                return None, None
        return None, None

    def _euclidean_distance_hk80(self, e1, n1, e2, n2) -> float:
        """Calculate straight-line distance using HK1980 Grid (most accurate for HongK)."""
        if None in (e1, n1, e2, n2):
            return float('inf')
        return math.sqrt((e1 - e2) ** 2 + (n1 - n2) ** 2)

    # ====================== BONUS 5: OPTIMIZED ROUTE ======================
    def find_optimized_route(self, start_lat=None, start_lng=None,
                             start_easting=None, start_northing=None,
                             radius_meters=2000):
        """
        BONUS 5: Greedy Nearest Neighbor Algorithm.

        This function creates an optimized inspection route by repeatedly
        selecting the nearest unvisited facility within the given radius.
        Distance is calculated using HK1980 Grid for local accuracy.
        """
        print(f"\n=== Optimized Route for: {self.facility_type.name} ===")

        if not self.facilities:
            print("No facility data available.")
            return None

        # Determine starting point
        if start_lat is not None and start_lng is not None:
            start_wgs84 = (start_lat, start_lng)
            start_e, start_n = self._wgs84_to_hk1980(start_lng, start_lat)
        elif start_easting is not None and start_northing is not None:
            start_e, start_n = start_easting, start_northing
            lng, lat = self._hk1980_to_wgs84(start_easting, start_northing)
            start_wgs84 = (lat, lng)
        else:
            print("Error: Please provide starting coordinates.")
            return None

        # Filter facilities within radius
        candidates = [fac for fac in self.facilities
                      if geo.great_circle(start_wgs84, (fac.latitude, fac.longitude)).meters <= radius_meters]

        if not candidates:
            print(f"No {self.facility_type.name} found within {radius_meters} meters.")
            return None

        # Greedy Nearest Neighbor using HK1980 Euclidean distance
        n = len(candidates)
        visited = [False] * n
        route = []
        current_e = start_e
        current_n = start_n
        total_dist = 0.0

        while len(route) < n:
            best_idx = -1
            best_dist = float('inf')
            for i, fac in enumerate(candidates):
                if not visited[i]:
                    d = self._euclidean_distance_hk80(current_e, current_n,
                                                      fac.easting, fac.northing)
                    if d < best_dist:
                        best_dist = d
                        best_idx = i
            if best_idx == -1:
                break

            visited[best_idx] = True
            route.append(candidates[best_idx])
            total_dist += best_dist
            current_e = candidates[best_idx].easting
            current_n = candidates[best_idx].northing

        # Final output
        print(f"\nTotal facilities visited: {len(route)}")
        print(f"Total route distance (HK1980 Grid): {total_dist:.2f} meters")
        print("\nOptimal Visit Order:")
        for idx, fac in enumerate(route, 1):
            print(f"{idx}. {fac.name} [{fac.dataset_type.value}]")

        return {
            'total_facilities': len(route),
            'total_route_distance_meters': total_dist,
            'route': route
        }


# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    #  path
    CSV_FILE = r"C:\Users\hosiu\Downloads\ElderlyDayCare.csv"

    print("=== BONUS 5: Optimized Inspection Route (Greedy Nearest Neighbor) ===\n")

    fac_obj = facility(CSV_FILE)

    print("\nEnter starting location for inspection route:")
    choice = input("1 = WGS84 (Latitude, Longitude) | 2 = HK1980 (Easting, Northing) → ").strip()

    if choice == '1':
        lat = float(input("Latitude  : "))
        lng = float(input("Longitude : "))
        east = north = None
    else:
        east = float(input("Easting   : "))
        north = float(input("Northing  : "))
        lat = lng = None

    radius = float(input("\nEnter search radius (meters, e.g. 2000): "))

    fac_obj.find_optimized_route(
        start_lat=lat, start_lng=lng,
        start_easting=east, start_northing=north,
        radius_meters=radius
    )