# -*- coding: utf-8 -*-
"""
Description: A class to handle healthcare/elderly care facility data for Task 2.
             Implements:
             - __init__ with file reading and data validation
             - create_feature_class (adds ALL original fields to ArcGIS point feature class)
             - find_nearest (single nearest, supports lat/lng or HK1980 Grid) with enhanced output
             - FacilityType enum for type safety
             - FacilityData dataclass for structured storage

Methodology & Design Choices:
1. Coordinate Systems:
   - Input CSV uses WGS84 (latitude/longitude) for global compatibility.
   - Hong Kong planning authorities use HK1980 Grid (EPSG:2326) for local distance calculations.
   - The class converts between both systems using pyproj, enabling Euclidean distance (meters) on the projected grid.
2. Distance Metrics:
   - Great-circle distance (geopy) for WGS84 – accounts for Earth's curvature, suitable for geographic coordinates.
   - Euclidean distance (Pythagorean) for HK1980 Grid – valid because the grid is a projected coordinate system (meters).
3. File Reading Robustness:
   - Supports multiple encodings (UTF-8 with BOM, cp1252, latin-1) to handle various CSV exports.
   - Auto-detects comma or tab delimiters.
4. ArcGIS Integration:
   - Creates a point feature class preserving ALL original CSV attributes (no data loss).
   - Optionally exports the nearest facility as a feature class for further spatial analysis.
5. Enhanced Output for `find_nearest`:
   - Returns a detailed dictionary containing input coordinates (both systems), facility attributes, and two distance measures.
   - Heverify correctness and provides complete metadata for reporting.
"""

import os
import csv
import re
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple

import arcpy
from arcpy import env
import geopy.distance as geo

# pyproj is optional but recommended for accurate HK1980 conversions.
try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    print("Warning: pyproj not installed. HK1980 Grid coordinates will be set to None.")


# -------------------------------------------------------------------
# Enum to define facility types
# -------------------------------------------------------------------
class FacilityType(Enum):
    """
    Enumeration of known facility types from the 'DatasetEN' column.
    Using an enum provides type safety and prevents invalid string values.
    The from_string() method allows graceful fallback to UNKNOWN.
    """
    HOSPITAL = "Hospital Authority Hospital/Institution List"
    CLINIC_DH = "Clinics / Health Centres under Department of Health"
    CLINIC_CAP343 = "Clinics registered under Cap 343"
    DAY_CARE_elderly = "Day Care Centres for the Elderly"
    RESIDENTIAL_CARE = "Location of Residential Care Homes for the Elderly in Hong Kong"
    PRIVATE_CAP633 = "Private healthcare facilities under Cap 633"
    UNKNOWN = "Unknown"

    @classmethod
    def from_string(cls, s: str) -> "FacilityType":
        """Convert a string (e.g., from CSV) to the corresponding enum member."""
        s = s.strip()
        for member in cls:
            if member.value == s:
                return member
        return cls.UNKNOWN


# -------------------------------------------------------------------
# Dataclass to store facility information
# -------------------------------------------------------------------
@dataclass
class FacilityData:
    """
    Structured representation of a single facility.
    Using a dataclass reduces boilerplate code and makes the data
    self-documenting. All original CSV fields are preserved in 'original_row'
    for later export to a feature class.
    """
    name: str
    latitude: float
    longitude: float
    easting: Optional[float]   # HK1980 Grid Easting (meters)
    northing: Optional[float]  # HK1980 Grid Northing (meters)
    dataset_type: FacilityType
    original_row: Dict[str, str]  # complete original CSV row


# -------------------------------------------------------------------
# Main facility class
# -------------------------------------------------------------------
class facility:
    """
    A class representing a collection of healthcare/elderly care facilities.
    Provides methods to load data from CSV, create an ArcGIS feature class,
    and find the nearest facility to a user-specified point.
    """

    # Class constants for EPSG codes (standardised coordinate reference systems)
    WGS84_EPSG = 4326          # WGS84 Geographic Coordinate System (lat/lon in degrees)
    HK1980_GRID_EPSG = 2326    # Hong Kong 1980 Grid System (projected, units = meters)

    @staticmethod
    def sanitize_name(name: str) -> str:
        """
        Convert a string to a valid ArcGIS feature class name.
        ArcGIS does not allow spaces, parentheses, hyphens, or leading digits.
        This method replaces problematic characters with underscores and
        prefixes 'Fac_' if the name starts with a digit.
        """
        name = name.strip()
        # Replace spaces, slashes, hyphens, parentheses with underscores
        name = re.sub(r'[\s/\-\(\)]+', '_', name)
        # Remove any remaining non-alphanumeric characters (except underscore)
        name = re.sub(r'[^a-zA-Z0-9_]', '', name)
        # Prefix if first character is a digit
        if name and name[0].isdigit():
            name = 'Fac_' + name
        # Truncate to 100 characters (ArcGIS limit)
        if len(name) > 100:
            name = name[:100]
        return name

    def __init__(self, input_file: str):
        """Load and validate facility data from a CSV file."""
        self.file_path = input_file
        self.facilities: List[FacilityData] = []
        self.field_names: List[str] = []
        self.dataset_name: Optional[str] = None
        self.facility_type: FacilityType = FacilityType.UNKNOWN

        # Initialise coordinate converters (pyproj)
        self._init_transformers()

        # Validate file existence
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File not found: {input_file}")

        # --- Handle different file encodings (common issue with downloaded CSVs) ---
        encodings_to_try = ['utf-8-sig', 'cp1252', 'latin-1']
        file_handle = None
        for enc in encodings_to_try:
            try:
                file_handle = open(input_file, 'r', encoding=enc)
                file_handle.read(1024)  # test read
                file_handle.seek(0)
                print(f"Successfully opened {input_file} with encoding: {enc}")
                break
            except UnicodeDecodeError:
                continue
        if file_handle is None:
            raise RuntimeError(f"Could not decode file with any of {encodings_to_try}")

        # --- Read CSV and parse rows ---
        try:
            with file_handle as f:
                # Auto-detect delimiter (comma or tab) by inspecting first line
                first_line = f.readline()
                delimiter = ',' if ',' in first_line else '\t'
                f.seek(0)
                reader = csv.DictReader(f, delimiter=delimiter)
                self.field_names = reader.fieldnames

                # Validate required columns
                required = {'DatasetEN', 'NameEN', 'Latitude', 'Longitude'}
                if not required.issubset(set(self.field_names)):
                    missing = required - set(self.field_names)
                    raise ValueError(f"Missing required columns: {missing}")

                dataset_types_set = set()
                for row in reader:
                    # Extract and validate coordinates
                    try:
                        lat = float(row['Latitude'])
                        lon = float(row['Longitude'])
                    except ValueError:
                        print(f"Skipping row with invalid coordinates: {row}")
                        continue

                    dtype_str = row['DatasetEN'].strip()
                    dataset_types_set.add(dtype_str)

                    # Convert WGS84 → HK1980 Grid (for Euclidean distance)
                    easting, northing = self._wgs84_to_hk1980(lon, lat)

                    fac_data = FacilityData(
                        name=row['NameEN'].strip(),
                        latitude=lat,
                        longitude=lon,
                        easting=easting,
                        northing=northing,
                        dataset_type=FacilityType.from_string(dtype_str),
                        original_row=dict(row)
                    )
                    self.facilities.append(fac_data)

            # Determine dataset name (if multiple types exist, use the most common)
            if len(dataset_types_set) == 1:
                self.dataset_name = next(iter(dataset_types_set))
            elif len(dataset_types_set) > 1:
                from collections import Counter
                self.dataset_name = Counter(dataset_types_set).most_common(1)[0][0]
                print(f"Warning: Multiple DatasetEN values found. Using '{self.dataset_name}'.")
            else:
                self.dataset_name = os.path.splitext(os.path.basename(input_file))[0]

            self.facility_type = FacilityType.from_string(self.dataset_name)
            print(f"Successfully loaded {len(self.facilities)} facilities from {input_file}")
            print(f"Dataset name: {self.dataset_name} (type: {self.facility_type.name})")

        except Exception as e:
            raise RuntimeError(f"Error reading file {input_file}: {e}")

    def _init_transformers(self):
        """Initialise pyproj transformers for WGS84 ↔ HK1980 Grid conversions."""
        if PYPROJ_AVAILABLE:
            try:
                # Transformer for WGS84 (lon,lat) → HK1980 Grid (easting,northing)
                self.transformer_to_hk = Transformer.from_crs(
                    f"EPSG:{self.WGS84_EPSG}",
                    f"EPSG:{self.HK1980_GRID_EPSG}",
                    always_xy=True
                )
                # Transformer for HK1980 Grid → WGS84
                self.transformer_to_wgs84 = Transformer.from_crs(
                    f"EPSG:{self.HK1980_GRID_EPSG}",
                    f"EPSG:{self.WGS84_EPSG}",
                    always_xy=True
                )
                print("Coordinate transformers initialized (pyproj)")
            except Exception as e:
                print(f"Warning: Failed to initialize transformers: {e}")
                self.transformer_to_hk = None
                self.transformer_to_wgs84 = None
        else:
            self.transformer_to_hk = None
            self.transformer_to_wgs84 = None

    def _wgs84_to_hk1980(self, longitude: float, latitude: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Convert WGS84 (longitude, latitude) to HK1980 Grid (easting, northing).
        Returns (None, None) if conversion fails or pyproj is unavailable.
        """
        if self.transformer_to_hk:
            try:
                easting, northing = self.transformer_to_hk.transform(longitude, latitude)
                return easting, northing
            except Exception as e:
                print(f"Conversion error ({longitude}, {latitude}): {e}")
                return None, None
        return None, None

    def _hk1980_to_wgs84(self, easting: float, northing: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Convert HK1980 Grid (easting, northing) to WGS84 (longitude, latitude).
        Returns (None, None) if conversion fails.
        """
        if self.transformer_to_wgs84:
            try:
                lon, lat = self.transformer_to_wgs84.transform(easting, northing)
                return lon, lat
            except Exception as e:
                print(f"Conversion error (easting={easting}, northing={northing}): {e}")
                return None, None
        return None, None

    def _euclidean_distance_hk80(self, easting1: float, northing1: float,
                                 easting2: float, northing2: float) -> float:
        """
        Euclidean distance in meters between two points in the HK1980 Grid.
        This is valid because HK1980 is a projected coordinate system with metric units.
        """
        return math.sqrt((easting1 - easting2) ** 2 + (northing1 - northing2) ** 2)

    # -------------------------------------------------------------------
    # Essential method: Create feature class with ALL original fields
    # -------------------------------------------------------------------
    def create_feature_class(self, gdb_path: str, feature_class_name: str = None) -> Optional[str]:
        """
        Convert the facility data into a point feature class inside an ArcGIS geodatabase.
        Adds ALL attribute fields from the original CSV file – no data loss.
        This method is intended for visualisation and further spatial analysis in ArcGIS Pro.

        Parameters:
            gdb_path : path to the file geodatabase (will be created if missing)
            feature_class_name : optional name; defaults to sanitised dataset name

        Returns:
            Full path to the created feature class, or None on failure.
        """
        try:
            # Create geodatabase if it does not exist
            if not arcpy.Exists(gdb_path):
                gdb_dir = os.path.dirname(gdb_path)
                gdb_name = os.path.basename(gdb_path)
                if not gdb_dir:
                    gdb_dir = "."
                arcpy.management.CreateFileGDB(gdb_dir, gdb_name)
                print(f"Created geodatabase: {gdb_path}")

            env.workspace = gdb_path
            env.overwriteOutput = True

            # Determine feature class name
            if feature_class_name is None:
                raw_name = self.dataset_name if self.dataset_name else os.path.splitext(os.path.basename(self.file_path))[0]
                fc_name = self.sanitize_name(raw_name)
            else:
                fc_name = self.sanitize_name(feature_class_name)

            full_path = os.path.join(gdb_path, fc_name)
            if arcpy.Exists(full_path):
                arcpy.management.Delete(full_path)
                print(f"Existing feature class deleted: {full_path}")

            # Create an empty point feature class with WGS84 spatial reference
            sr = arcpy.SpatialReference(self.WGS84_EPSG)
            arcpy.management.CreateFeatureclass(
                out_path=gdb_path,
                out_name=fc_name,
                geometry_type="POINT",
                spatial_reference=sr,
                has_m="DISABLED",
                has_z="DISABLED"
            )
            print(f"Empty feature class created: {full_path}")

            # Add fields for every column in the original CSV
            for field_name in self.field_names:
                # Infer field type from the first non-null value
                sample = None
                for fac in self.facilities:
                    val = fac.original_row.get(field_name)
                    if val:
                        sample = val
                        break
                if sample is None:
                    field_type = "TEXT"
                else:
                    try:
                        float(sample)
                        field_type = "DOUBLE"
                    except ValueError:
                        field_type = "TEXT"

                # Clean field name for ArcGIS (no special characters, no leading digits)
                clean_field = re.sub(r'[^\w]', '_', field_name)
                if clean_field[0].isdigit():
                    clean_field = "Field_" + clean_field

                try:
                    if field_type == "TEXT":
                        arcpy.management.AddField(full_path, clean_field, "TEXT", field_length=255)
                    else:
                        arcpy.management.AddField(full_path, clean_field, "DOUBLE")
                except Exception as e:
                    print(f"Could not add field {clean_field}: {e}")

            # Add two additional fields for HK1980 coordinates (to support Euclidean distance queries)
            arcpy.management.AddField(full_path, "Easting_HK80", "DOUBLE")
            arcpy.management.AddField(full_path, "Northing_HK80", "DOUBLE")

            # Build the list of field names for the InsertCursor
            field_names_for_cursor = ["SHAPE@"]
            for fname in self.field_names:
                clean = re.sub(r'[^\w]', '_', fname)
                if clean[0].isdigit():
                    clean = "Field_" + clean
                field_names_for_cursor.append(clean)
            field_names_for_cursor.append("Easting_HK80")
            field_names_for_cursor.append("Northing_HK80")

            # Insert each facility as a point geometry with all attributes
            with arcpy.da.InsertCursor(full_path, field_names_for_cursor) as cursor:
                for fac in self.facilities:
                    point = arcpy.Point(fac.longitude, fac.latitude)
                    point_geom = arcpy.PointGeometry(point, sr)
                    row = [point_geom]
                    # Add each original CSV field value (convert to float if possible)
                    for fname in self.field_names:
                        val = fac.original_row.get(fname, "")
                        if val and isinstance(val, str):
                            try:
                                val = float(val)
                            except ValueError:
                                pass
                        row.append(val)
                    row.append(fac.easting)
                    row.append(fac.northing)
                    cursor.insertRow(row)

            print(f"Successfully inserted {len(self.facilities)} points into {full_path}")
            return full_path

        except Exception as e:
            print(f"Error creating feature class: {e}")
            return None

    # -------------------------------------------------------------------
    # Essential method: Find nearest facility
    # -------------------------------------------------------------------
    def find_nearest(self,
                     target_lat: float = None,
                     target_lng: float = None,
                     target_easting: float = None,
                     target_northing: float = None,
                     return_as: str = 'object') -> Union[Dict, str, None]:
        """
        Find the single nearest facility to the given target point.
        Supports input in either WGS84 (lat/lon) or HK1980 Grid (easting/northing).

        Distance calculation:
        - Great-circle distance (geopy) is used for accuracy on the WGS84 ellipsoid.
        - Euclidean distance (HK1980 Grid) is also computed for verification and
          because local planning data often uses projected coordinates.

        Enhanced output returns a dictionary with:
          - Input coordinates (both systems)
          - Complete information about the nearest facility
          - Both distance metrics (meters)

        If return_as = 'feature_class', a temporary point feature class is created
        containing the nearest facility (useful for mapping).

        Parameters:
            target_lat, target_lng : WGS84 coordinates (degrees)
            target_easting, target_northing : HK1980 Grid coordinates (meters)
            return_as : 'object' (dictionary) or 'feature_class' (path string)

        Returns:
            Dictionary or feature class path, or None if no facilities or invalid input.
        """
        if not self.facilities:
            print("No facility data available.")
            return None

        # --- Step 1: Convert input to both WGS84 and HK1980 ---
        if target_lat is not None and target_lng is not None:
            target_wgs84 = (target_lat, target_lng)
            easting, northing = self._wgs84_to_hk1980(target_lng, target_lat)
            target_hk80 = (easting, northing)
        elif target_easting is not None and target_northing is not None:
            lng, lat = self._hk1980_to_wgs84(target_easting, target_northing)
            if lng is None or lat is None:
                print("Failed to convert HK1980 coordinates to WGS84.")
                return None
            target_wgs84 = (lat, lng)
            target_hk80 = (target_easting, target_northing)
        else:
            print("Provide either (lat, lng) or (easting, northing).")
            return None

        # --- Step 2: Iterate all facilities to find the minimum distance ---
        best_fac = None
        min_dist_wgs84 = float('inf')
        min_dist_hk80 = float('inf')

        for fac in self.facilities:
            # Great-circle distance (accurate for geographic coordinates)
            dist_wgs84 = geo.great_circle(target_wgs84, (fac.latitude, fac.longitude)).meters

            # Euclidean distance if HK1980 coordinates are available
            if fac.easting is not None and target_hk80[0] is not None:
                dist_hk80 = self._euclidean_distance_hk80(target_hk80[0], target_hk80[1],
                                                          fac.easting, fac.northing)
            else:
                dist_hk80 = float('inf')

            if dist_wgs84 < min_dist_wgs84:
                min_dist_wgs84 = dist_wgs84
                min_dist_hk80 = dist_hk80
                best_fac = fac

        if best_fac is None:
            print("No nearest facility found.")
            return None

        # --- Step 3: Return result as dictionary or feature class ---
        if return_as == 'object':
            result = {
                'input_location_latitude': target_wgs84[0],
                'input_location_longitude': target_wgs84[1],
                'input_location_easting': target_hk80[0] if target_hk80[0] is not None else 'N/A',
                'input_location_northing': target_hk80[1] if target_hk80[1] is not None else 'N/A',
                'facility_name': best_fac.name,
                'facility_type': self.facility_type.name,
                'dataset_type': best_fac.dataset_type.value,
                'facility_latitude': best_fac.latitude,
                'facility_longitude': best_fac.longitude,
                'facility_easting': best_fac.easting if best_fac.easting is not None else 'N/A',
                'facility_northing': best_fac.northing if best_fac.northing is not None else 'N/A',
                'distance_meters_great_circle': min_dist_wgs84,
                'distance_meters_euclidean_hk80': min_dist_hk80 if min_dist_hk80 != float('inf') else 'N/A'
            }
            return result
        else:
            # Create a temporary geodatabase and feature class containing only the nearest facility
            temp_dir = os.path.dirname(self.file_path) if self.file_path else os.getcwd()
            temp_gdb = os.path.join(temp_dir, "temp_nearest.gdb")
            if not arcpy.Exists(temp_gdb):
                gdb_dir = os.path.dirname(temp_gdb)
                gdb_name = os.path.basename(temp_gdb)
                arcpy.management.CreateFileGDB(gdb_dir, gdb_name)
            env.workspace = temp_gdb
            env.overwriteOutput = True
            sr = arcpy.SpatialReference(self.WGS84_EPSG)
            fc_name = f"nearest_{self.sanitize_name(self.dataset_name)}"
            fc_path = os.path.join(temp_gdb, fc_name)
            if arcpy.Exists(fc_path):
                arcpy.management.Delete(fc_path)
            arcpy.management.CreateFeatureclass(temp_gdb, fc_name, "POINT", spatial_reference=sr)
            # Add fields to store facility name and distances
            arcpy.management.AddField(fc_path, "Facility_Name", "TEXT", field_length=255)
            arcpy.management.AddField(fc_path, "Distance_WGS84_m", "DOUBLE")
            arcpy.management.AddField(fc_path, "Distance_HK80_m", "DOUBLE")
            arcpy.management.AddField(fc_path, "DatasetEN", "TEXT", field_length=100)
            point = arcpy.Point(best_fac.longitude, best_fac.latitude)
            point_geom = arcpy.PointGeometry(point, sr)
            with arcpy.da.InsertCursor(fc_path, ["SHAPE@", "Facility_Name", "Distance_WGS84_m", "Distance_HK80_m", "DatasetEN"]) as cursor:
                cursor.insertRow([point_geom, best_fac.name, min_dist_wgs84, min_dist_hk80, best_fac.dataset_type.value])
            return fc_path


# -------------------------------------------------------------------
# Interactive User Input Function
# -------------------------------------------------------------------
def get_target_coordinates_from_user():
    """
    Prompt the user to enter target coordinates either as WGS84 (lat/lon)
    or HK1980 Grid (easting/northing). Provides basic range validation for
    Hong Kong territory to catch common typos.
    Returns a tuple (lat, lng, easting, northing) where the unused pair is None.
    """
    print("\n" + "=" * 50)
    print("TARGET COORDINATE INPUT")
    print("=" * 50)
    print("Please select coordinate system:")
    print("1. WGS84 (Latitude, Longitude) - e.g., 22.304563, 114.179577")
    print("2. HK1980 Grid (Easting, Northing) - e.g., 832242, 816128")
    print("-" * 50)

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == '1':
            while True:
                try:
                    lat_input = input("Enter Latitude (e.g., 22.304563): ").strip()
                    lng_input = input("Enter Longitude (e.g., 114.179577): ").strip()
                    target_lat = float(lat_input)
                    target_lng = float(lng_input)
                    # Warn if outside Hong Kong approximate bounding box
                    if not (22.0 <= target_lat <= 22.6):
                        print("Warning: Latitude outside typical Hong Kong range (22.0-22.6)")
                    if not (113.7 <= target_lng <= 114.5):
                        print("Warning: Longitude outside typical Hong Kong range (113.7-114.5)")
                    print(f"\nTarget location (WGS84): Latitude={target_lat}, Longitude={target_lng}")
                    return (target_lat, target_lng, None, None)
                except ValueError:
                    print("Invalid input. Please enter numeric values.")
        elif choice == '2':
            while True:
                try:
                    easting_input = input("Enter Easting (e.g., 832242): ").strip()
                    northing_input = input("Enter Northing (e.g., 816128): ").strip()
                    target_easting = float(easting_input)
                    target_northing = float(northing_input)
                    # Range validation for HK1980 Grid covering Hong Kong territory
                    if not (799500 <= target_easting <= 867500):
                        print("Warning: Easting outside typical Hong Kong range (799500-867500)")
                    if not (799000 <= target_northing <= 848000):
                        print("Warning: Northing outside typical Hong Kong range (799000-848000)")
                    print(f"\nTarget location (HK1980 Grid): Easting={target_easting}, Northing={target_northing}")
                    return (None, None, target_easting, target_northing)
                except ValueError:
                    print("Invalid input. Please enter numeric values.")
        else:
            print("Invalid choice. Please enter 1 or 2.")


# -------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    """
    Demonstrate the usage of the facility class:
    1. Load facility data from a CSV file.
    2. Optionally create a feature class in a geodatabase (all original attributes).
    3. Ask the user for a target location and find the nearest facility.
    4. Print the enhanced result in a readable format.
    """
    # ========== CONFIGURATION ==========
    #paths
    CSV_FILE = r"C:\Users\hosiu\Downloads\ElderlyDayCare.csv"
    OUTPUT_GDB = r"C:\Users\hosiu\Downloads\Lab4_23079474D_Ho Siu Fai（good)\Lab4_ArcGIS pro_visualization_copy\task2_cloest.gdb"
    # ===================================


    try:
        print("Loading facility data...")
        fac_obj = facility(CSV_FILE)
        print(f"Loaded {len(fac_obj.facilities)} facilities.\n")
    except Exception as e:
        print(f"Failed to load file: {e}")
        exit(1)

    # 2. Create feature class (with ALL fields)
    if OUTPUT_GDB:
        create_fc = input("Do you want to create a feature class in the geodatabase? (y/n): ").strip().lower()
        if create_fc == 'y':
            print("\nCreating feature class...")
            result = fac_obj.create_feature_class(OUTPUT_GDB)
            if result:
                print(f"Feature class created at: {result}")
            else:
                print("Failed to create feature class.")
        else:
            print("Skipping feature class creation.\n")

    # 3. Find nearest facility (single) with enhanced output
    print("\n--- Find the single nearest facility ---")
    lat, lng, east, north = get_target_coordinates_from_user()
    nearest = fac_obj.find_nearest(target_lat=lat, target_lng=lng,
                                   target_easting=east, target_northing=north,
                                   return_as='object')
    if nearest:
        print("\n" + "=" * 60)
        print("NEAREST FACILITY RESULT")
        print("=" * 60)
        print(f"1. Input location latitude & longitude: {nearest['input_location_latitude']}, {nearest['input_location_longitude']}")
        print(f"2. Input location Northing & Easting: {nearest['input_location_northing']}, {nearest['input_location_easting']}")
        print(f"3. facility_name: {nearest['facility_name']}")
        print(f"4. facility_type: {nearest['facility_type']}")
        print(f"5. dataset_type: {nearest['dataset_type']}")
        print(f"6. facility_latitude: {nearest['facility_latitude']}")
        print(f"7. facility_longitude: {nearest['facility_longitude']}")
        print(f"8. facility_Northing: {nearest['facility_northing']}")
        print(f"9. facility_Easting: {nearest['facility_easting']}")
        print(f"10. distance_meters (great-circle / lat/lng): {nearest['distance_meters_great_circle']:.2f} m")
        print(f"11. distance_meters (Euclidean / Easting/Northing): {nearest['distance_meters_euclidean_hk80']} m")
    else:
        print("No facility found.")