# This script implements a multi-criteria decision analysis (MCDA) for optimal
# hospital site selection in a dense urban environment (Kwun Tong, Hong Kong).
# The analysis considers:
#   - Land use patterns (2021 HK land utilization data)
#   - Accessibility to public transport (bus stops; highest weight)
#   - Road network adjacency (ambulance and patient access)
#   - Distance from existing hospitals (avoid oversaturation)
#   - Terrain slope (degree) derived from a Digital Terrain Model (DTM)
#
# The output is a suitability map (1–10, where 10 is most suitable) and a
# final polygon feature class of optimal sites that are adjacent to roads and
# meet a minimum area threshold.
#
# Data sources are based on 2021 (population, land use, DTM, bus stops, roads).
# The methodology is validated against government planning blueprints (e.g., New Kai Tak Hospital).
# ----------------------------------------------------------------------------


import os
import csv
import arcpy
from arcpy.sa import *
from arcpy import env

# pyproj for accurate coordinate conversion from WGS84 (lat/lon) to HK1980 Grid
try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    print("Warning: pyproj not installed. Install with: pip install pyproj")
    exit(1)

env.overwriteOutput = True

# -------------------------------------------------------------------
# Helper: Convert WGS84 (lat/lon) to HK1980 Grid (easting/northing)
# -------------------------------------------------------------------
def wgs84_to_hk1980(lat, lon):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2326", always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return easting, northing

# -------------------------------------------------------------------
# 1. Environment Settings
# -------------------------------------------------------------------
def set_environment(workspace_gdb, dem_raster):
    if arcpy.CheckExtension("Spatial") == "Available":
        arcpy.CheckOutExtension("Spatial")
    else:
        raise Exception("Spatial Analyst license not available")
    if not arcpy.Exists(dem_raster):
        raise FileNotFoundError(f"DEM raster not found: {dem_raster}")

    env.workspace = workspace_gdb
    env.scratchWorkspace = workspace_gdb
    env.cellSize = dem_raster
    env.extent = dem_raster
    print("Environment set. Cell size and extent from DEM.")

# -------------------------------------------------------------------
# 2. Convert hospitals CSV to point feature class (HK1980 Grid)
# -------------------------------------------------------------------
def csv_hospitals_to_points(csv_path, output_fc):
    out_dir = os.path.dirname(output_fc)
    if not arcpy.Exists(out_dir):
        arcpy.management.CreateFileGDB(os.path.dirname(out_dir), os.path.basename(out_dir))

    sr = arcpy.SpatialReference(2326)  # HK1980 Grid
    arcpy.management.CreateFeatureclass(
        os.path.dirname(output_fc),
        os.path.basename(output_fc),
        "POINT",
        spatial_reference=sr
    )
    # Store original coordinates and transformed grid coordinates for verification
    arcpy.management.AddField(output_fc, "Latitude", "DOUBLE")
    arcpy.management.AddField(output_fc, "Longitude", "DOUBLE")
    arcpy.management.AddField(output_fc, "Easting", "DOUBLE")
    arcpy.management.AddField(output_fc, "Northing", "DOUBLE")
    arcpy.management.AddField(output_fc, "Name", "TEXT", field_length=255)

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        lat_col = next((c for c in reader.fieldnames if 'lat' in c.lower()), 'Latitude')
        lon_col = next((c for c in reader.fieldnames if 'lon' in c.lower() or 'lng' in c.lower()), 'Longitude')
        name_col = next((c for c in reader.fieldnames if 'name' in c.lower()), None)

        with arcpy.da.InsertCursor(output_fc, ["SHAPE@", "Latitude", "Longitude", "Easting", "Northing", "Name"]) as cursor:
            for row in reader:
                try:
                    lat = float(row[lat_col])
                    lon = float(row[lon_col])
                except:
                    continue
                easting, northing = wgs84_to_hk1980(lat, lon)
                point = arcpy.Point(easting, northing)
                point_geom = arcpy.PointGeometry(point, sr)
                name = row[name_col] if name_col and name_col in row else ""
                cursor.insertRow([point_geom, lat, lon, easting, northing, name])
    count = int(arcpy.management.GetCount(output_fc)[0])
    print(f"Hospital points created: {output_fc} with {count} features.")
# Automatically detect column names (case-insensitive)
# -------------------------------------------------------------------
# 3. Derive slope from DTM
# -------------------------------------------------------------------
def derive_slope(dem, output_slope):
    slope_raster = Slope(dem, "DEGREE", 0.3048)
    slope_raster.save(output_slope)
    print("Slope derived.")

# -------------------------------------------------------------------
# 4. Distance to bus stops
# -------------------------------------------------------------------
def distance_to_bus_stops(bus_stops_fc, output_distance):
    dist_raster = EucDistance(bus_stops_fc)
    dist_raster.save(output_distance)
    print("Distance to bus stops calculated.")

# -------------------------------------------------------------------
# 5. Distance to existing hospitals
# -------------------------------------------------------------------
def distance_to_hospitals(hospitals_fc, output_distance):
    dist_raster = EucDistance(hospitals_fc)
    dist_raster.save(output_distance)
    print("Distance to existing hospitals calculated.")

# -------------------------------------------------------------------
# 6. Reclassification helpers(for slicing and reverse ranking)
# -------------------------------------------------------------------
def create_reverse_remap(levels):
    remap_list = [[i+1, levels - i] for i in range(levels)]
    remap_list.append(["NODATA", "NODATA"])
    return RemapValue(remap_list)

def create_direct_remap(levels):
    remap_list = [[i+1, i+1] for i in range(levels)]
    remap_list.append(["NODATA", "NODATA"])
    return RemapValue(remap_list)

# -------------------------------------------------------------------
# 7. Reclassify slope, bus distance&hospital distance
# -------------------------------------------------------------------
def reclassify_criteria(slope_raster, dist_bus_raster, dist_hospitals_raster,
                        output_slope_reclass, output_bus_reclass, output_hospitals_reclass, levels=10):
    slice_slope = Slice(slope_raster, levels, "EQUAL_INTERVAL")
    reclass_slope = Reclassify(slice_slope, "Value", create_reverse_remap(levels))
    reclass_slope.save(output_slope_reclass)

    slice_bus = Slice(dist_bus_raster, levels, "EQUAL_INTERVAL")
    reclass_bus = Reclassify(slice_bus, "Value", create_reverse_remap(levels))
    reclass_bus.save(output_bus_reclass)

    slice_hosp = Slice(dist_hospitals_raster, levels, "EQUAL_INTERVAL")
    reclass_hosp = Reclassify(slice_hosp, "Value", create_reverse_remap(levels))
    reclass_hosp.save(output_hospitals_reclass)

    print("Slope, bus, and hospital distance reclassified.")

# -------------------------------------------------------------------
# 8. Reclassify land use pattern
# -------------------------------------------------------------------
def reclassify_landuse(landuse_raster, output_reclass):
    remap_table = [
        [1, 3], [2, 3], [3, 4], [11, 6], [21, 2], [22, 2], [23, 1],
        [31, 7], [32, 8], [41, 1], [42, 1], [43, 1], [44, 1],
        [51, 2], [52, 4], [53, 10], [54, 5], [61, 4], [62, 3],
        [71, 6], [72, 5], [73, 5], [74, 1], [81, 1], [83, 1],
        [91, 0], [92, 0]
    ]
    remap_obj = RemapValue(remap_table)
    reclassified = Reclassify(landuse_raster, "Value", remap_obj)
    reclassified_nodata = SetNull(reclassified == 0, reclassified)
    reclassified_nodata.save(output_reclass)
    print("Land use reclassified – Vacant Land (53) prioritised, restricted areas set to NoData.")

# -------------------------------------------------------------------
# 9. Weighted Overlay – combine all criteria into a single suitability map
# -------------------------------------------------------------------
def weighted_overlay(slope_reclass, bus_reclass, hosp_reclass, landuse_reclass,
                     weights, output_suitability):
    wotable = WOTable([
        [slope_reclass, weights['slope'], "Value", create_direct_remap(10)],
        [bus_reclass, weights['bus'], "Value", create_direct_remap(10)],
        [hosp_reclass, weights['hospitals'], "Value", create_direct_remap(10)],
        [landuse_reclass, weights['landuse'], "Value", create_direct_remap(10)]
    ], [1, 10, 1])
    suitability = WeightedOverlay(wotable)
    suitability.save(output_suitability)
    print("Weighted overlay completed.")

# -------------------------------------------------------------------
# 10. Optimal Site Selection – extract highest‑suitability areas,
# #     then constrain by road adjacency and minimum area.
# -------------------------------------------------------------------
def select_optimal_sites(suitability_raster, roads_fc, final_site_fc, min_area_sqm=40469):
    max_val = int(arcpy.GetRasterProperties_management(suitability_raster, "MAXIMUM").getOutput(0))
    con_raster = Con(suitability_raster, suitability_raster, "", f"VALUE = {max_val}")
    con_raster.save("in_memory/con_output")

    filtered = MajorityFilter(con_raster, "EIGHT", "MAJORITY")
    filtered.save("in_memory/filtered")

    temp_poly = "in_memory/candidate_polygons"
    arcpy.conversion.RasterToPolygon(filtered, temp_poly, "SIMPLIFY", "Value")

    # Add area field and calculate
    arcpy.management.AddField(temp_poly, "Area_sqm", "DOUBLE")
    arcpy.management.CalculateGeometryAttributes(temp_poly, [["Area_sqm", "AREA"]], area_unit="SQUARE_METERS")

    temp_layer = "candidate_layer"
    arcpy.management.MakeFeatureLayer(temp_poly, temp_layer)
    arcpy.management.SelectLayerByLocation(temp_layer, "INTERSECT", roads_fc, "", "NEW_SELECTION")
    arcpy.management.SelectLayerByAttribute(temp_layer, "SUBSET_SELECTION", f"Area_sqm >= {min_area_sqm}")
    arcpy.management.CopyFeatures(temp_layer, final_site_fc)

    for item in ["con_output", "filtered", "candidate_polygons"]:
        arcpy.Delete_management(f"in_memory/{item}")
    print(f"Optimal sites saved to {final_site_fc}")

# -------------------------------------------------------------------
# Main workflow
# -------------------------------------------------------------------
def main():
    # ==================== UPDATE THESE PATHS ====================
    workspace_gdb = r"C:\Users\hosiu\Downloads\Lab4_23079474D_Ho Siu Fai（good)\Lab4_ArcGIS pro_visualization_copy\Task2c.gdb"
    dem_raster = r"C:\Users\hosiu\Downloads\DigitalTerrainModelDTM_GEOTIFF\Digital Terrain Model.tif"
    landuse_raster = r"C:\Users\hosiu\Downloads\LUMHK_RasterGrid_2021\LUMHK_RasterGrid_2021\LUM_end2021.tif"
    bus_stops = r"C:\Users\hosiu\Downloads\Bus_Stop_Locations_in_Hong_Kong_\Bus_Stop_Locations_in_Hong_Kong_.shp"
    roads_fc = r"C:\Users\hosiu\Downloads\Lab4_23079474D_Ho Siu Fai（good)\Centerline_KwunTongDistrict.shp"
    hospitals_csv = r"C:\Users\hosiu\Downloads\Hospital.csv"
    # =============================================================

    for path, name in [(dem_raster, "DEM"), (landuse_raster, "Land use"), (bus_stops, "Bus stops"), (roads_fc, "Roads")]:
        if not arcpy.Exists(path):
            print(f"ERROR: {name} not found at: {path}")
            return

    hospitals_fc = os.path.join(workspace_gdb, "hospitals_points")
    weights = {"slope": 20, "bus": 40, "hospitals": 10, "landuse": 30}

    slope_raw = "slope_raw"
    dist_bus = "dist_bus"
    dist_hospitals = "dist_hospitals"
    slope_reclass = "slope_reclass"
    bus_reclass = "bus_reclass"
    hosp_reclass = "hospitals_reclass"
    landuse_reclass = "landuse_reclass"
    suitability = "suitability"
    final_site = "Final_Hospital_Site"

    try:
        set_environment(workspace_gdb, dem_raster)
        csv_hospitals_to_points(hospitals_csv, hospitals_fc)
        derive_slope(dem_raster, slope_raw)
        distance_to_bus_stops(bus_stops, dist_bus)
        distance_to_hospitals(hospitals_fc, dist_hospitals)
        reclassify_criteria(slope_raw, dist_bus, dist_hospitals,
                            slope_reclass, bus_reclass, hosp_reclass, levels=10)
        reclassify_landuse(landuse_raster, landuse_reclass)
        weighted_overlay(slope_reclass, bus_reclass, hosp_reclass, landuse_reclass,
                         weights, suitability)
        select_optimal_sites(suitability, roads_fc, final_site, min_area_sqm=8100)
        print("\nHospital site selection completed successfully!")
        print(f"Output geodatabase: {workspace_gdb}")
        print(f"Final site feature class: {final_site}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()