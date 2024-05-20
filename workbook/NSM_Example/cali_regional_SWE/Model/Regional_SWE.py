import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer
from pystac_client import Client
import planetary_computer
from tqdm import tqdm
import math
import xarray
import rioxarray
import richdem as rd
import elevation
import pickle
import bz2
from datetime import date, datetime, timedelta

import National_Snow_Model_Regional


class Region_SWE_Simulation:
    """
    This class contains the necessary functions to run historical and up-to-date SWE simulations for a given shapefile region.
    Args:
        cwd (str): current working diectory. This should be the "National-Snow-Model" directory
        area (str): Name of the region to model. This should exactly match the name of the shapefile.
        year (str): 'YYYY' - Starting year of the Water Year to be modeled.
        shapefile_path (str): Path of the area shapefile.
        start_date (str): 'YYYY-MM-DD' - First date of model inference. If Regional_predict has not been executed before, this must be 'YYYY-10-01'.
        end_date (str): 'YYYY-MM-DD' - Final date of model infernce. 
        day_interval (int): Default = 7 days. Interval between model inferences. If changed, initial start_date must be 'YYYY-09-24' + interval
        plot (boolean): Default =True. Plot interactive map of modeled region SWE inline. Suggest setting False to improve speed and performance. 
    """ 
    def __init__(self, cwd, area, year):
        self = self
        self.cwd = cwd
        self.area = area
        self.year = year
            
        if not os.path.exists(self.cwd+'//'+area):
            os.makedirs(self.cwd+'//'+area)
            print('Created new directory:', self.cwd+'//'+area)
        if not os.path.exists(self.cwd+'//'+area+'//Predictions'):
            os.makedirs(self.cwd+'//'+area+'//Predictions')
            print('Created new directory:', self.cwd+'//'+area+'//Predictions')
        if not os.path.exists(self.cwd+'//'+area+'//Data//Processed'):
            os.makedirs(self.cwd+'//'+area+'//Data//Processed')
            print('Created new directory:', self.cwd+'//'+area+'//Data//Processed')
        if not os.path.exists(self.cwd+'//'+area+'//Data//NetCDF'):
            os.makedirs(self.cwd+'//'+area+'//Data//NetCDF')
            print('Created new directory:', self.cwd+'//'+area+'//Data//NetCDF')
        if not os.path.exists(self.cwd+'//'+area+'//Data//WBD'):
            os.makedirs(self.cwd+'//'+area+'//Data//WBD')

    # @staticmethod
    # def PreProcess(self, shapefile_path):
    def PreProcess(self, shapefile_path):
        """ Creates geospatial information for a triangular mesh shapefile.
            This may take a long time for larger areas. Once Geo_df.csv is made for an area, it does not need to be executed again."""

        gdf_shapefile = gpd.read_file(shapefile_path)

        gdf_shapefile = gdf_shapefile.drop(columns=['id'])

        # Get bounding box coordinates of shapefile
        minx, miny, maxx, maxy = gdf_shapefile.total_bounds

        ### Begin process to import geospatial features into DF

        # Define the source and target coordinate reference systems
        src_crs = CRS('EPSG:4326')  # WGS84

        # Check if the CRS is in UTM already, if not assing correct UTM zone - should specify shapefiles should be in EPSG 4326, but UTM is okay.
        if gdf_shapefile.crs.is_projected and 'utm' in gdf_shapefile.crs.name.lower():
            target_crs=gdf_shapefile.crs
        elif -126 < minx < -120:
            # transformer = Transformer.from_crs(src_crs, 'epsg:32610', always_xy=True)
            target_crs = CRS('EPSG:32610') #UTM zone 10
        elif -120 < minx < -114:
            # transformer = Transformer.from_crs(src_crs, 'epsg:32611', always_xy=True)
            target_crs = CRS('EPSG:32611') #UTM zone 11
        elif -114 < minx < -108:
            # transformer = Transformer.from_crs(src_crs, 'epsg:32612', always_xy=True)
            target_crs = CRS('EPSG:32612') #UTM zone 12
        elif -108 < minx < -102:
            # transformer = Transformer.from_crs(src_crs, 'epsg:32613', always_xy=True)
            target_crs = CRS('EPSG:32613') #UTM zone 13
        else:
            # transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
            target_crs = CRS('EPSG:3857') #Web Mercator

        #convert crs from UTM to WGS84
        transformer = Transformer.from_crs(target_crs, src_crs, always_xy=True)

        # Convert the bounding box coordinates to Web Mercator
        minx, miny = transformer.transform(minx, miny)
        maxx, maxy = transformer.transform(maxx, maxy)

        #Define AOI for DEM coverage
        area_of_interest = {"type": "Polygon","coordinates": [
        [
            #lower left
            [minx, miny],
            #upper left
            [minx, maxy],
            #upper right
            [maxx, maxy],
            #lower right
            [maxx, miny],
            #lower left
            [minx, miny],
            ]],}

        # Create columns to store vertex coordinates
        vertex_columns = ['vertex_1', 'vertex_2', 'vertex_3']

        # Iterate through each polygon row in the GeoDataFrame
        for index, row in gdf_shapefile.iterrows():
            # Extract coordinates of the vertices
            vertices = row['geometry'].exterior.coords[:3]  # Assuming triangles only
            # Add coordinates to the respective columns
            for i, vertex in enumerate(vertices):
                gdf_shapefile.at[index, f'vertex_{i+1}_long'] = vertex[0] 
                gdf_shapefile.at[index, f'vertex_{i+1}_lat'] = vertex[1]

        # Find centroid of each polygon and add it as new columns
        gdf_shapefile['centroid_long'] = gdf_shapefile['geometry'].centroid.x
        gdf_shapefile['centroid_lat'] = gdf_shapefile['geometry'].centroid.y

        # Convert the centroid and vertex coordinates to Web Mercator
        gdf_shapefile['centroid_long'], gdf_shapefile['centroid_lat'] = transformer.transform(gdf_shapefile['centroid_long'], gdf_shapefile['centroid_lat'])
        gdf_shapefile['vertex_1_long'], gdf_shapefile['vertex_1_lat'] = transformer.transform(gdf_shapefile['vertex_1_long'], gdf_shapefile['vertex_1_lat'])
        gdf_shapefile['vertex_2_long'], gdf_shapefile['vertex_2_lat'] = transformer.transform(gdf_shapefile['vertex_2_long'], gdf_shapefile['vertex_2_lat'])
        gdf_shapefile['vertex_3_long'], gdf_shapefile['vertex_3_lat'] = transformer.transform(gdf_shapefile['vertex_3_long'], gdf_shapefile['vertex_3_lat'])
        
        #Make a connection to get 90m Copernicus Digital Elevation Model (DEM) data with the Planetary Computer STAC API
        client = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            ignore_conformance=True,
        )

        search = client.search(
            collections=["cop-dem-glo-90"],
            intersects=area_of_interest
        )

        tiles = list(search.get_items())

        #Make a DF to connect locations with the larger data tile, and then extract elevations
        regions = []

        print("Retrieving Copernicus 90m DEM tiles")
        for i in tqdm(range(0, len(tiles))):
            row = [i, tiles[i].id]
            regions.append(row)
        regions = pd.DataFrame(columns = ['sliceID', 'tileID'], data = regions)
        regions = regions.set_index(regions['tileID'])
        del regions['tileID']

        #pull elevation, slope and aspect from DEM for point
        def GeoStat_func(i, Geospatial_df, regions, elev_L, slope_L, aspect_L, Long, Lat, tile):

            # convert coordinate to raster value
            lon = Geospatial_df.iloc[i][Long]
            lat = Geospatial_df.iloc[i][Lat]

            #connect point location to geotile
            tileid = 'Copernicus_DSM_COG_30_N' + str(math.floor(lat)) + '_00_W'+str(math.ceil(abs(lon))) +'_00_DEM'
            
            indexid = regions.loc[tileid]['sliceID']

            #Assing region
            signed_asset = planetary_computer.sign(tiles[indexid].assets["data"])
            #get elevation data in xarray object
            elevation = rioxarray.open_rasterio(signed_asset.href)

            #create copies to extract other geopysical information
            #Create Duplicate DF's
            slope = elevation.copy()
            aspect = elevation.copy()
                
            #transform projection
            transformer = Transformer.from_crs("EPSG:4326", elevation.rio.crs, always_xy=True)
            xx, yy = transformer.transform(lon, lat)
            
            #extract elevation values into numpy array
            tilearray = np.around(elevation.values[0]).astype(int)

            #set tile geo to get slope and set at rdarray
            geo = (math.floor(float(lon)), 90, 0.0, math.ceil(float(lat)), 0.0, -90)
            tilearray = rd.rdarray(tilearray, no_data = -9999)
            tilearray.projection = 'EPSG:4326'
            tilearray.geotransform = geo

            #get slope, note that slope needs to be fixed, way too high
            #get aspect value
            slope_arr = rd.TerrainAttribute(tilearray, attrib='slope_degrees')
            aspect_arr = rd.TerrainAttribute(tilearray, attrib='aspect')

            #save slope and aspect information 
            slope.values[0] = slope_arr
            aspect.values[0] = aspect_arr
            
            elev = round(elevation.sel(x=xx, y=yy, method="nearest").values[0])
            slop = round(slope.sel(x=xx, y=yy, method="nearest").values[0])
            asp = round(aspect.sel(x=xx, y=yy, method="nearest").values[0])
            
            #add point values to list
            elev_L.append(elev)
            slope_L.append(slop)
            aspect_L.append(asp)

        print("Interpolating Grid Cell Spatial Features")
        ###---------------------------------------------------------- Need to Parallelize This ----------------------------------------------------------
        print("Calcuating Vertex 1 Geo")
        v1_el = []
        v1_slope = []
        v1_aspect = []

        #run the elevation function, added tqdm to show progress
        [GeoStat_func(i, gdf_shapefile, regions, v1_el, v1_slope, v1_aspect,
                        'vertex_1_long', 'vertex_1_lat', tiles) for i in tqdm(range(0, len(gdf_shapefile)))]

        #Save each points elevation in DF
        gdf_shapefile['vertex_1_Elev_m'] = v1_el
        gdf_shapefile['vertex_1_slp_Deg'] = v1_slope
        gdf_shapefile['vertex_1_asp'] = v1_aspect

        print("Calcuating Vertex 2 Geo")
        v2_el = []
        v2_slope = []
        v2_aspect = []

        #run the elevation function, added tqdm to show progress
        [GeoStat_func(i, gdf_shapefile, regions, v2_el, v2_slope, v2_aspect,
                        'vertex_2_long', 'vertex_2_lat', tiles) for i in tqdm(range(0,len(gdf_shapefile)))]

        #Save each points elevation in DF
        gdf_shapefile['vertex_2_Elev_m'] = v2_el
        gdf_shapefile['vertex_2_slp_Deg'] = v2_slope
        gdf_shapefile['vertex_2_asp'] = v2_aspect

        print("Calcuating Vertex 3 Geo")
        v3_el = []
        v3_slope = []
        v3_aspect = []

        #run the elevation function, added tqdm to show progress
        [GeoStat_func(i, gdf_shapefile, regions, v3_el, v3_slope, v3_aspect,
                        'vertex_3_long', 'vertex_3_lat', tiles) for i in tqdm(range(0,len(gdf_shapefile)))]

        #Save each points elevation in DF
        gdf_shapefile['vertex_3_Elev_m'] = v3_el
        gdf_shapefile['vertex_3_slp_Deg'] = v3_slope
        gdf_shapefile['vertex_3_asp'] = v3_aspect

        print("Calcuating Centroid Geo")
        centroid_el = []
        centroid_slope = []
        centroid_aspect = []

        #run the elevation function, added tqdm to show progress
        [GeoStat_func(i, gdf_shapefile, regions, centroid_el, centroid_slope, centroid_aspect,
                        'centroid_long', 'centroid_lat', tiles) for i in tqdm(range(0,len(gdf_shapefile)))]

        #Save each points elevation in DF
        gdf_shapefile['centroid_Elev_m'] = centroid_el
        gdf_shapefile['centroid_slp_Deg'] = centroid_slope
        gdf_shapefile['centroid_asp'] = centroid_aspect

        #get mean Geospatial data
        def mean_Geo(df, geo):
            Centroid = 'centroid'+geo
            V1 = 'vertex_1'+geo
            V2 = 'vertex_2'+geo
            V3 = 'vertex_3'+geo
            
            df[geo] = (df[Centroid] + df[V1]+ df[V2] + df[V3]) /4

        Geo_df = gdf_shapefile.copy()
        #Get geaspatial means
        geospatialcols = ['_Elev_m', '_slp_Deg' , '_asp']

        #Training data
        [mean_Geo(Geo_df, i) for i in geospatialcols]

        #list of key geospatial component means
        geocol = ['_Elev_m','_slp_Deg','_asp']
        Geo_df = Geo_df[geocol].copy()
        Geo_df['_Long'] = gdf_shapefile['centroid_long']
        Geo_df['_Lat'] = gdf_shapefile['centroid_lat']
        Geo_df['area_m'] = gdf_shapefile['area']

        #adjust column names to be consistent with snotel
        Geo_df = Geo_df.rename( columns = {'_Long':'Long', '_Lat':'Lat', '_Elev_m': 'elevation_m',
                                    '_slp_Deg':'slope_deg' , '_asp': 'aspect'})
        Geo_df = Geo_df[['Long', 'Lat', 'area_m', 'elevation_m', 'slope_deg', 'aspect']]
            
        #This function defines northness: :  sine(Slope) * cosine(Aspect). this gives you a northness range of -1 to 1.
        #Note you'll need to first convert to radians. 
        #Some additional if else statements to get around sites with low obervations
        def northness(df):    
            
            if len(df) == 8: #This removes single value observations, need to go over and remove these locations from training too
                #Determine northness for site
                #convert to radians
                df = pd.DataFrame(df).T
                
                df['aspect_rad'] = df['aspect']*0.0174533
                df['slope_rad'] = df['slope_deg']*0.0174533
                
                df['northness'] = -9999
                for i in range(0, len(df)):
                    df['northness'].iloc[i] = math.sin(df['slope_rad'].iloc[i])*math.cos(df['aspect_rad'].iloc[i])

                #remove slope and aspects to clean df up
                df = df.drop(columns = ['aspect', 'slope_deg', 'aspect_rad', 'slope_rad', 'Region'])
                
                return df
                
            else:
                #convert to radians
                df['aspect_rad'] = df['aspect']*0.0174533
                df['slope_rad'] = df['slope_deg']*0.0174533
                
                df['northness'] = -9999
                for i in range(0, len(df)):
                    df['northness'].iloc[i] = math.sin(df['slope_rad'].iloc[i])*math.cos(df['aspect_rad'].iloc[i])

                
                #remove slope and aspects to clean df up
                df = df.drop(columns = ['aspect', 'slope_deg', 'aspect_rad', 'slope_rad'])
                
                return df
            
        Geo_df = northness(Geo_df)

        Geo_df.index.names = ['cell_id']

        Geo_df.to_csv(self.cwd+'//'+self.area+'//'+self.area+'_Geo_df.csv', index= True)


    def Prepare_Prediction(self):
        """ Create the necessary files to store SWE estimates. Estimates will begin on YYYY-09-24."""

        Geo_df = pd.read_csv(self.cwd+'//'+self.area+'//'+self.area+'_Geo_df.csv', index_col='cell_id')

        submission_format = Geo_df.drop(['Long','Lat','area_m','elevation_m','northness'], axis=1)
        submission_format.to_csv(self.cwd+'//'+self.area+'/Predictions/submission_format_'+self.area+'.csv')

        #also need to save initial 9/24 submission format
        submission_format.to_csv(self.cwd+'//'+self.area+'/Predictions/submission_format_'+self.area+'_'+str(self.year)+'-09-24.csv')

        print("Prediction CSV Created")

        Geo_df['Region']=""
        print('Defining Encompassing Model Regions')
        National_Snow_Model_Regional.SWE_Prediction(self.cwd, self.area, str(self.year)+'-10-01', self.year).Region_id(Geo_df)     

        #Split Sierras by elevation
        def region_split(df):
            if df['Region'] == 'S_Sierras' and (df['elevation_m'] > 2500): return 'S_Sierras_High'
            elif df['Region'] == 'S_Sierras' and (df['elevation_m'] <= 2500): return 'S_Sierras_Low'
            else: return df['Region']

        Geo_df['Region'] = Geo_df.apply(region_split, axis=1)

        Region_df_dict = Geo_df.groupby("Region")

        # Convert the GroupBy DataFrame to a dictionary of DataFrames
        grouped_dict = {}
        for group_label, group_df in Region_df_dict:
            group_df=group_df.drop("Region",axis=1)
            grouped_dict[group_label] = group_df
        

        Region_list = National_Snow_Model_Regional.SWE_Prediction(self.cwd, self.area, str(self.year)+'-10-01', self.year).Region_list

        init_dict = {}
        Region_Pred={}
        for region in Region_list:
            init_dict[region] = pd.read_hdf(self.cwd+'/Data/Processed/initialize_predictions.h5', key = region)
        
        for i in init_dict.keys():
            init_dict[i] = init_dict[i].drop(init_dict[i].index[1:])
            init_dict[i][str(self.year)+'-09-24'] = 0
            if i in grouped_dict.keys():
                init_dict[i] = pd.concat([init_dict[i], grouped_dict[i]], ignore_index=False, axis=0)
                init_dict[i]['WYWeek']=52
                init_dict[i] = init_dict[i].fillna(0)

            if len(init_dict[i]) > 1:
                init_dict[i] = init_dict[i].tail(-1)
            Region_Pred[i] = init_dict[i].iloc[:,:4].reset_index()

        #merge back for QA/QC step - does not impact predictions
        Region_Pred['S_Sierras']=Region_Pred['S_Sierras_High'].merge(Region_Pred['S_Sierras_Low'], how='outer')
        del Region_Pred['S_Sierras_High'], Region_Pred['S_Sierras_Low']

        for region in Region_list:
            init_dict[region].to_hdf(self.cwd+'//'+self.area+'//Predictions/predictions'+str(self.year)+'-09-24.h5', key = region)


        o_path=self.cwd+'//'+self.area+'//Data//Processed/Prediction_DF_'+str(self.year)+'-09-24.pkl'
        outfile = bz2.BZ2File(o_path, 'wb')
        pickle.dump(init_dict, outfile)
        outfile.close()

        outfile = self.cwd+'//'+self.area+'//Data//Processed/Region_Pred.pkl'
        with open(outfile, 'wb') as pickle_file:
            pickle.dump(Region_Pred, pickle_file)
        pickle_file.close()

        GM_template = pd.read_csv(self.cwd+'/Data/Pre_Processed_DA/ground_measures_features_template.csv')
        GM_template = GM_template.rename(columns = {'Unnamed: 0': 'station_id', 'Date': str(self.year)+'-09-24'})
        GM_template.index =GM_template['station_id']
        cols = [str(self.year)+'-09-24']
        GM_template =GM_template[cols]
        GM_template[cols] = GM_template[cols].fillna(0)
        GM_template.to_csv(self.cwd+'/Data/Pre_Processed_DA/ground_measures_features_'+str(self.year)+'-09-24.csv')

        DA_template = pd.read_csv(self.cwd+'/Data/Processed/DA_ground_measures_features_template.csv')
        DA_template['Date'] = str(self.year)+'-09-24'
        DA_template.to_csv(self.cwd+'//'+self.area+'/Data/Processed/DA_ground_measures_features_'+str(self.year)+'-09-24.csv')

        print('All initial files created for predictions beginning', 'October 1', str(self.year))



    def Regional_Predict(self, start_date, end_date, day_interval=7, plot =True):
        """ Create SWE estimates for the desired area. If first time executing, start_date must be YYYY-10-01."""
        # day interval can be altered to create list every n number of days by changing 7 to desired skip length.

        def daterange(start_date, end_date):
            for n in range(0, int((end_date - start_date).days) + 1, day_interval):
                yield start_date + timedelta(n)
                
        #create empty list to store dates
        datelist = []
        start_date = datetime.strptime(start_date, ("%Y-%m-%d"))
        end_date = datetime.strptime(end_date, ("%Y-%m-%d"))
        start_dt = date(start_date.year, start_date.month, start_date.day)
        end_dt = date(end_date.year, end_date.month, end_date.day)
        #append dates to list
        for dt in daterange(start_dt, end_dt):
            dt=dt.strftime('%Y-%m-%d')
            datelist.append(dt)
        
        #run the model through all time (data acqusition already completed)
        for date_ in datelist:
            print('Updating SWE predictions for ', date_)
            #connect interactive script to Wasatch Snow module, add model setup input for temporal resolution here.(eg. self.resolution = 7 days)
            Snow = National_Snow_Model_Regional.SWE_Prediction(self.cwd, self.area, date_, self.year, day_interval=day_interval)
            
            #Go get SNOTEL observations -- currently saving to csv, change to H5,
            #dd if self.data < 11-1-2022 and SWE = -9999, 
            Snow.Get_Monitoring_Data_Threaded()

            #Get the prediction extent
            bbox = Snow.getPredictionExtent()

            #Initialize/Download the granules
            Snow.initializeGranules(bbox, Snow.SCA_folder)
            
            #Process observations into Model prediction ready format,
            Snow.Data_Processing()

            #Agument with SCA
            Snow.augmentPredictionDFs()

            #Sometimes need to run twice for some reason, has a hard time loading the model (distributed computation engines for each region (multithreaded, cpus, GPUs))
            Snow.SWE_Predict()

            #Make CONUS netCDF file, compressed.
            Snow.netCDF_compressed(plot=False)
            if plot == True:
                #Make GeoDataframe and plot, self.Geo_df() makes the geo df.
                Snow.Geo_df()
                Snow.plot_interactive_SWE_comp(pinlat = 39.3, pinlong = -107, web = False)