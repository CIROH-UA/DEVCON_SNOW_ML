#Wasatch snow model
#Author: Ryan C. Johnson
#Date: 2022-3-09. Updated: 07/25/2023 
#This script assimilates SNOTEL observations, processes the data into a model friendly
#format, then uses a calibrated multi-layered perceptron network to make 1 km x 1 km
#CONUS scale SWE estimates. 


#required modules
import copy
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pickle 
from pickle import dump
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
#import contextily as cx
import rasterio
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
import netCDF4 as nc
#from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
import folium
from folium import plugins
import branca.colormap as cm
import rioxarray as rxr
import earthpy as et
import earthpy.spatial as es
import datetime as dt
from netCDF4 import date2num,num2date
from osgeo import osr
#import warningspip
from pyproj import CRS
import requests
import geojson
import pandas as pd
from multiprocessing import Pool, cpu_count
from shapely.ops import unary_union
import json
import geopandas as gpd, fiona, fiona.crs
import webbrowser
import warnings
from progressbar import ProgressBar
import shapely.geometry
import threading 
import bz2

#import contextily as ctx
import ulmo
from datetime import timedelta


#SCA packages

# Dataframe Packages
import numpy as np
import xarray as xr
import pandas as pd

# Vector Packages
import geopandas as gpd
import shapely

# Raster Packages
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import rasterstats as rs

# Data Access Packages
import earthaccess as ea
from nsidc_fetch import download, format_date, format_boundingbox
import h5py
import pickle
from tensorflow.keras.models import load_model

# General Packages
import re
from datetime import datetime
import glob
from pprint import pprint
from typing import Union
from pathlib import Path
from tqdm import tqdm
import time



warnings.filterwarnings("ignore")


class SWE_Prediction():
    def __init__(self,cwd, area, date, prev_year, day_interval=7, delta=7, timeDelay=0, threshold=0.2): # also need to update threshold in AgumentGeoDF func
        self = self
        self.area = area
        self.date = date
        self.prevdate = pd.to_datetime(date)-timedelta(days=day_interval)
        self.prevdate = self.prevdate.strftime('%Y-%m-%d')
        self.prev_year = prev_year

        #set path directory
        self.cwd = cwd
        
        """
            Initializes the NSM_SCA class.

            Parameters:
                cwd (str): The current working directory.
                area (str): Name of the region to model. This should exactly match the name of the shapefile.
                date (str): The date of the prediction.
                delta (int): How many days back to go for Last SWE.
                timeDelay (int): Use the SCA rasters from [timeDelay] days ago. Simulates operations in the real world.
                threshold (float): The threshold for NDSI, if greater than this value, it is considered to be snow.
        """
        if type(cwd) != Path:
            cwd = Path(cwd)  # Convert to Path object if necessary

        if type(date) != datetime:
            date = datetime.strptime(date, "%Y-%m-%d")  # Convert to datetime object if necessary
        
        # self.area = area
        # self.timeDelay = timeDelay
        self.delayedDate = date - pd.Timedelta(days=timeDelay)

        self.SCA_folder = self.cwd + "//Data//VIIRS_SCA//2021-2022NASA"
        # print(self.cwd)
        # print(self.SCA_folder)
        # self.SCA_folder = r"C:/Users/Dane Liljestrand/Desktop/NSM/SCA_test/Data/VIIRS_SCA//2022-2023NASA"
        self.threshold = threshold * 100  # Convert percentage to values used in VIIRS NDSI

#         self.auth = ea.login(strategy="netrc")
        self.auth = ea.login()
        if self.auth is None:
            print("Error logging into Earth Data account. Things will probably break")



         #make other date tags
      #  self.datestr = self.date.strftime.strftime('%Y-%m-%d')
       # m = datestr[5:7]
        #d = datestr[-2:]
        #y = datestr[:4]

        #pm = self.prevdate[0:2]
        #p = self.prevdate[3:5]
        #py = self.prevdate[-4:]

        #self.datekey = m[1]+'/'+d+'/'+y
        #self.date = y+'-'+m+'-'+d
        #self.prevcol = py+'-'+pm+'-'+p
        
        #Define Model Regions
        self.Region_list = ['N_Sierras',
                       'S_Sierras_High',
                       'S_Sierras_Low'
                    #    'Greater_Yellowstone',
                    #    'N_Co_Rockies',
                    #    'SW_Mont',
                    #    'SW_Co_Rockies',
                    #    'GBasin',
                    #    'N_Wasatch',
                    #    'N_Cascade',
                    #    'S_Wasatch',
                    #    'SW_Mtns',
                    #    'E_WA_N_Id_W_Mont',
                    #    'S_Wyoming',
                    #    'SE_Co_Rockies',
                    #    'Sawtooth',
                    #    'Ca_Coast',
                    #    'E_Or',
                    #    'N_Yellowstone',
                    #    'S_Cascade',
                    #    'Wa_Coast',
                    #    'Greater_Glacier',
                    #    'Or_Coast'
                      ]
        
        #Original Region List needed to remove bad features
        self.OG_Region_list = ['N_Sierras',
                   'S_Sierras'
                #    'Greater_Yellowstone',
                #    'N_Co_Rockies',
                #    'SW_Mont',
                #    'SW_Co_Rockies',
                #    'GBasin',
                #    'N_Wasatch',
                #    'N_Cascade',
                #    'S_Wasatch',
                #    'SW_Mtns',
                #    'E_WA_N_Id_W_Mont',
                #    'S_Wyoming',
                #    'SE_Co_Rockies',
                #    'Sawtooth',
                #    'Ca_Coast',
                #    'E_Or',
                #    'N_Yellowstone',
                #    'S_Cascade',
                #    'Wa_Coast',
                #    'Greater_Glacier',
                #    'Or_Coast'
                  ]
      
    #make Region identifier. The data already includes Region, but too many 'other' labels
    def Region_id(self, df):
        
        
        #put obervations into the regions
        # for i in tqdm(range(0, len(df))):
        for i in tqdm(df.index):

            #Sierras
            #Northern Sierras
            if -122.5 <= df['Long'][i] <=-119 and 39 <=df['Lat'][i] <= 42:
                loc = 'N_Sierras'
                df['Region'].loc[i] = loc

            #Southern Sierras
            if -122.5 <= df['Long'][i] <=-117 and df['Lat'][i] <= 39:
                loc = 'S_Sierras'
                df['Region'].loc[i] = loc

            #West Coast    
            #CACoastal (Ca-Or boarder)
            if df['Long'][i] <=-122.5 and df['Lat'][i] <= 42:
                loc = 'Ca_Coast'
                df['Region'].loc[i] = loc

            #Oregon Coastal (Or)?
            if df['Long'][i] <=-122.7 and 42<= df['Lat'][i] <= 46:
                loc = 'Or_Coast'
                df['Region'].loc[i] = loc

            #Olympis Coastal (Wa)
            if df['Long'][i] <=-122.5 and 46<= df['Lat'][i]:
                loc = 'Wa_Coast'
                df['Region'].loc[i] = loc    

            #Cascades    
             #Northern Cascades
            if -122.5 <= df['Long'][i] <=-119.4 and 46 <=df['Lat'][i]:
                loc = 'N_Cascade'
                df['Region'].loc[i] = loc

            #Southern Cascades
            if -122.7 <= df['Long'][i] <=-121 and 42 <=df['Lat'][i] <= 46:
                loc = 'S_Cascade'
                df['Region'].loc[i] = loc

            #Eastern Cascades and Northern Idaho and Western Montana
            if -119.4 <= df['Long'][i] <=-116.4 and 46 <=df['Lat'][i]:
                loc = 'E_WA_N_Id_W_Mont'
                df['Region'].loc[i] = loc
            #Eastern Cascades and Northern Idaho and Western Montana
            if -116.4 <= df['Long'][i] <=-114.1 and 46.6 <=df['Lat'][i]:
                loc = 'E_WA_N_Id_W_Mont'
                df['Region'].loc[i] = loc

            #Eastern Oregon
            if -121 <= df['Long'][i] <=-116.4 and 43.5 <=df['Lat'][i] <= 46:
                loc = 'E_Or'
                df['Region'].loc[i] = loc

            #Great Basin
            if -121 <= df['Long'][i] <=-112 and 42 <=df['Lat'][i] <= 43.5:
                loc = 'GBasin'
                df['Region'].loc[i] = loc

            if -119 <= df['Long'][i] <=-112 and 39 <=df['Lat'][i] <= 42:
                loc = 'GBasin'
                df['Region'].loc[i] = loc
                #note this section includes mojave too
            if -117 <= df['Long'][i] <=-113.2 and df['Lat'][i] <= 39:
                loc = 'GBasin'
                df['Region'].loc[i] = loc


            #SW Mtns (Az and Nm)
            if -113.2 <= df['Long'][i] <=-107 and df['Lat'][i] <= 37:
                loc = 'SW_Mtns'
                df['Region'].loc[i] = loc


            #Southern Wasatch + Utah Desert Peaks
            if -113.2 <= df['Long'][i] <=-109 and 37 <= df['Lat'][i] <= 39:
                loc = 'S_Wasatch'
                df['Region'].loc[i] = loc
            #Southern Wasatch + Utah Desert Peaks
            if -112 <= df['Long'][i] <=-109 and 39 <= df['Lat'][i] <= 40:
                loc = 'S_Wasatch'
                df['Region'].loc[i] = loc

            #Northern Wasatch + Bear River Drainage
            if -112 <= df['Long'][i] <=-109 and 40 <= df['Lat'][i] <= 42.5:
                loc = 'N_Wasatch'
                df['Region'].loc[i] = loc

            #YellowStone, Winds, Big horns
            if -111 <= df['Long'][i] <=-106.5 and 42.5 <= df['Lat'][i] <= 45.8:
                loc = 'Greater_Yellowstone'
                df['Region'].loc[i] = loc

            #North of YellowStone to Boarder
            if -112.5 <= df['Long'][i] <=-106.5 and 45.8 <= df['Lat'][i]:
                loc = 'N_Yellowstone'
                df['Region'].loc[i] = loc

             #SW Montana and nearby Idaho
            if -112 <= df['Long'][i] <=-111 and 42.5 <= df['Lat'][i] <=45.8:
                loc = 'SW_Mont'
                df['Region'].loc[i] = loc 
             #SW Montana and nearby Idaho
            if -113 <= df['Long'][i] <=-112 and 43.5 <= df['Lat'][i] <=45.8:
                loc = 'SW_Mont'
                df['Region'].loc[i] = loc
            #SW Montana and nearby Idaho
            if -113 <= df['Long'][i] <=-112.5 and 45.8 <= df['Lat'][i] <=46.6:
                loc = 'SW_Mont'
                df['Region'].loc[i] = loc
             #Sawtooths, Idaho
            if -116.4 <= df['Long'][i] <=-113 and 43.5 <= df['Lat'][i] <=46.6:
                loc = 'Sawtooth'
                df['Region'].loc[i] = loc

            #Greater Glacier
            if -114.1 <= df['Long'][i] <=-112.5 and 46.6 <= df['Lat'][i]:
                loc = 'Greater_Glacier'
                df['Region'].loc[i] = loc 

             #Southern Wyoming 
            if -109 <= df['Long'][i] <=-104.5 and 40.99 <= df['Lat'][i] <= 42.5 :
                loc = 'S_Wyoming'
                df['Region'].loc[i] = loc 
            #Southern Wyoming
            if -106.5 <= df['Long'][i] <=-104.5 and 42.5 <= df['Lat'][i] <= 43.2:
                loc = 'S_Wyoming'
                df['Region'].loc[i] = loc 
                
             #Northern Colorado Rockies
            if -109 <= df['Long'][i] <=-104.5 and 38.3 <= df['Lat'][i] <= 40.99:
                loc = 'N_Co_Rockies'
                df['Region'].loc[i] = loc 

             #SW Colorado Rockies
            if -109 <= df['Long'][i] <=-106 and 36.99 <= df['Lat'][i] <= 38.3:
                loc = 'SW_Co_Rockies'
                df['Region'].loc[i] = loc 

            #SE Colorado Rockies + Northern New Mexico
            if -106 <= df['Long'][i] <=-104.5 and 34 <= df['Lat'][i] <= 38.3:
                loc = 'SE_Co_Rockies'
                df['Region'].loc[i] = loc  
            #SE Colorado Rockies + Northern New Mexico
            if -107 <= df['Long'][i] <=-106 and 34 <= df['Lat'][i] <= 36.99:
                loc = 'SE_Co_Rockies'
                df['Region'].loc[i] = loc 
                              
                
    def get_SNOTEL(self, sitecode, start_date, end_date):
      #  print(sitecode)

        #This is the latest CUAHSI API endpoint
        wsdlurl = 'https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL'

        #Daily SWE
        variablecode = 'SNOTEL:WTEQ_D'

        values_df = None
        try:
            #Request data from the server
            site_values = ulmo.cuahsi.wof.get_values(wsdlurl, sitecode, variablecode, start=start_date, end=end_date)

            end_date=end_date.strftime('%Y-%m-%d')
            #Convert to a Pandas DataFrame   
            SNOTEL_SWE = pd.DataFrame.from_dict(site_values['values'])
            #Parse the datetime values to Pandas Timestamp objects
            SNOTEL_SWE['datetime'] = pd.to_datetime(SNOTEL_SWE['datetime'], utc=True)
            #Set the DataFrame index to the Timestamps
            SNOTEL_SWE = SNOTEL_SWE.set_index('datetime')
            #Convert values to float and replace -9999 nodata values with NaN
            SNOTEL_SWE['value'] = pd.to_numeric(SNOTEL_SWE['value']).replace(-9999, np.nan)
            #Remove any records flagged with lower quality
            SNOTEL_SWE = SNOTEL_SWE[SNOTEL_SWE['quality_control_level_code'] == '1']

            SNOTEL_SWE['station_id'] = sitecode
            SNOTEL_SWE.index = SNOTEL_SWE.station_id
            SNOTEL_SWE = SNOTEL_SWE.rename(columns = {'value':end_date})
            col = [end_date]
            SNOTEL_SWE = SNOTEL_SWE[col].iloc[-1:]


        except:
            print('Unable to fetch SWE data for site ', sitecode, 'SWE value: -9999')
            end_date=end_date.strftime('%Y-%m-%d')
            SNOTEL_SWE = pd.DataFrame(-9999, columns = ['station_id', end_date], index =[1])
            SNOTEL_SWE['station_id'] = sitecode
            SNOTEL_SWE = SNOTEL_SWE.set_index('station_id')


        return SNOTEL_SWE


    def get_CDEC(self, station_id, sensor_id, resolution, start_date, end_date ):

        try:
            # old url = 'https://cdec.water.ca.gov/dynamicapp/selectQuery?Stations=%s' % (station_id) + '&SensorNums=%s' % (
                #sensor_id) + '&dur_code=%s' % (resolution) + '&Start=%s' % (start_date) + '&End=%s' % (end_date)
            url = 'https://cdec.water.ca.gov/dynamicapp/selectSnow?Stations=%s' % (station_id) + '&SensorNums=%s' % (
                sensor_id) + '&Start=%s' % (start_date) + '&End=%s' % (end_date)
            #CDEC_SWE = pd.read_html(url)[0]
            CDEC_SWE = pd.read_html(url)[1]
            CDEC_SWE['station_id'] = 'CDEC:' + station_id
            CDEC_SWE = CDEC_SWE.set_index('station_id')
            CDEC_SWE = pd.DataFrame(CDEC_SWE.iloc[-1]).T
            #col = ['SNOW WC INCHES']
            col = 'W.C.'
            CDEC_SWE = CDEC_SWE[col]
            CDEC_SWE = CDEC_SWE.rename(columns={'W.C.': end_date})

        except:
            print('Unable to fetch SWE data for site ', station_id, 'SWE value: -9999')
            CDEC_SWE = pd.DataFrame(-9999, columns = ['station_id', end_date], index =[1])
            CDEC_SWE['station_id'] = 'CDEC:'+station_id
            CDEC_SWE = CDEC_SWE.set_index('station_id')



        return CDEC_SWE
    
    
    def Get_Monitoring_Data(self):
        GM_template = pd.read_csv(self.cwd+'/Data/Pre_Processed_DA/ground_measures_features_template.csv')
        GM_template = GM_template.rename(columns = {'Unnamed: 0': 'station_id'})
        GM_template.index =GM_template['station_id']
        cols = ['Date']
        GM_template =GM_template[cols]


        #Get all records, can filter later,
        self.CDECsites = list(GM_template.index)
        self.CDECsites = list(filter(lambda x: 'CDEC' in x, self.CDECsites))
        self.CDECsites = [x[-3:] for x in self.CDECsites]
        
        date = pd.to_datetime(self.date)

        start_date = date-timedelta(days = 1)
        start_date = start_date.strftime('%Y-%m-%d')

        resolution = 'D'
        sensor_id='3'

        SWE_df = pd.DataFrame(columns = ['station_id', date.strftime('%Y-%m-%d')], index =[1])
        SWE_df = SWE_df.set_index('station_id')

        print('Getting California Data Exchange Center SWE data from sites: ')
        for site in self.CDECsites:
            print(site)
            CDEC = self.get_CDEC(site, sensor_id, resolution, start_date, date.strftime('%Y-%m-%d') )
            frames = [SWE_df, CDEC]
            SWE_df = pd.concat(frames)

    #    cols = [date]
     #   SWE_df = SWE_df[cols]



        self.Snotelsites = list(GM_template.index)
        self.Snotelsites = list(filter(lambda x: 'SNOTEL' in x, self.Snotelsites))


        print('Getting NRCS SNOTEL SWE data from sites: ')
        for site in self.Snotelsites:
            print(site)
            Snotel = self.get_SNOTEL(site, start_date, date)
            frames = [SWE_df, Snotel]
            SWE_df = pd.concat(frames)


        #SWE_df = SWE_df[cols]
        SWE_df = SWE_df.iloc[1:]
        date = date.strftime('%Y-%m-%d')

        #SWE_df= SWE_df[~SWE_df.index.duplicated(keep = 'first')]


        SWE_df[date] = SWE_df[date].replace(['--'], -9999)

        SWE_df[date] = SWE_df[date].astype(float)


        NegSWE = SWE_df[SWE_df[date].between(-10,-.1)].copy()
        NegSWE[date] =0


        SWE_df.update(NegSWE)

        #SWE_df = SWE_df.rename(columns = {'Unnamed: 0': 'station_id'})
        #SWE_df = SWE_df.set_index('station_id')

        SWE_df.to_csv(self.cwd+'/Data/Pre_Processed_DA/ground_measures_features_'+date+'.csv')
        
        
    def get_SNOTEL_Threaded(self, sitecode, start_date, end_date):
        #print(sitecode)

        #This is the latest CUAHSI API endpoint
        wsdlurl = 'https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL'

        #Daily SWE
        variablecode = 'SNOTEL:WTEQ_D'

        values_df = None
        try:
            #Request data from the server
            site_values = ulmo.cuahsi.wof.get_values(wsdlurl, sitecode, variablecode, start=start_date, end=end_date)

            end_date=end_date.strftime('%Y-%m-%d')
            #Convert to a Pandas DataFrame   
            SNOTEL_SWE = pd.DataFrame.from_dict(site_values['values'])
            #Parse the datetime values to Pandas Timestamp objects
            SNOTEL_SWE['datetime'] = pd.to_datetime(SNOTEL_SWE['datetime'], utc=True)
            #Set the DataFrame index to the Timestamps
            SNOTEL_SWE = SNOTEL_SWE.set_index('datetime')
            #Convert values to float and replace -9999 nodata values with NaN
            SNOTEL_SWE['value'] = pd.to_numeric(SNOTEL_SWE['value']).replace(-9999, np.nan)
            #Remove any records flagged with lower quality
            SNOTEL_SWE = SNOTEL_SWE[SNOTEL_SWE['quality_control_level_code'] == '1']

            #SNOTEL_SWE['station_id'] = sitecode
            #SNOTEL_SWE.index = SNOTEL_SWE.station_id
            #SNOTEL_SWE = SNOTEL_SWE.rename(columns = {'value':end_date})
            #col = [end_date]
            #SNOTEL_SWE = SNOTEL_SWE[col].iloc[-1:]
            self.SWE_df[self.date].loc[sitecode] = SNOTEL_SWE['value'].values[0]

        except:
            print('Unable to fetch SWE data for site ', sitecode, 'SWE value: -9999')
            #end_date=end_date.strftime('%Y-%m-%d')
            #SNOTEL_SWE = pd.DataFrame(-9999, columns = ['station_id', end_date], index =[1])
            #SNOTEL_SWE['station_id'] = sitecode
            #SNOTEL_SWE = SNOTEL_SWE.set_index('station_id')
            self.SWE_df[self.date].loc[sitecode] = -9999

        #frames = [self.SWE_df, SNOTEL_SWE]
        #self.SWE_df = pd.concat(frames)
        #return SNOTEL_SWE


    def get_CDEC_Threaded(self, station_id, sensor_id, resolution, start_date, end_date ):
        try:
            # old url = 'https://cdec.water.ca.gov/dynamicapp/selectQuery?Stations=%s' % (station_id) + '&SensorNums=%s' % (
                #sensor_id) + '&dur_code=%s' % (resolution) + '&Start=%s' % (start_date) + '&End=%s' % (end_date)
            url = 'https://cdec.water.ca.gov/dynamicapp/selectSnow?Stations=%s' % (station_id) + '&SensorNums=%s' % (
                sensor_id) + '&Start=%s' % (start_date) + '&End=%s' % (end_date)
            #CDEC_SWE = pd.read_html(url)[0]
            CDEC_SWE = pd.read_html(url)[1]
            CDEC_SWE['station_id'] = 'CDEC:' + station_id
            CDEC_SWE = CDEC_SWE.set_index('station_id')
            CDEC_SWE = pd.DataFrame(CDEC_SWE.iloc[-1]).T
            #col = ['SNOW WC INCHES']
            col = 'W.C.'
            CDEC_SWE = CDEC_SWE[col]
            CDEC_SWE = CDEC_SWE.rename(columns={'W.C.': end_date})

        except:
            print('Unable to fetch SWE data for site ', station_id, 'SWE value: -9999')
            CDEC_SWE = pd.DataFrame(-9999, columns = ['station_id', end_date], index =[1])
            CDEC_station_id = 'CDEC:'+station_id
            CDEC_SWE['station_id'] = CDEC_station_id
            CDEC_SWE = CDEC_SWE.set_index('station_id')
            self.SWE_df[self.date].loc[CDEC_station_id] = CDEC_SWE[end_date]

            
        #frames = [self.SWE_df, CDEC_SWE]    
        #self.SWE_df = pd.concat(frames)
        #return CDEC_SWE
        

    def Get_Monitoring_Data_Threaded(self):
        GM_template = pd.read_csv(self.cwd+'/Data/Pre_Processed_DA/ground_measures_features_template.csv')
        GM_template = GM_template.rename(columns = {'Unnamed: 0': 'station_id'})
        GM_template.index =GM_template['station_id']
        cols = ['Date']
        GM_template =GM_template[cols]


        #Get all records, can filter later,
        self.CDECsites = list(GM_template.index)
        self.CDECsites = list(filter(lambda x: 'CDEC' in x, self.CDECsites))
        self.CDECsites_complete = self.CDECsites.copy()
        self.CDECsites = [x[-3:] for x in self.CDECsites]
        
        self.Snotelsites = list(GM_template.index)
        self.Snotelsites = list(filter(lambda x: 'SNOTEL' in x, self.Snotelsites))

        
        date = pd.to_datetime(self.date)

        start_date = date-timedelta(days = 1)
        start_date = start_date.strftime('%Y-%m-%d')

        resolution = 'D'
        sensor_id='3'
        
        #Make SWE observation dataframe        
        self.station_ids = self.CDECsites_complete + self.Snotelsites
        self.SWE_NA_fill = [-9999]*len(self.station_ids)
        self.SWE_df = pd.DataFrame(list(zip(self.station_ids, self.SWE_NA_fill)),
                                   columns = ['station_id', date.strftime('%Y-%m-%d')])
        self.SWE_df = self.SWE_df.set_index('station_id')

        print('Getting California Data Exchange Center SWE data from sites: ')
        threads = []  # create list to store thread references
        
         # create new threads and append them to the list of threads
        for site in self.CDECsites:
            print(site)
            # functions with arguments must have an 'empty' arg at the end of the passed 'args' tuple
            t = threading.Thread(target=self.get_CDEC_Threaded, args=(site, sensor_id, resolution, start_date, date.strftime('%Y-%m-%d')))
            threads.append(t)

        # start all threads
        for t in threads:
            t.start()
        # !!!!! IMPORTANT !!!!!
        # join all threads to queue so the system will wait until every thread has completed
        for t in threads:
            t.join()
    

        print('Getting NRCS SNOTEL SWE data from sites: ')
        threads = []  # create list to store thread references
        
         # create new threads and append them to the list of threads
        for site in self.Snotelsites:
            print(site)
            # functions with arguments must have an 'empty' arg at the end of the passed 'args' tuple
            t = threading.Thread(target=self.get_SNOTEL_Threaded, args=(site,  start_date, date))
            threads.append(t)

        # start all threads
        for t in threads:
            t.start()
        # !!!!! IMPORTANT !!!!!
        # join all threads to queue so the system will wait until every thread has completed
        for t in threads:
            t.join()
        
        date = date.strftime('%Y-%m-%d')

        self.SWE_df= self.SWE_df[~self.SWE_df.index.duplicated(keep = 'first')]

        #remove -- from CDEC predictions and make df a float
        self.SWE_df[date] = self.SWE_df[date].astype(str)
        self.SWE_df[date] = self.SWE_df[date].replace(['--'], -9999)
        self.SWE_df[date] = pd.to_numeric(self.SWE_df[date], errors = 'coerce')
        self.SWE_df[date] = self.SWE_df[date].fillna(-9999)

        NegSWE = self.SWE_df[self.SWE_df[date].between(-10,-.1)].copy()
        NegSWE[date] =0


        self.SWE_df.update(NegSWE)
        self.SWE_df.reset_index(inplace = True)
        self.SWE_df=self.SWE_df.rename(columns = {'index': 'station_id'})
        self.SWE_df = self.SWE_df.set_index('station_id')

        self.SWE_df.to_csv(self.cwd+'/Data/Pre_Processed_DA/ground_measures_features_'+date+'.csv')
  
    #Data Assimilation script, takes date and processes to run model.            
    def Data_Processing(self):
          

        #load ground truth values (SNOTEL): Testing
        obs_path = self.cwd+'/Data/Pre_Processed_DA/ground_measures_features_' + self.date + '.csv'
        self.GM_Test = pd.read_csv(obs_path)

        #load ground truth values (SNOTEL): previous week, these have Na values filled by prev weeks obs +/- mean region Delta SWE
        obs_path = self.cwd+'\\'+self.area+'/Data/Processed/DA_ground_measures_features_' + self.prevdate + '.csv'
        self.GM_Prev = pd.read_csv(obs_path)
        colrem = ['Region','Prev_SWE','Delta_SWE']
        self.GM_Prev = self.GM_Prev.drop(columns =colrem)

        #All coordinates of 1 km polygon used to develop ave elevation, ave slope, ave aspect
        path = self.cwd+'\\'+self.area+'/Data/Processed/Region_Pred.pkl'
        #load regionalized geospatial data
        self.RegionTest = open(path, "rb")
        self.RegionTest = pickle.load(self.RegionTest)

        ### Load H5 previous prediction files into dictionary
        self.prev_SWE = {}
        for region in self.Region_list:
            #The below file will serve as a starting point, if interested in making predictions for a specific region, set other
            #locations to only 1 location (vs. all region locations)
            self.prev_SWE[region] = pd.read_hdf(self.cwd+'\\'+self.area+'/Predictions/predictions'+ self.prevdate+'.h5', key = region)
            self.prev_SWE[region] = pd.DataFrame(self.prev_SWE[region][self.prevdate])
            self.prev_SWE[region] = self.prev_SWE[region].rename(columns = {self.prevdate: 'prev_SWE'})

        #change first column to station id
        self.GM_Test = self.GM_Test.rename(columns = {'Unnamed: 0':'station_id'})
        self.GM_Prev = self.GM_Prev.rename(columns = {'Unnamed: 0':'station_id'})


        #Fill NA observations
        #self.GM_Test[self.date] = self.GM_Test[self.date].fillna(-9999)

        #drop na and put into modeling df format
        self.GM_Test = self.GM_Test.melt(id_vars=["station_id"]).dropna()

        #change variable to Date and value to SWE
        self.GM_Test = self.GM_Test.rename(columns ={'variable': 'Date', 'value':'SWE'})

        #load ground truth meta
        self.GM_Meta = pd.read_csv(self.cwd+'/Data/Pre_Processed/ground_measures_metadata.csv')

        #merge testing ground truth location metadata with snotel data
        self.GM_Test = self.GM_Meta.merge(self.GM_Test, how='inner', on='station_id')
        self.GM_Test = self.GM_Test.set_index('station_id')
        self.GM_Prev = self.GM_Prev.set_index('station_id')

        self.GM_Test.rename(columns={'name': 'location', 'latitude': 'Lat', 'longitude': 'Long', 'value': 'SWE'}, inplace=True)
        
        #drop NA columns from initial observations
        prev_index = self.GM_Prev.index
        self.GM_Test = self.GM_Test.loc[prev_index]
        
        #Make a dictionary for current snotel observations
        self.Snotel = self.GM_Test.copy()
        self.Snotel['Region'] = 'other'
        self.Region_id(self.Snotel)
        self.RegionSnotel  = {name: self.Snotel.loc[self.Snotel['Region'] == name] for name in self.Snotel.Region.unique()}

        #Make a dictionary for previous week's snotel observations
        self.prev_Snotel = self.GM_Prev.copy()
        self.prev_Snotel['Region'] = 'other'
        self.Region_id(self.prev_Snotel)
        self.prev_RegionSnotel  = {name: self.prev_Snotel.loc[self.prev_Snotel['Region'] == name] for name in self.prev_Snotel.Region.unique()}
        

        #add week number to observations
        for i in self.RegionTest.keys():
            self.RegionTest[i] = self.RegionTest[i].reset_index(drop=True)
            self.RegionTest[i]['Date'] = pd.to_datetime(self.RegionSnotel[i]['Date'][0])
            self.week_num(i)

        #set up dataframe to save to be future GM_Pred
        col = list(self.GM_Test.columns)+['Region']
        self.Future_GM_Pred = pd.DataFrame(columns = col)
        
        print('Regional data QA/QC')
        dfs_to_concat = []
        for region in self.OG_Region_list:
            self.NaReplacement(region)
            self.RegionSnotel[region]['Prev_SWE'] =self.prev_RegionSnotel[region]['SWE']
            self.RegionSnotel[region]['Delta_SWE'] = self.RegionSnotel[region]['SWE'] - self.RegionSnotel[region]['Prev_SWE']
            dfs_to_concat.append(self.RegionSnotel[region])

            #make dataframe to function as next forecasts GM_Prev
            self.Future_GM_Pred = pd.concat(dfs_to_concat)

        #Need to save 'updated non-na' df's
        GM_path = self.cwd+'//'+self.area+'/Data/Processed/DA_ground_measures_features_'+ self.date + '.csv'

        self.Future_GM_Pred.to_csv(GM_path)


        #This needs to be here to run in next codeblock
        self.Regions = list(self.RegionTest.keys()).copy()


        #Make dictionary in Regions dict for each region's dictionary of Snotel sites
    #Regions = list(RegionTrain.keys()).copy()

        for i in tqdm(self.Regions):

            snotel = i+'_Snotel'
            self.RegionTest[snotel] = {site: self.RegionSnotel[i].loc[site] for site in self.RegionSnotel[i].index.unique()}

           #get training and testing sites that are the same
            test = self.RegionTest[snotel].keys()

            for j in test:
                self.RegionTest[snotel][j] = self.RegionTest[snotel][j].to_frame().T
            #remove items we do not need
                self.RegionTest[snotel][j] = self.RegionTest[snotel][j].drop(columns = ['Long', 'Lat', 'location',
                                                                             'elevation_m', 'state', 'Region'])
            #make date index
                self.RegionTest[snotel][j] = self.RegionTest[snotel][j].set_index('Date')

            #rename columns to represent site info
                colnames = self.RegionTest[snotel][j].columns
                sitecolnames = [x +'_'+ j for x in colnames]
                names = dict(zip(colnames, sitecolnames))
                self.RegionTest[snotel][j] = self.RegionTest[snotel][j].rename(columns = names)

            #make a df for training each region, 

        for R in tqdm(self.Regions):
            snotels = R+'_Snotel'  
           # RegionTest[R] = RegionTest[R].reset_index()
           # print(R)
            sites = list(self.RegionTest[R]['cell_id'])
            sitelen = len(sites)-1
            self.RegionTest[R] = self.RegionTest[R].set_index('cell_id')

            for S in self.RegionTest[snotels].keys():
                self.RegionTest[snotels][S] = self.RegionTest[snotels][S]._append([self.RegionTest[snotels][S]]*sitelen, ignore_index = True)
                self.RegionTest[snotels][S].index = sites
                self.RegionTest[R]= pd.concat([self.RegionTest[R], self.RegionTest[snotels][S].reindex(self.RegionTest[R].index)], axis=1)
            self.RegionTest[R] = self.RegionTest[R].fillna(-9999.99)
            del self.RegionTest[R]['Date']


        #Perform the splitting for S_Sierras High and Low elevations
        self.RegionTest['S_Sierras_High'] =self.RegionTest['S_Sierras'].loc[self.RegionTest['S_Sierras']['elevation_m'] > 2500].copy()
        self.RegionTest['S_Sierras_Low'] = self.RegionTest['S_Sierras'].loc[self.RegionTest['S_Sierras']['elevation_m'] <= 2500].copy()
        del self.RegionTest['S_Sierras']


        #Add previous Cell SWE
        for region in self.Region_list:
            self.RegionTest[region] = pd.concat([self.RegionTest[region], self.prev_SWE[region]], axis =1, join = 'inner')


            #save dictionaries as pkl
        # create a compressed binary pickle file 
        RVal = bz2.BZ2File(self.cwd+'//'+self.area+'/Data/Processed/Prediction_DF_'+self.date + '.pkl', 'wb')
        
        # write the python object (dict) to pickle file
        pickle.dump(self.RegionTest,RVal)

        # close file
        RVal.close()

    #Get the week number of the observations, from beginning of water year    
    def week_num(self, region):
        #week of water year
        weeklist = []

        for i in tqdm(range(0,len(self.RegionTest[region]))):
            if self.RegionTest[region]['Date'][i].month<11:
                y = self.RegionTest[region]['Date'][i].year-1
            else:
                y = self.RegionTest[region]['Date'][i].year

            WY_start = pd.to_datetime(str(y)+'-10-01')
            deltaday = self.RegionTest[region]['Date'][i]-WY_start
            deltaweek = round(deltaday.days/7)
            weeklist.append(deltaweek)


        self.RegionTest[region]['WYWeek'] = weeklist
            
   #NA Replacement script for necessary SNOTEL sites without observations     
    def NaReplacement(self, region):
        #Make NA values mean snowpack values, put in >= for no snow times
        meanSWE = np.mean(self.RegionSnotel[region]['SWE'][self.RegionSnotel[region]['SWE']>=0])
        #print(region, meanSWE)
        #add if statement to meanSWE
        if meanSWE < 0.15:
            meanSWE = 0
        self.RegionSnotel[region]['SWE'][self.RegionSnotel[region]['SWE']<0]= meanSWE


        prev_meanSWE = np.mean(self.prev_RegionSnotel[region]['SWE'][self.prev_RegionSnotel[region]['SWE']>=0])
        #print(region, prev_meanSWE)
        #add if statement to meanSWE
        if prev_meanSWE < 0.15:
            prev_meanSWE = 0
        self.prev_RegionSnotel[region]['SWE'][self.prev_RegionSnotel[region]['SWE']<0]= prev_meanSWE

#        delta = self.RegionSnotel[region]['SWE']-self.prev_RegionSnotel[region]['SWE']
#        delta = pd.DataFrame(delta)
#        delta = delta.rename(columns = {'SWE':'Delta'})

        #get values that are not affected by NA
#        delta = delta[delta['Delta']> -9000]

        #Get mean Delta to adjust observed SWE
#        meanD = np.mean(delta['Delta'])

        #go and fix current SWE observations
        #Get bad obsevations and snotel sites
#        badSWE_df = self.RegionSnotel[region][self.RegionSnotel[region]['SWE']< 0].copy()
#        bad_sites = list(badSWE_df.index)


        #remove bad observations from SWE obsevations
#        self.RegionSnotel[region] = self.RegionSnotel[region][self.RegionSnotel[region]['SWE'] >= 0]

        #Fix bad observatoins by taking previous obs +/- mean delta SWE
       # print('Fixing these bad sites in ', region, ':')
#        for badsite in bad_sites:
 #           print(badsite)
  #          badSWE_df.loc[badsite,'SWE']=self.prev_RegionSnotel[region].loc[badsite]['SWE'] + meanD

        #Add observations back to DF
#        self.RegionSnotel[region] = pd.concat([self.RegionSnotel[region], badSWE_df])


#     #Take in and make prediction
#     def SWE_Predict(self):
        
#         #self.plot = plot
#         #load first SWE observation forecasting dataset with prev and delta swe for observations. 
        
#         #load regionalized forecast data
#         ifile = bz2.BZ2File(self.cwd+'//'+self.area+'/Data/Processed/Prediction_DF_'+self.date + '.pkl','rb')

#         self.Forecast = pickle.load(ifile)
       
#         ifile.close()

#         #load RFE optimized features
#         self.Region_optfeatures= pickle.load(open(self.cwd+"/Model/Prev_SWE_Models_Final/opt_features_prevSWE.pkl", "rb"))
        

#         #Reorder regions
#         self.Forecast = {k: self.Forecast[k] for k in self.Region_list}
        

#         #Make and save predictions for each reagion
#         self.Prev_df = pd.DataFrame()
#         self.predictions ={}
#         print ('Making predictions for: ', self.date)

#         for Region in self.Region_list:
#             print(Region)
#             self.predictions[Region] = self.Predict(Region)
#             self.predictions[Region] = pd.DataFrame(self.predictions[Region])
            
#           #  if self.plot == True:
#            #     del self.predictions[Region]['geometry']
#             self.Prev_df = self.Prev_df.append(pd.DataFrame(self.predictions[Region][self.date]))
#             self.Prev_df = pd.DataFrame(self.Prev_df)

#             self.predictions[Region].to_hdf(self.cwd+'//'+self.area+'/Predictions/predictions'+self.date+'.h5', key = Region)


#         #load submission DF and add predictions, if locations are removed or added, this needs to be modified
#         self.subdf = pd.read_csv(self.cwd+'//'+self.area+'/Predictions/submission_format_'+self.area+'_'+self.prevdate+'.csv')
#         self.subdf.index = list(self.subdf.iloc[:,0].values)
#         self.subdf = self.subdf.iloc[:,1:]

#         self.sub_index = self.subdf.index
#         #reindex predictions
#         self.Prev_df = self.Prev_df.loc[self.sub_index]
#         self.subdf[self.date] = self.Prev_df[self.date].astype(float)
#         #subdf.index.names = [' ']
#         self.subdf.to_csv(self.cwd+'//'+self.area+'/Predictions/submission_format_'+self.area+'_'+self.date+'.csv')
  
#     #set up model prediction function
#     def Predict(self, Region):

#         ##region specific features
#         features = self.Region_optfeatures[Region]

#         #Make prediction dataframe
#         forecast_data = self.Forecast[Region].copy()
#         forecast_data = forecast_data[features]


#         #change all na values to prevent scaling issues
#         forecast_data[forecast_data< -9000]= -10


#         #load and scale data

#         #set up model checkpoint to be able to extract best models
#         checkpoint_filepath = self.cwd+'/Model/Prev_SWE_Models_Final/' +Region+ '/'
#         model = checkpoint_filepath+Region+'_model.h5'
#         print(model)
#         model=load_model(model)


#         #load SWE scaler
#         SWEmax = np.load(checkpoint_filepath+Region+'_SWEmax.npy')
#         SWEmax = SWEmax.item()

#         #load features scaler
#         #save scaler data here too
#         scaler =  pickle.load(open(checkpoint_filepath+Region+'_scaler.pkl', 'rb'))
#         scaled = scaler.transform(forecast_data)
#         x_forecast = pd.DataFrame(scaled, columns = forecast_data.columns)



#          #make predictions and rescale
#         y_forecast = (model.predict(x_forecast))
#         y_forecast[y_forecast < 0 ] = 0
#         y_forecast = (SWEmax * y_forecast)
#         #remove forecasts less than 0.5 inches SWE
#         y_forecast[y_forecast < 0.5 ] = 0
#         self.Forecast[Region][self.date] = y_forecast
         

# #        if self.plot == True:
# #            #plot predictions    
# #            plt.scatter( self.Forecast[Region]['elevation_m'],self.Forecast[Region][self.date], s=5, color="blue", label="Predictions")
# #            plt.xlabel('elevation m')
# #            plt.ylabel('Predicted SWE')
# #            plt.legend()


#             #plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
# #            plt.title(Region)
# #            plt.show()


#             #plot geolocation information
# #            _geom = [Point(xy) for xy in zip(self.Forecast[Region]['Long'], self.Forecast[Region]['Lat'])]
# #            _geom_df = gpd.GeoDataFrame(self.Forecast[Region], crs="EPSG:4326", geometry=_geom)

#  #           dfmax = max(self.Forecast[Region][self.date])*1.05

#             # fig, ax = plt.subplots(figsize=(14,6))
#  #           ax = _geom_df.plot(self.date, cmap="cool", markersize=30,figsize=(25,25), legend=True, vmin=0, vmax=dfmax)#vmax=test_preds['delta'].max(), vmin=test_preds['delta'].min())
#  #           cx.add_basemap(ax, alpha = .7, crs=_geom_df.crs.to_string())

# #
#  #           plt.show()
        
#         return self.Forecast[Region]
    
#     # construct a full grid
    def expand_grid(self, lat, lon):
        '''list all combinations of lats and lons using expand_grid(lat,lon)'''
        test = [(A,B) for A in lat for B in lon]
        test = np.array(test)
        test_lat = test[:,0]
        test_lon = test[:,1]
        full_grid = pd.DataFrame({'Long': test_lon, 'Lat': test_lat})
        full_grid = full_grid.sort_values(by=['Lat','Long'])
        full_grid = full_grid.reset_index(drop=True)
        return full_grid


    def netCDF(self, plot):

        #get all SWE regions data into one DF

        self.NA_SWE = pd.DataFrame()
        columns = ['Long', 'Lat', 'elevation_m', 'northness', self.date]

        dfs_to_concat = []

        for region in self.Forecast:
            dfs_to_concat.append(self.Forecast[region][columns])

        self.NA_SWE = pd.concat(dfs_to_concat)

        self.NA_SWE = self.NA_SWE.rename(columns={self.date: 'SWE'})
        

        #round to 2 decimals
        self.NA_SWE['Lat'] = round(self.NA_SWE['Lat'],2)
        self.NA_SWE['Long'] = round(self.NA_SWE['Long'],2)

        #NA_SWE = NA_SWE.set_index('Date')

        #Get the range of lat/long to put into xarray
        self.lonrange = np.arange(min(self.NA_SWE['Long'])-1, max(self.NA_SWE['Long'])+2, 0.01)
        self.latrange = np.arange(min(self.NA_SWE['Lat'])-1, max(self.NA_SWE['Lat']), 0.01)

        self.lonrange = [round(num, 2) for num in self.lonrange]
        self.latrange = [round(num, 2) for num in self.latrange]


        #Make grid of lat long
        FG = self.expand_grid(self.latrange, self.lonrange)

        #Merge SWE predictions with gridded df
        self.DFG = pd.merge(FG, self.NA_SWE, on = ['Long','Lat'], how = 'left')

        #drop duplicate lat/long
        self.DFG = self.DFG.drop_duplicates(subset = ['Long', 'Lat'], keep = 'last').reset_index(drop = True)
        
        #fill NaN values with 0
        self.DFG['SWE'] = self.DFG['SWE'].fillna(0)

        #Reshape DFG DF
        target_variable_2D = self.DFG['SWE'].values.reshape(1,len(self.latrange),len(self.lonrange))

        #put into xarray formate
        target_variable_xr = xr.DataArray(target_variable_2D, coords=[('lat', self.latrange),('lon', self.lonrange)])

        #set target variable name
        target_variable_xr = target_variable_xr.rename("SWE")

        #save as netCDF
        target_variable_xr.to_netcdf(self.cwd+'//'+self.area+'/Data/NetCDF/SWE_MAP_1km_'+self.date+'.nc')
        
        #show plot
        print('File conversion to netcdf complete')
        
        if plot == True:
            print('Plotting results')
            self.plot_netCDF()


    def netCDF2(self, plot):

        #get all SWE regions data into one DF

        self.NA_SWE = pd.DataFrame()
        columns = ['Long', 'Lat', 'elevation_m', 'northness', self.date]

        dfs_to_concat = []

        for region in self.Forecast:
            dfs_to_concat.append(self.Forecast[region][columns])

        self.NA_SWE = pd.concat(dfs_to_concat)

        self.NA_SWE = self.NA_SWE.rename(columns={self.date: 'SWE'})
        

        #round to 2 decimals
        self.NA_SWE['Lat'] = round(self.NA_SWE['Lat'],2)
        self.NA_SWE['Long'] = round(self.NA_SWE['Long'],2)

        #NA_SWE = NA_SWE.set_index('Date')

        #Get the range of lat/long to put into xarray
        self.lonrange = np.arange(min(self.NA_SWE['Long'])-1, max(self.NA_SWE['Long'])+2, 0.01)
        self.latrange = np.arange(min(self.NA_SWE['Lat'])-1, max(self.NA_SWE['Lat']), 0.01)

        self.lonrange = [round(num, 2) for num in self.lonrange]
        self.latrange = [round(num, 2) for num in self.latrange]


        #Make grid of lat long
        FG = self.expand_grid(self.latrange, self.lonrange)

        #Merge SWE predictions with gridded df
        self.DFG = pd.merge(FG, self.NA_SWE, on = ['Long','Lat'], how = 'left')

        #drop duplicate lat/long
        self.DFG = self.DFG.drop_duplicates(subset = ['Long', 'Lat'], keep = 'last').reset_index(drop = True)
        
        #fill NaN values with 0
        self.DFG['SWE'] = self.DFG['SWE'].fillna(0)

        #Reshape DFG DF
        self.SWE_array = self.DFG['SWE'].values.reshape(1,len(self.latrange),len(self.lonrange))

       # create nc filepath
        fn = self.cwd+'//'+self.area+'/Data/NetCDF/SWE_MAP_1km_'+self.date+'.nc'
        
        # make nc file, set lat/long, time
        ds = nc.Dataset(fn, 'w', format = 'NETCDF4')
        lat = ds.createDimension('lat', len(self.latrange))
        lon = ds.createDimension('lon', len(self.lonrange)) 
        time = ds.createDimension('time', None)
        
        #make nc file metadata
        ds.title = 'SWE interpolation for ' + self.date

        lat = ds.createVariable('lat', np.float32, ('lat',))
        lat.units = 'degrees_north'
        lat.long_name = 'latitude'

        lon = ds.createVariable('lon', np.float32, ('lon',))
        lon.units = 'degrees_east'
        lon.long_name = 'longitude'

        time = ds.createVariable('time', np.float64, ('time',))
        time.units = 'hours since 1800-01-01'
        time.long_name = 'time'

        SWE = ds.createVariable('SWE', np.float64, ('time', 'lat', 'lon',))
        SWE.units = 'inches'
        SWE.standard_name = 'snow_water_equivalent'
        SWE.long_name = 'Interpolated SWE product @1-km'

        #add projection information
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(4326) # GCS_WGS_1984
        SWE.esri_pe_string = proj.ExportToWkt()

        #set lat lon info in file
        SWE.coordinates = 'lon lat'
        

        # Write latitudes, longitudes.
        lat[:] = self.latrange
        lon[:] = self.lonrange

        # Write the data.  This writes the whole 3D netCDF variable all at once.
        SWE[:,:,:] = self.SWE_array 
        
        #Set date/time information
        times_arr = time[:]
        dates = [dt.datetime(int(self.date[0:4]),int(self.date[5:7]),int(self.date[8:]),0)]
        times = date2num(dates, time.units)
        time[:] = times
        
        print(ds)
        ds.close()
        print('File conversion to netcdf complete')
        
        if plot == True:
            print('Plotting results')
            self.plot_netCDF()
                   
            
    def netCDF_CONUS(self, plot):

        #get all SWE regions data into one DF

        self.NA_SWE = pd.DataFrame()
        columns = ['Long', 'Lat', 'elevation_m', 'northness', self.date]

        dfs_to_concat = []

        for region in self.Forecast:
            dfs_to_concat.append(self.Forecast[region][columns])

        self.NA_SWE = pd.concat(dfs_to_concat)

        self.NA_SWE = self.NA_SWE.rename(columns={self.date: 'SWE'})
        

        #round to 2 decimals
        self.NA_SWE['Lat'] = round(self.NA_SWE['Lat'],2)
        self.NA_SWE['Long'] = round(self.NA_SWE['Long'],2)

        #NA_SWE = NA_SWE.set_index('Date')

        #Get the range of lat/long to put into xarray
        self.lonrange = np.arange(-124.75, -66.95, 0.01)
        self.latrange = np.arange(25.52, 49.39, 0.01)

        self.lonrange = [round(num, 2) for num in self.lonrange]
        self.latrange = [round(num, 2) for num in self.latrange]

         #Make grid of lat long
        FG = self.expand_grid(self.latrange, self.lonrange)

        #Merge SWE predictions with gridded df
        self.DFG = pd.merge(FG, self.NA_SWE, on = ['Long','Lat'], how = 'left')

        #drop duplicate lat/long
        self.DFG = self.DFG.drop_duplicates(subset = ['Long', 'Lat'], keep = 'last').reset_index(drop = True)
        
        #fill NaN values with 0
        self.DFG['SWE'] = self.DFG['SWE'].fillna(0)

        #Reshape DFG DF
        self.SWE_array = self.DFG['SWE'].values.reshape(1,len(self.latrange),len(self.lonrange))

       # create nc filepath
        fn = self.cwd+'//'+self.area+'/Data/NetCDF/SWE_MAP_1km_'+self.date+'_CONUS.nc'
        
        # make nc file, set lat/long, time
        ds = nc.Dataset(fn, 'w', format = 'NETCDF4')
        lat = ds.createDimension('lat', len(self.latrange))
        lon = ds.createDimension('lon', len(self.lonrange)) 
        time = ds.createDimension('time', None)
        
        #make nc file metadata
        ds.title = 'SWE interpolation for ' + self.date

        lat = ds.createVariable('lat', np.float32, ('lat',))
        lat.units = 'degrees_north'
        lat.long_name = 'latitude'

        lon = ds.createVariable('lon', np.float32, ('lon',))
        lon.units = 'degrees_east'
        lon.long_name = 'longitude'

        time = ds.createVariable('time', np.float64, ('time',))
        time.units = 'hours since 1800-01-01'
        time.long_name = 'time'

        SWE = ds.createVariable('SWE', np.float64, ('time', 'lat', 'lon',))
        SWE.units = 'inches'
        SWE.standard_name = 'snow_water_equivalent'
        SWE.long_name = 'Interpolated SWE product @1-km'

        #add projection information
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(4326) # GCS_WGS_1984
        SWE.esri_pe_string = proj.ExportToWkt()

        #set lat lon info in file
        SWE.coordinates = 'lon lat'
        

        # Write latitudes, longitudes.
        lat[:] = self.latrange
        lon[:] = self.lonrange

        # Write the data.  This writes the whole 3D netCDF variable all at once.
        SWE[:,:,:] = self.SWE_array 
        
        #Set date/time information
        times_arr = time[:]
        dates = [dt.datetime(int(self.date[0:4]),int(self.date[5:7]),int(self.date[8:]),0)]
        times = date2num(dates, time.units)
        time[:] = times
        
        # print(ds)
        ds.close()
        print('File conversion to netcdf complete')
        
        if plot == True:
            print('Plotting results')
            self.plot_netCDF()
                    
    #https://unidata.github.io/netcdf4-python/
    def netCDF_compressed(self, plot):

       #get all SWE regions data into one DF

        self.NA_SWE = pd.DataFrame()
        columns = ['Long', 'Lat', 'elevation_m', 'northness', self.date]

        dfs_to_concat = []

        for region in self.Forecast:
            dfs_to_concat.append(self.Forecast[region][columns])

        self.NA_SWE = pd.concat(dfs_to_concat)

        self.NA_SWE = self.NA_SWE.rename(columns={self.date: 'SWE'})


        #round to 2 decimals
        self.NA_SWE['Lat'] = round(self.NA_SWE['Lat'],2)
        self.NA_SWE['Long'] = round(self.NA_SWE['Long'],2)

        #NA_SWE = NA_SWE.set_index('Date')

        #Get the range of lat/long to put into xarray
        self.lonrange = np.arange(-124.75, -66.95, 0.01)
        self.latrange = np.arange(25.52, 49.39, 0.01)

        self.lonrange = [round(num, 2) for num in self.lonrange]
        self.latrange = [round(num, 2) for num in self.latrange]

         #Make grid of lat long
        FG = self.expand_grid(self.latrange, self.lonrange)

        #Merge SWE predictions with gridded df
        self.DFG = pd.merge(FG, self.NA_SWE, on = ['Long','Lat'], how = 'left')

        #drop duplicate lat/long
        self.DFG = self.DFG.drop_duplicates(subset = ['Long', 'Lat'], keep = 'last').reset_index(drop = True)

        #fill NaN values with 0
        #self.DFG['SWE'] = self.DFG['SWE'].fillna(0)

        #Reshape DFG DF
        self.SWE_array = self.DFG['SWE'].values.reshape(1,len(self.latrange),len(self.lonrange))

        # create nc filepath
        fn = self.cwd +'//'+self.area+'/Data/NetCDF/SWE_'+self.date+'_compressed.nc'

        # make nc file, set lat/long, time
        ncfile  = nc.Dataset(fn, 'w', format = 'NETCDF4')
        # print(ncfile)

        #Create ncfile group
        grp1 = ncfile.createGroup('SWE_1-km')

        # for grp in ncfile.groups.items():
        #     print(grp)


        lat = ncfile.createDimension('lat', len(self.latrange))
        lon = ncfile.createDimension('lon', len(self.lonrange)) 
        time = ncfile.createDimension('time', None)

        #make nc file metadata
        grp1.title = 'SWE interpolation for ' + self.date

        lat = ncfile.createVariable('lat', np.float32, ('lat',))
        lat.units = 'degrees_north'
        lat.long_name = 'latitude'

        lon = ncfile.createVariable('lon', np.float32, ('lon',))
        lon.units = 'degrees_east'
        lon.long_name = 'longitude'

        time = ncfile.createVariable('time', np.float64, ('time',))
        time.units = 'hours since 1800-01-01'
        time.long_name = 'time'

        SWE = grp1.createVariable('SWE', np.float64, ('time', 'lat', 'lon'), zlib = True)
        # for grp in ncfile.groups.items():
        #     print(grp)

        SWE.units = 'inches'
        SWE.standard_name = 'snow_water_equivalent'
        SWE.long_name = 'Interpolated SWE product @1-km'

        #add projection information
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(4326) # GCS_WGS_1984
        SWE.esri_pe_string = proj.ExportToWkt()

        #set lat lon info in file
        SWE.coordinates = 'lon lat'


        # Write latitudes, longitudes.
        lat[:] = self.latrange
        lon[:] = self.lonrange
       

        # Write the data.  This writes the whole 3D netCDF variable all at once.
        SWE[:,:,:] = self.SWE_array 

        #Set date/time information
        times_arr = time[:]
        dates = [dt.datetime(int(self.date[0:4]),int(self.date[5:7]),int(self.date[8:]),0)]
        times = date2num(dates, time.units)
        time[:] = times

        print(ncfile)
        ncfile.close()
        print('File conversion to netcdf complete')

        if plot == True:
            print('Plotting results')
            self.plot_netCDF()
         
        
    def plot_netCDF(self):
        
        #set up colormap that is transparent for zero values
        # get colormap
        ncolors = 256
        color_array = plt.get_cmap('viridis')(range(ncolors))

        # change alpha values
        color_array[:,-1] = np.linspace(0.0,1.0,ncolors)

        # create a colormap object
        map_object = LinearSegmentedColormap.from_list(name='viridis_alpha',colors=color_array)

        # register this new colormap with matplotlib
        plt.register_cmap(cmap=map_object)

        
        
        
        #load file
        fn = self.cwd+'//'+self.area+'/Data/NetCDF/SWE_MAP_1km_'+ self.date+'.nc'
        SWE = nc.Dataset(fn)

        #Get area of interest
        lats = SWE.variables['lat'][:]
        lons = SWE.variables['lon'][:]
        swe = SWE.variables['SWE'][:]

        #get basemap
        plt.figure(figsize=(20,10))
        map = Basemap(projection='merc',llcrnrlon=-130.,llcrnrlat=30,urcrnrlon=-100,urcrnrlat=50.,resolution='i')

        map.drawcoastlines()
        map.drawstates()
        map.drawcountries()
        map.drawlsmask(land_color='Linen', ocean_color='#CCFFFF') # can use HTML names or codes for colors
        map.drawcounties()

        #put lat / long into appropriate projection grid
        lons, lats = np.meshgrid(lons, lats)
        x,y = map(lons, lats)
        map.pcolor(x, y,swe, cmap= map_object)
        plt.colorbar()
            
    #produce an interactive plot using Folium
    def plot_interactive(self, pinlat, pinlong, web):
        
        #set up colormap that is transparent for zero values
        # get colormap
        ncolors = 256
        color_array = plt.get_cmap('viridis')(range(ncolors))

        # change alpha values
        color_array[:,-1] = np.linspace(0.0,1.0,ncolors)

        # create a colormap object
        map_object = LinearSegmentedColormap.from_list(name='viridis_alpha',colors=color_array)

        # register this new colormap with matplotlib
        plt.register_cmap(cmap=map_object)
        
        
        #load file
        fn = self.cwd+'//'+self.area+'/Data/NetCDF/SWE_'+self.date+'_compressed.nc'
        
        #open netcdf file with rioxarray
        xr = rxr.open_rasterio(fn)

        xr.rio.write_crs("epsg:4326", inplace=True)

        #replace x and y numbers with coordinate rangres
        xr.coords['x'] = self.lonrange
        xr.coords['y'] = self.latrange

        # Create a variable for destination coordinate system 
        dst_crs = 'EPSG:4326' 

        #scale the array from 0 - 255
        scaled_xr = es.bytescale(xr.values[0])
        
        #set max for color map
        maxSWE = xr.values[0].max()
        #set color range for color map
        SWErange = np.arange(0,maxSWE+1, maxSWE/5).tolist()

        f = folium.Figure(width=750, height=500)
        m = folium.Map(location=[pinlat, pinlong],
               tiles = 'Stamen Terrain', zoom_start = 6, control_scale=True).add_to(f)

        #map bounds, must be minutally adjusted for correct lat/long placement
        map_bounds = [[min(self.latrange)-0.86, min(self.lonrange)], 
                      [max(self.latrange)-0.86, max(self.lonrange), 0.01]]

        rasterlayer = folium.FeatureGroup(name = 'SWE')

        rasterlayer.add_child(folium.raster_layers.ImageOverlay(
                                image=scaled_xr,
                                bounds=map_bounds,
                                interactive=True,
                                cross_origin=False,
                                zindex=1,
                                colormap=map_object,
                                opacity=1
                                    ))
        
        #add colorbar
        colormap = cm.LinearColormap(colors=['violet', 'darkblue', 'blue', 'cyan', 'green', 'yellow'],
                                     index=SWErange, vmin=0.1, vmax=xr.values[0].max(),
                                     caption='Snow Water Equivalent (SWE) in inches')

        m.add_child(rasterlayer)
        m.add_child(folium.LayerControl())
        m.add_child(colormap)
        
        #code for webbrowser app
        if web == True:
            output_file =  self.cwd+'//'+self.area+'/Data/NetCDF/SWE_'+self.date+'.html'
            m.save(output_file)
            webbrowser.open(output_file, new=2)
            
        else:
            display(m)
            
        xr.close()
      
    #produce an interactive plot using Folium
    def plot_interactive_SWE(self, pinlat, pinlong, web):
        print('loading file')
        fnConus = self.cwd+'//'+self.area+'/Data/NetCDF/SWE_'+self.date+'_compressed.nc'

        #xr = rxr.open_rasterio(fn)
        xrConus = rxr.open_rasterio(fnConus)
        
        #Convert rxr df to geodataframe
        x, y, elevation = xrConus.x.values, xrConus.y.values, xrConus.values
        x, y = np.meshgrid(x, y)
        x, y, elevation = x.flatten(), y.flatten(), elevation.flatten()

        print("Converting to GeoDataFrame...")
        SWE_pd = pd.DataFrame.from_dict({'SWE': elevation, 'x': x, 'y': y})
        SWE_threshold = 0.1
        SWE_pd = SWE_pd[SWE_pd['SWE'] > SWE_threshold]
        SWE_gdf = gpd.GeoDataFrame(
            SWE_pd, geometry=gpd.points_from_xy(SWE_pd.x, SWE_pd.y), crs=4326)

        SWE_gdf.geometry = SWE_gdf.geometry.buffer(0.01, cap_style=3)
        SWE_gdf.geometry = SWE_gdf.geometry.to_crs(epsg= 4326)

        SWE_gdf =  SWE_gdf.reset_index(drop = True)
        SWE_gdf['geoid'] = SWE_gdf.index.astype(str)
        Chorocols = ['geoid', 'SWE', 'geometry']
        SWE_gdf = SWE_gdf[Chorocols]
        SWE_gdf.crs = CRS.from_epsg(4326)

        print('File conversion complete, creating mapping instance')
        # Create a Map instance
        f = folium.Figure(width=750, height=500)
        m = folium.Map(location=[pinlat, pinlong], tiles = 'Stamen Terrain', zoom_start=6, 
                       control_scale=True).add_to(f)

        # Plot a choropleth map
        # Notice: 'geoid' column that we created earlier needs to be assigned always as the first column
        folium.Choropleth(
            geo_data=SWE_gdf,
            name='SWE estimates',
            data=SWE_gdf,
            columns=['geoid', 'SWE'],
            key_on='feature.id',
            fill_color='YlGnBu_r',
            fill_opacity=0.7,
            line_opacity=0.2,
            line_color='white', 
            line_weight=0,
            highlight=False, 
            smooth_factor=1.0,
            #threshold_scale=[100, 250, 500, 1000, 2000],
            legend_name= 'SWE in inches for '+ self.date).add_to(m)

        # Convert points to GeoJson
        folium.features.GeoJson(SWE_gdf,  
                                name='Snow Water Equivalent',
                                style_function=lambda x: {'color':'transparent','fillColor':'transparent','weight':0},
                                tooltip=folium.features.GeoJsonTooltip(fields=['SWE'],
                                                                        aliases = ['Snow Water Equivalent (in) for '+ self.date+ ':'],
                                                                        labels=True,
                                                                        sticky=True,
                                                                         localize=True
                                                                                    )
                               ).add_to(m)

        
         #code for webbrowser app
        if web == True:
            output_file =  self.cwd+'//'+self.area+'/Data/NetCDF/SWE_'+self.date+'_Interactive.html'
            m.save(output_file)
            webbrowser.open(output_file, new=2)

        else:
            display(m)
            
        xrConus.close()
           
   
    def Geo_df(self):
        print('loading file')
        fnConus = self.cwd+'//'+self.area+'/Data/NetCDF/SWE_'+self.date+'_compressed.nc'

       #requires the netCDF4 package rather than rioxarray
        xrConus = nc.Dataset(fnConus)
        
        #Convert rxr df to geodataframe
        x, y, SWE = xrConus.variables['lon'][:], xrConus.variables['lat'][:], xrConus.groups['SWE_1-km']['SWE'][:]
        x, y = np.meshgrid(x, y)
        x, y, SWE = x.flatten(), y.flatten(), SWE.flatten()
        SWE = np.ma.masked_invalid(SWE).filled(0)

        print("Converting to GeoDataFrame...")
        SWE_pd = pd.DataFrame.from_dict({'SWE': SWE, 'x': x, 'y': y})
        SWE_threshold = 0.1
        SWE_pd = SWE_pd[SWE_pd['SWE'] > SWE_threshold]
        SWE_gdf = gpd.GeoDataFrame(
            SWE_pd, geometry=gpd.points_from_xy(SWE_pd.x, SWE_pd.y), crs=4326)

        SWE_gdf.geometry = SWE_gdf.geometry.buffer(0.01, cap_style=3)
        SWE_gdf.geometry = SWE_gdf.geometry.to_crs(epsg= 4326)

        SWE_gdf =  SWE_gdf.reset_index(drop = True)
        SWE_gdf['geoid'] = SWE_gdf.index.astype(str)
        Chorocols = ['geoid', 'SWE', 'geometry']
        self.SWE_gdf = SWE_gdf[Chorocols]
        self.SWE_gdf.crs = CRS.from_epsg(4326)
        
        xrConus.close()
        
    #produce an interactive plot using Folium
    def plot_interactive_SWE_comp(self, pinlat, pinlong, web):
        self.Geo_df()
        
        try:
            print('File conversion complete, creating mapping instance')
            # Create a Map instance
            f = folium.Figure(width=750, height=500)
            m = folium.Map(location=[pinlat, pinlong], tiles = 'Stamen Terrain', zoom_start=6, 
                           control_scale=True).add_to(f)

            # Plot a choropleth map
            # Notice: 'geoid' column that we created earlier needs to be assigned always as the first column
            folium.Choropleth(
                geo_data=self.SWE_gdf,
                name='SWE estimates',
                data=self.SWE_gdf,
                columns=['geoid', 'SWE'],
                key_on='feature.id',
                fill_color='YlGnBu_r',
                fill_opacity=0.7,
                line_opacity=0.2,
                line_color='white', 
                line_weight=0,
                highlight=False, 
                smooth_factor=1.0,
                #threshold_scale=[100, 250, 500, 1000, 2000],
                legend_name= 'SWE in inches for '+ self.date).add_to(m)

            # Convert points to GeoJson
            folium.features.GeoJson(self.SWE_gdf,  
                                    name='Snow Water Equivalent',
                                    style_function=lambda x: {'color':'transparent','fillColor':'transparent','weight':0},
                                    tooltip=folium.features.GeoJsonTooltip(fields=['SWE'],
                                                                            aliases = ['Snow Water Equivalent (in) for '+ self.date+ ':'],
                                                                            labels=True,
                                                                            sticky=True,
                                                                             localize=True
                                                                                        )
                                   ).add_to(m)


             #code for webbrowser app

            if web == True:
                output_file =  self.cwd+'//'+self.area+'/Data/NetCDF/SWE_'+self.date+'_Interactive.html'
                m.save(output_file)
                webbrowser.open(output_file, new=2)

            else:
                display(m)

        except IndexError:
            print("No modeled SWE")
        
    #Get a list of HUCs with snow    
    def huc_list(self, huc):
        West_HU = ['HU10', 'HU11' ,'HU13' ,'HU14' ,'HU15' ,'HU16' ,'HU17' ,'HU18']
        HUC_list = []

        for i in West_HU:

            HUC = self.HUC_SWE_read(i, huc)

            HUC_list = HUC_list + [int(i) for i in HUC]

        self.HUC_list = [int(i) for i in HUC_list]
        

    def get_HUC_Info(self, HUC):
    
        HUC_df = pd.DataFrame()

        print('Retrieving HUC watershed boundary geodataframes.')

        pbar = ProgressBar()
        for i in pbar(np.arange(0,len(HUC),1)):
            HUC[i] = str(HUC[i])

            #if len(HUC[i]) % 2 == 0:

            HU = HUC[i][:2]

            HUC_len = len(HUC[i])

            #else:
             #   HU = '0' + HUC[i][:1]
            #
             #   HUC_len = len(HUC[i])+1



            HUCunit = 'WBDHU'+ str(HUC_len)

            HUCunit2 = 'huc'+str(HUC_len)

            gdb_file = self.cwd+'//'+self.area+'/Data/WBD//WBD_' +HU+'_HU2_GDB/WBD_'+HU+'_HU2_GDB.gdb'

            # Get HUC unit from the .gdb file 

            H = gpd.read_file(gdb_file,layer=HUCunit)


            h = H[H[HUCunit2]==HUC[i]]
            HUC_df = HUC_df.append(h)
        HUC_df.reset_index(inplace = True, drop = True)
        #display(HUC_df)
        return HUC_df
          
    #These HUCS contain SWE    
    def HUC_SWE_read(self, HU, HUC):
        H = h5py.File(self.cwd+'//'+self.area+'/Data/WBD/WBD_HUC_SWE.h5','r')
        H_SWE = H[HU]['HUC8'][:].tolist()
        H_SWE= [str(i) for i in H_SWE]
        H.close()
        return H_SWE
    
    #This is just a function to identify key sites, not used in operations
    def HUC_SWE(self, df, HU, HUC):
        print('Saving Key HUCs containing SWE to speed up spatial aggregation of geodataframes')
        HU_wSWE = list(df['geoid'])
        HU_wSWE = [int(i) for i in HU_wSWE]
        f = h5py.File(self.cwd+'//'+self.area+'/Data/WBD/WBD_HUC_SWE.h5','a')
        HU10 = f.create_group(HU)
        HUC8 = HU10.create_dataset(HUC, data = HU_wSWE)
        f.close()
        
   #This function gets all of the HUCs for a list of HUs at a specified level, ie HU09:HU11 at huc8 '8'
    def get_HU_sites(self, HU, HU_level):

        HUs = pd.DataFrame()
        HUCunit = 'WBDHU'+ HU_level
        HU_level = 'huc'+HU_level

        #for i in np.arange(0,len(HU),1):


        gdb_file = self.cwd+'//'+self.area+'/Data/WBD//WBD_' +HU[2:] +'_HU2_GDB/WBD_'+HU[2:]+'_HU2_GDB.gdb'

        # Get HUC unit from the .gdb file 

        H = gpd.read_file(gdb_file,layer=HUCunit)

        HUs = HUs.append(pd.DataFrame(H[HU_level]))

        HUs = list(HUs['huc8'])

        HUs = [int(i) for i in HUs]

        return HUs
    
    '''
    Use the below code interactively to identify the HUCs with SWE
    '''
    #f = h5py.File(cwd+'/Data/WBD/WBD_HUC_SWE.h5','w')
    #f.close()

    #West_HU = ['HU10', 'HU11' ,'HU13' ,'HU14' ,'HU15' ,'HU16' ,'HU17' ,'HU18']

    #for HU in West_HU:
    #   print('Getting ', HU, ' HUCs')
    #  West_HU_huc8 = get_HU_sites(HU, '8')
    # HUC_SWE_mean = get_Mean_HUC_SWE(West_HU_huc8, Snow.SWE_gdf)
        #HUC_SWE(HUC_SWE_mean, HU , 'HUC8')

    
    #Get mean swe per HUC and convert to GeoDataFrame
    def get_Mean_HUC_SWE(self):
    
        self.SWE_gdf['centroid'] = self.SWE_gdf['geometry'].centroid
        HUC_df = self.get_HUC_Info(self.HUC_list)

        HUC_SWE_df = pd.DataFrame()

        print('Calculating mean SWE per HUC')

        pbar = ProgressBar()
        for i in pbar(np.arange(0,len(HUC_df),1)):

            HU = self.HUC_list[i][:2]

            HUC_len = len(self.HUC_list[i])

            HUCunit2 = 'huc'+str(HUC_len)

            huc = gpd.GeoDataFrame(pd.DataFrame(HUC_df.iloc[i]).T)        
            joined = gpd.sjoin(left_df=huc, right_df=self.SWE_gdf, how='left')
            #print(joined.columns)

            #display(HUC[i])
            HUC_SWE = joined[joined[HUCunit2] == str(self.HUC_list[i])]

            HUC_SWE_mean = HUC_SWE.copy()

            HUC_SWE_mean['Mean_SWE'] = np.mean(HUC_SWE_mean['SWE'])

            del HUC_SWE_mean['SWE']

            HUC_SWE_mean['geoid'] = self.HUC_list[i]

            HUC_mean_cols = ['geoid', 'Mean_SWE', 'geometry']

            HUC_SWE_mean = HUC_SWE_mean[HUC_mean_cols].drop_duplicates()

            HUC_SWE_df = HUC_SWE_df.append(HUC_SWE_mean)

        HUC_SWE_df.crs = "EPSG:4326"

        HUC_SWE_df.dropna(inplace = True)

        self.HUC_SWE_df = HUC_SWE_df
        
        


        print("Converting to GeoDataFrame...")
        target = 'Mean_SWE'
        self.HUC_SWE_df.geometry = self.HUC_SWE_df.geometry.to_crs(epsg= 4326)

        self.HUC_SWE_df =  self.HUC_SWE_df.reset_index(drop = True)
        #SWE_gdf['geoid'] = SWE_gdf.index.astype(str)
        Chorocols = ['geoid', target, 'geometry']
        self.HUC_SWE_df = self.HUC_SWE_df[Chorocols]
        self.HUC_SWE_df.crs = CRS.from_epsg(4326)
        
     
    def trunc(self, values, decs=0):
        return np.trunc(values*10**decs)/(10**decs)
             
        
    def GeoDF_HUC_NetCDF_compressed(self):
        print('Geodataframe conversion')
        #make point for all USA
        #Get the range of lat/long to put into xarray
        lonrange = np.arange(-124.75, -66.95, 0.01)
        latrange = np.arange(25.52, 49.39, 0.01)

        lonrange = [round(num, 2) for num in lonrange]
        latrange = [round(num, 2) for num in latrange]

         #Make grid of lat long
        FG = self.expand_grid(latrange, lonrange)

        GFG = gpd.GeoDataFrame(FG, geometry = gpd.points_from_xy(FG.Long, FG.Lat))

        #merge multipoint with point
        GFGjoin = GFG.sjoin(self.HUC_SWE_df, how ='inner', predicate = 'intersects')

        #Select key columns
        cols = ['Long', 'Lat', 'geoid', 'Mean_SWE']
        MSWE = GFGjoin[cols]

        #Merge SWE predictions with gridded df
        DFG = pd.merge(FG, MSWE, on = ['Long','Lat'], how = 'left')

        #drop duplicate lat/long
        DFG = DFG.drop_duplicates(subset = ['Long', 'Lat'], keep = 'last').reset_index(drop = True)


        #Reshape DFG DF
        SWE_array = DFG['Mean_SWE'].values.reshape(1,len(latrange),len(lonrange))

        # create nc filepath
        fn = self.cwd+'//'+self.area+'/Data/NetCDF/SWE_'+self.date+'_compressed.nc'
        print('Setting up NetCDF4')
        # make nc file, set lat/long, time
        ncfile  = nc.Dataset(fn, 'a', format = 'NETCDF4')
        
        #Create ncfile group
        grp2 = ncfile.createGroup('HUC8')

        #for grp in ncfile.groups.items():
         #   print(grp)


       # lat = ncfile.createDimension('lat', len(latrange))
       # lon = ncfile.createDimension('lon', len(lonrange)) 
        #time = ncfile.createDimension('time', None)

        #make nc file metadata
        grp2.title = 'HUC SWE estimate for ' + self.date

        #lat = ncfile.createVariable('lat', np.float32, ('lat',))
        #lat.units = 'degrees_north'
        #lat.long_name = 'latitude'

        #lon = ncfile.createVariable('lon', np.float32, ('lon',))
        #lon.units = 'degrees_east'
        #lon.long_name = 'longitude'

        #time = ncfile.createVariable('time', np.float64, ('time',))
        #time.units = 'hours since 1800-01-01'
        #time.long_name = 'time'

        Mean_SWE = grp2.createVariable('Mean_SWE', np.float64, ('time', 'lat', 'lon'), zlib = True)
        for grp in ncfile.groups.items():
            print(grp)

        Mean_SWE.units = 'inches'
        Mean_SWE.standard_name = 'snow_water_equivalent'
        Mean_SWE.long_name = 'Mean SWE product HUC8'

        #add projection information
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(4326) # GCS_WGS_1984
        Mean_SWE.esri_pe_string = proj.ExportToWkt()

        #set lat lon info in file
        Mean_SWE.coordinates = 'lon lat'


        # Write latitudes, longitudes.
       # lat[:] = latrange
        #lon[:] = lonrange


        # Write the data.  This writes the whole 3D netCDF variable all at once.
        Mean_SWE[:,:,:] = SWE_array 

        #Set date/time information
        #times_arr = time[:]
        #dates = [dt.datetime(int(self.date[0:4]),int(self.date[5:7]),int(self.date[8:]),0)]

        #times = date2num(dates, time.units)
        #time[:] = times

        print(ncfile)
        ncfile.close()
        print('File conversion to netcdf complete')


    def plot_interactive_SWE_comp_HUC(self, pinlat, pinlong, web):
        target = 'Mean_SWE'

        SWE_gdf = self.HUC_SWE_df

        print("Converting to GeoDataFrame...")
        SWE_gdf.geometry = SWE_gdf.geometry.to_crs(epsg= 4326)

        SWE_gdf =  SWE_gdf.reset_index(drop = True)
        #SWE_gdf['geoid'] = SWE_gdf.index.astype(str)
        Chorocols = ['geoid', target, 'geometry']
        SWE_gdf = SWE_gdf[Chorocols]
        SWE_gdf.crs = CRS.from_epsg(4326)


        print('File conversion complete, creating mapping instance')
        # Create a Map instance
        f = folium.Figure(width=750, height=500)
        m = folium.Map(location=[pinlat, pinlong], tiles = 'Stamen Terrain', zoom_start=6, 
                       control_scale=True).add_to(f)
        print('Map made, creating choropeth')

        # Plot a choropleth map
        # Notice: 'geoid' column that we created earlier needs to be assigned always as the first column
        folium.Choropleth(
            geo_data=SWE_gdf,
            name='SWE estimates',
            data=SWE_gdf,
            columns=['geoid', target],
            key_on='feature.properties.geoid',
            fill_color='YlGnBu_r',
            fill_opacity=0.7,
            line_opacity=0.1,
            line_color='black', 
            line_weight=1,
            highlight=False, 
            smooth_factor=1.0,
            #threshold_scale=[100, 250, 500, 1000, 2000],
            legend_name= 'SWE in inches for '+ self.date).add_to(m)

        print('Choropeth complete, adding features')

        # Convert points to GeoJson
        folium.features.GeoJson(SWE_gdf,  
                                name='Snow Water Equivalent',
                                style_function=lambda x: {'color':'transparent','fillColor':'transparent','weight':0},
                                tooltip=folium.features.GeoJsonTooltip(fields=[target],
                                                                        aliases = ['Snow Water Equivalent (in) for '+ self.date+ ':'],
                                                                        labels=True,
                                                                        sticky=True,
                                                                         localize=True
                                                                                    )
                               ).add_to(m)

        print('map made, saving and deploying')

         #code for webbrowser app
        #self.SWE_gdf = SWE_gdf
        #if web == True:
        output_file =  self.cwd+'//'+self.area+'/Data/NetCDF/HUC8_Mean_SWE_'+self.date+'_HUC8.html'
        m.save(output_file)
        webbrowser.open(output_file, new=2)




### FRACTIONAL SNOW COVER FUNCTIONALITY

# class SCA():

    # def __init__(self, cwd, area, date: Union[str, datetime], delta=7, timeDelay=3, threshold=0.6):
    #     """
    #         Initializes the NSM_SCA class.

    #         Parameters:
    #             cwd (str): The current working directory.
    #             area (str): Name of the region to model. This should exactly match the name of the shapefile.
    #             date (str): The date of the prediction.
    #             delta (int): How many days back to go for Last SWE.
    #             timeDelay (int): Use the SCA rasters from [timeDelay] days ago. Simulates operations in the real world.
    #             threshold (float): The threshold for NDSI, if greater than this value, it is considered to be snow.
    #     """
    #     if type(cwd) != Path:
    #         cwd = Path(cwd)  # Convert to Path object if necessary

    #     if type(date) != datetime:
    #         date = datetime.strptime(date, "%Y-%m-%d")  # Convert to datetime object if necessary
        
    #     self.area = area
    #     self.timeDelay = timeDelay
    #     self.delayedDate = date - pd.Timedelta(days=timeDelay)

    #     self.SCA_folder = self.cwd + "//Data//VIIRS_SCA//"
    #     self.threshold = threshold * 100  # Convert percentage to values used in VIIRS NDSI

    #     self.auth = ea.login(strategy="netrc")
    #     if self.auth is None:
    #         print("Error logging into Earth Data account. Things will probably break")


    def getPredictionExtent(self):
        """
            Gets the extent of the prediction dataframe.

            Returns:
                extent (list[float, float, float, float]): The extent of the prediction dataframe.
        """
        Geo_df = pd.read_csv(self.cwd+'//'+self.area+'//'+self.area+'_Geo_df.csv', index_col='cell_id')

        Geo_df_bounds = gpd.GeoDataFrame(Geo_df, geometry=gpd.points_from_xy(Geo_df.Long, Geo_df.Lat, crs="EPSG:4326"))

        return Geo_df_bounds.total_bounds


    def initializeGranules(self, bbox: list[float, float, float, float],
                           dataFolder: Union[str, Path]):
        """
            Initalizes SCA information by fetching granules and merging them.

            Parameters:
                bbox (list[float, float, float, float]): The bounding box to fetch granules for.
                dataFolder (str): The folder with the granules.

            Returns:
                None - Initializes the following class variables: extentDF, granules, raster
        """
        self.extentDF = calculateGranuleExtent(bbox, self.delayedDate)  # Get granule extent
        self.granules = fetchGranules(bbox, dataFolder, self.delayedDate, self.extentDF)  # Fetch granules
        self.raster = createMergedRxr(self.granules["filepath"])  # Merge granules



    def augment_SCA(self, region: str):
        """
            Augments the region's forecast dataframe with SCA data.

            Parameters:
                region (str): The region to augment.

            Returns:
                adf (GeoDataFrame): The augmented dataframe.
        """

        # Load forecast dataframe
        try:
            self.Forecast  # Check if forecast dataframe has been initialized
        except AttributeError:
            #load regionalized forecast data
            ifile = bz2.BZ2File(self.cwd+'//'+self.area+'/Data/Processed/Prediction_DF_'+self.date + '.pkl','rb')
            self.Forecast = pickle.load(ifile)
            ifile.close()

        region_df = self.Forecast[region]
        geoRegionDF = gpd.GeoDataFrame(region_df, geometry=gpd.points_from_xy(region_df.Long, region_df.Lat,
                                                                              crs="EPSG:4326"))  # Convert to GeoDataFrame

        try:
            regional_raster = self.raster  # Check if raster has been initialized
        except AttributeError:
            # Fetch granules and merge them
            # region_extentDF = calculateGranuleExtent(geoRegionDF.total_bounds, self.delayedDate)  # Get granule extent TODO fix delayedDate
            region_granules = fetchGranules(geoRegionDF.total_bounds, self.SCA_folder,
                                            self.delayedDate)  # Fetch granules
            regional_raster = createMergedRxr(region_granules["filepath"])  # Merge granules

        adf = augmentGeoDF(geoRegionDF, regional_raster, buffer=500, threshold=self.threshold)  # Buffer by 500 meters -> 1km square
        # adf.drop(columns=["geometry"], inplace=True)  # Drop geometry column

        return adf


    def augmentPredictionDFs(self):
        """
            Augments the forecast dataframes with SCA data.
        """
        print("Calculating mean SCA for each geometry in each region...")
        # ifile = bz2.BZ2File(self.cwd+'//'+self.area+'/Data/Processed/Prediction_DF_SCA_'+self.date + '.pkl','rb')
        ifile = bz2.BZ2File(self.cwd+'//'+self.area+'/Data/Processed/Prediction_DF_'+self.date+'.pkl','rb')
        self.Forecast = pickle.load(ifile)
        ifile.close()

        # Augment each Forecast dataframes
        for region in tqdm(self.Region_list):
            self.Forecast[region] = self.augment_SCA(region).drop(columns=["geometry"])

        # Save augmented forecast dataframes
        ## path = self.cwd +'//'+self.area+'/Data//Processed//Prediction_DF_SCA_' + self.date + ".pkl"
        # path = self.cwd +'//'+self.area+'/Data//Processed//Prediction_DF_' + self.date +".pkl"
        # file = open(path, "wb")

        # # write the python object (dict) to pickle file
        # pickle.dump(self.Forecast, file)

        # # close file
        # file.close()

        # Save augmented forecast dataframes 
        path = bz2.BZ2File(self.cwd+'//'+self.area+'/Data/Processed/Prediction_DF_'+self.date + '.pkl', 'wb')
        
        # write the python object (dict) to pickle file
        pickle.dump(self.Forecast,path)

        # close file
        path.close()

    
    def SWE_Predict(self, SCA=True):
        # load first SWE observation forecasting dataset with prev and delta swe for observations.

        # if SCA:
        #     path = self.cwd+'//'+self.area+'/Data/Processed/Prediction_DF_SCA_' + self.date + '.pkl'
        # else:
        #     path = self.cwd+'//'+self.area+'/Data/Processed/Prediction_DF_'+self.date + '.pkl'
        path = self.cwd+'//'+self.area+'/Data/Processed/Prediction_DF_'+self.date + '.pkl'
        # #self.plot = plot
        # #load first SWE observation forecasting dataset with prev and delta swe for observations. 
        
        # if SCA:
        #     #load regionalized forecast data
        #     ifile = bz2.BZ2File(self.cwd+'//'+self.area+'/Data/Processed/Prediction_DF_SCA_'+self.date + '.pkl','rb')
        #     self.Forecast = pickle.load(ifile)
        #     ifile.close()
        # else:
        #     #load regionalized forecast data
        #     ifile = bz2.BZ2File(self.cwd+'//'+self.area+'/Data/Processed/Prediction_DF_'+self.date + '.pkl','rb')
        #     self.Forecast = pickle.load(ifile)
        #     ifile.close()

        #load regionalized forecast data
        ifile = bz2.BZ2File(self.cwd+'//'+self.area+'/Data/Processed/Prediction_DF_'+self.date + '.pkl','rb')
        self.Forecast = pickle.load(ifile)
        ifile.close()

        #load RFE optimized features
        self.Region_optfeatures= pickle.load(open(self.cwd+"/Model/Prev_SWE_Models_Final/opt_features_prevSWE.pkl", "rb"))

        #Reorder regions
        self.Forecast = {k: self.Forecast[k] for k in self.Region_list}
        
        #Make and save predictions for each reagion
        self.Prev_df = pd.DataFrame()
        self.predictions ={}
        print ('Making predictions for: ', self.date)

        if not os.path.exists(f"{self.cwd}//{self.area}//Predictions//{self.threshold}//Predictions"):
            os.makedirs(f"{self.cwd}//{self.area}//Predictions//{self.threshold}//Predictions")
        
        dfs_to_concat = []

        for Region in self.Region_list:
            print(Region)
            self.predictions[Region] = self.Predict(Region)
            self.predictions[Region] = pd.DataFrame(self.predictions[Region])
            dfs_to_concat.append(pd.DataFrame(self.predictions[Region][self.date]))

        self.Prev_df = pd.concat(dfs_to_concat, ignore_index=False) ########True?
        for Region in self.Region_list:
            self.predictions[Region].to_hdf(f"{self.cwd}//{self.area}//Predictions//{self.threshold}//Predictions//predictions{self.date}.h5", key=Region)
            self.predictions[Region].to_hdf(f"{self.cwd}//{self.area}//Predictions//predictions{self.date}.h5", key=Region)


        #load submission DF and add predictions, if locations are removed or added, this needs to be modified
        self.subdf = pd.read_csv(f"{self.cwd}//{self.area}//Predictions//{self.threshold}//Predictions//submission_format_{self.prevdate}.csv", index_col = 'cell_id')  
        # self.subdf.index = list(self.subdf.iloc[:, 0].values)
        # self.subdf = self.subdf.iloc[:, 1:]  # TODO replace with drop("cell_id")

        self.sub_index = self.subdf.index
        #reindex predictions
        self.Prev_df = self.Prev_df.loc[self.sub_index]
        self.subdf[self.date] = self.Prev_df[self.date].astype(float)
        #subdf.index.names = [' ']
        self.subdf.to_csv(f"{self.cwd}//{self.area}//Predictions//{self.threshold}//Predictions//submission_format_{self.date}.csv")
        # self.subdf.to_csv(f"{self.cwd}//{self.area}//Predictions//submission_format_{self.date}.csv")


    def Predict(self, Region, SCA=True):
        """
            Run model inference on a region

            Parameters:
                Region (str): The region to run inference on
                SCA (bool): Whether or not to use SCA data

            Returns:
                Forcast[Region] (DataFrame): The forecast df for the region
        """
        ##region specific features
        features = self.Region_optfeatures[Region]

        #Make prediction dataframe
        forecast_data = self.Forecast[Region].copy()
        
        if SCA:
            # drop all rows that have a False value in "hasSnow", i.e. no snow, so skip inference
            inference_locations = forecast_data.drop(forecast_data[~forecast_data["hasSnow"]].index)
        else:
            # keep all rows
            inference_locations = forecast_data

        forecast_data = inference_locations[features]  # Keep only features needed for inference

        if len(inference_locations) == 0:  # makes sure that we don't run inference on empty regions
            print("No snow in region: ", Region)
            self.Forecast[Region][self.date] = 0.0
        else:
            #change all na values to prevent scaling issues
            forecast_data[forecast_data< -9000]= -10

            #load and scale data

            #set up model checkpoint to be able to extract best models
            checkpoint_filepath = self.cwd + '//Model//Prev_SWE_Models_Final//' + Region + '//'
            model = checkpoint_filepath + Region + '_model.h5'
            print(model)
            model=load_model(model)

            #load SWE scaler
            SWEmax = np.load(checkpoint_filepath+Region+'_SWEmax.npy')
            SWEmax = SWEmax.item()

            #load features scaler
            #save scaler data here too
            scaler =  pickle.load(open(checkpoint_filepath + Region + '_scaler.pkl', 'rb'))
            scaled = scaler.transform(forecast_data)
            x_forecast = pd.DataFrame(scaled, columns = forecast_data.columns)

            #make predictions and rescale
            y_forecast = (model.predict(x_forecast))
            y_forecast[y_forecast < 0 ] = 0
            y_forecast = (SWEmax * y_forecast)
            #remove forecasts less than 0.5 inches SWE
            y_forecast[y_forecast < 0.5] = 0  # TODO address this with research, try smaller values/no value

            # add predictions to forecast dataframe

            self.Forecast[Region][self.date] = 0.0  # initialize column
            forecast_data[self.date] = y_forecast  # add column
            self.Forecast[Region][self.date].update(forecast_data[self.date])  # update forecast dataframe

        return self.Forecast[Region]
    

def calculateGranuleExtent(boundingBox: list[float, float, float, float],
                        day: Union[datetime, str] = datetime(2018, 7, 7)):
    """
        Fetches relevant VIIRS granules from NASA's EarthData's CMR API.

        Parameters:
            boundingBox (list[float, float, float, float]): The bounding box of the region of interest.

                lower_left_lon : lower left longitude of the box (west)
                lower_left_lat : lower left latitude of the box (south)
                upper_right_lon : upper right longitude of the box (east)
                upper_right_lat : upper right latitude of the box (north)

            day (datetime, str): The day to query granules for.

        Returns:
            cells (GeoDataFrame): A dataframe containing the horizontal and vertical tile numbers and their boundaries

    """

    if not isinstance(day, datetime):
        day = datetime.strptime(day, "%Y-%m-%d")

    # Get params situated
    datasetName = "VNP10A1F"  # NPP-SUOMI VIIRS, but JPSS1 VIIRS also exists
    version = "2" if day > datetime(2018, 1, 1) else "1"  # TODO v1 supports 2013-on, but v2 currently breaks <2018??? - RJ, V2 breaks because no data. Need to adjsut  the coverage product to '/NPP_Grid_IMG_2D/VNP10A1_NDSI_Snow_Cover'  below
    query = (ea.granule_query()
            .short_name(datasetName)
            .version(version)
            .bounding_box(*boundingBox)
            .temporal(day.strftime("%Y-%m-%d"), day.strftime("%Y-%m-%d"))
            # Grab one day's worth of data, we only care about spatial extent
            )

    results = query.get(100)  # The Western CONUS is usually 7, so this is plenty

    cells = []
    for result in results:
        geometry = shapely.geometry.Polygon(
            [(x["Longitude"], x["Latitude"]) for x in
            result["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"][
                "Points"]]
        )
        cell = {
            "h": result["umm"]["AdditionalAttributes"][1]["Values"][0],  # HORIZONTAL TILE NUMBER
            "v": result["umm"]["AdditionalAttributes"][2]["Values"][0],  # VERTICAL TILE NUMBER
            "geometry": geometry
        }
        cells.append(cell)

    geo = gpd.GeoDataFrame(cells, geometry="geometry", crs="EPSG:4326")

    return geo


def createGranuleGlobpath(dataRoot: str, date: datetime, h: int, v: int) -> str:
    """
        Creates a filepath for a VIIRS granule.

        Parameters:
            dataRoot (str): The root folder for the data.
            date (str): The date of the data.
            h (int): The horizontal tile number.
            v (int): The vertical tile number.

        Returns:
            filepath (str): The filepath of the granule.
    """
    dayOfYear = date.strftime("%Y%j")  # Format date as YearDayOfYear
    WY_split = datetime(date.year, 10, 1)  # Split water years on October 1st

    # if day is after water year, then we need to adjust the year
    if date.month < 10:
        year = date.year - 1
        next_year = date.year
    else:
        year = date.year
        next_year = date.year + 1

    #return str(Path(dataRoot, f"{year}-{next_year}NASA", f"VNP10A1F_A{dayOfYear}_h{h}v{v}_*.tif"))
    '''Changed the directory to work with user inputs'''

    return str(Path(dataRoot, f"VNP10A1F_A{dayOfYear}_h{h}v{v}_*.tif"))


def granuleFilepath(filepath: str) -> str:
    """
        return matched filepath if it exists, otherwise return empty string
    """
    result = glob.glob(filepath)
    if result:
        return result[0]  # There should only be one match
    else:
        return ''


def fetchGranules(boundingBox: list[float, float, float, float],
                dataFolder: Union[Path, str],
                date: Union[datetime, str],
                extentDF: gpd.GeoDataFrame = None,
                shouldDownload: bool = True) -> gpd.GeoDataFrame:
    """
            Fetches VIIRS granules from local storage.

            Parameters:
                boundingBox (list[float, float, float, float]): The bounding box of the region of interest. (west, south, east, north)
                date (datetime, str): The start date of the data to fetch.
                dataFolder (Path, str): The folder to save the data to, also used to check for existing data.
                extentDF (GeoDataFrame): A dataframe containing the horizontal and vertical tile numbers and their boundaries
                shouldDownload (bool): Whether to fetch the data from the API or not.

            Returns:
                df (GeoDataFrame): A dataframe of the granules that intersect with the bounding box
        """

    if extentDF is None:
        cells = calculateGranuleExtent(boundingBox, date)  # Fetch granules from API, no need to check bounding box

    else:
        # Find granules that intersect with the bounding box
        cells = extentDF.cx[boundingBox[0]:boundingBox[2],
                boundingBox[1]:boundingBox[3]]  # FIXME if there is only one point, this will fail

    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")

    if not isinstance(dataFolder, Path):
        dataFolder = Path(dataFolder)

    day = date.strftime("%Y-%m-%d")
    cells["date"] = date  # record the date
    cells["filepath"] = cells.apply(
        lambda x: granuleFilepath(createGranuleGlobpath(dataFolder, date, x['h'], x['v'])),
        axis=1
    )  # add filepath if it exists, otherwise add empty string

    return cells



def fetchGranulesRange(boundingBox: list[float, float, float, float],
                    dataFolder: str,
                    startDate: Union[datetime, str],
                    endDate: Union[datetime, str] = None,
                    extentDF: gpd.GeoDataFrame = None,
                    frequency: str = "D",
                    fetch: bool = True) -> dict:
    """
        Fetches VIIRS granules from local storage.

        Parameters:
            boundingBox (list[float, float, float, float]): The bounding box of the region of interest. (west, south, east, north)
            startDate (str): The start date of the data to fetch.
            endDate (str): The end date of the data to fetch. Defaults to same day as startDate.
            dataFolder (str): The folder to save the data to, also used to check for existing data.
            extentDF (GeoDataFrame): A dataframe containing the horizontal and vertical tile numbers and their boundaries

        Returns:
            dfs (dict): A dictionary of dataframes the granules that intersect with the bounding box by day
    """
    if type(startDate) != datetime:
        startDate = datetime.strptime(startDate, "%Y-%m-%d")  # If start date is specified, convert to datetime

    if endDate is None:
        endDate = startDate  # If no end date is specified, assume we only want one day
    elif type(endDate) != datetime:
        endDate = datetime.strptime(endDate, "%Y-%m-%d")  # If end date is specified, convert to datetime

    if extentDF is None:
        cells = calculateGranuleExtent(boundingBox, startDate)  # Fetch granules from API, no need to check bounding box
    else:
        # Find granules that intersect with the bounding box
        cells = extentDF.cx[boundingBox[0]:boundingBox[2], boundingBox[1]:boundingBox[3]]

    missing = {}
    dfs = {}
    # check and see if we already have the data for that day
    for date in pd.date_range(startDate, endDate, freq=frequency):
        # for each granule, check if we have the data
        granules = fetchGranules(boundingBox, dataFolder, date, cells, shouldDownload=False)

        missingCells = granules[granules["filepath"] == ''][["h", "v"]].to_dict("records")
        if len(missingCells) > 0:
            missing[date.strftime("%Y-%m-%d")] = missingCells
        dfs[date.strftime("%Y-%m-%d")] = granules

    if fetch and len(missing) > 0:
        print(f"Missing data for the following days: {list(missing.keys())}")

        # create strings of days to request in batches
        for datestr in missing:
            dateObj = datetime.strptime(datestr, "%Y-%m-%d")

            # TODO make list of days that are consecutive


    return dfs  # TODO theres probably a better way to store this, but you can send this to .h5 for storage


def createMergedRxr(files: list[str]) -> xr.DataArray:
    """
        Creates a merged (mosaic-ed) rasterio dataset from a list of files.

        Parameters:
            files (list[str]): A list of filepaths to open and merge.

        Returns:
            merged (DataArray): A merged DataArray.
    """
    # FIXME sometimes throws "CPLE_AppDefined The definition of geographic CRS EPSG:4035 got from GeoTIFF keys is not
    #   the same as the one from the EPSG registry, which may cause issues during reprojection operations. Set
    #   GTIFF_SRS_SOURCE configuration option to EPSG to use official parameters (overriding the ones from GeoTIFF
    #   keys), or to GEOKEYS to use custom values from GeoTIFF keys and drop the EPSG code."
    tifs = [rxr.open_rasterio(file) for file in files]  # Open all the files as Rioxarray DataArrays

    noLakes = [tif.where(tif != 237, other=0) for tif in tifs]  # replace all the lake values with 0
    noOceans = [tif.where(tif != 239, other=0) for tif in noLakes]  # replace all the ocean values with 0
    noErrors = [tif.where(tif <= 100, other=100) for tif in
                noOceans]  # replace all the other values with 100 (max Snow)
    return merge_arrays(noErrors, nodata=255)  # Merge the arrays


def augmentGeoDF(gdf: gpd.GeoDataFrame,
                raster: xr.DataArray,
                threshold: float = 20,  # TODO try 10
                noData: int = 255,
                buffer: float = None) -> gpd.GeoDataFrame:
    """
        Augments a GeoDataFrame with a raster's values.

        Parameters:
            gdf (GeoDataFrame): The GeoDataFrame to append the SCA to. Requires geometry to be an area, see buffer param
            raster (DataArray): The raster to augment the GeoDataFrame with.
            threshold (int): The threshold to use to determine if a pixel is snow or not.
            noData (int): The no data value of the raster.
            buffer (float): The buffer to use around the geometry. Set if the geometry is a point.

        Returns:
            gdf (GeoDataFrame): The augmented GeoDataFrame.
    """

    if buffer is not None:
#         buffered = gdf.to_crs("3857").buffer(buffer,
#                                             cap_style=3)  # Convert CRS to a projected CRS and buffer points into squares
#         buffered = buffered.to_crs(raster.rio.crs)  # Convert the GeoDataFrame to the same CRS as the raster
          buffered = gdf.to_crs(raster.rio.crs)
    else:
        buffered = gdf.to_crs(raster.rio.crs)  # Convert the GeoDataFrame to the same CRS as the raster

    stats = rs.zonal_stats(buffered,  # pass the buffered geometry
                        raster.values[0],  # pass the raster values as a numpy array, TODO investigate passing GDAL
                        no_data=noData,
                        affine=raster.rio.transform(),  # required for passing numpy arrays
                        stats=['mean'],  # we only want the mean, others are available if needed
                        geojson_out=False,  # we will add the result back into a GeoDataFrame, so no need for GeoJSON
                        )

    gdf["VIIRS_SCA"] = [stat['mean'] for stat in stats]  # add the mean to the GeoDataFrame
    gdf["hasSnow"] = gdf["VIIRS_SCA"] > threshold  # snow value is above 60%

    return gdf


def augment_SCA_TDF(region_df, raster, delayedDate, SCA_folder, threshold):

    '''
    Augments the region's forecast dataframe with SCA data.

    Parameters:
        region (str): The region to augment.

    Returns:
        adf (GeoDataFrame): The augmented dataframe.
    '''
    
    # region_df = Forecast[region]
    geoRegionDF = gpd.GeoDataFrame(region_df, geometry=gpd.points_from_xy(region_df.Long, region_df.Lat,
                                                                        crs="EPSG:4326"))  # Convert to GeoDataFrame

    try:
        regional_raster = raster  # Check if raster has been initialized
    except AttributeError:
        # Fetch granules and merge them
        # region_extentDF = calculateGranuleExtent(geoRegionDF.total_bounds, delayedDate)  # Get granule extent TODO fix delayedDate
        region_granules = fetchGranules(geoRegionDF.total_bounds, SCA_folder,
                                        delayedDate)  # Fetch granules
        regional_raster = createMergedRxr(region_granules["filepath"])  # Merge granules

    adf = augmentGeoDF(geoRegionDF, regional_raster, buffer=500, threshold=threshold)  # Buffer by 500 meters -> 1km square
    # adf.drop(columns=["geometry"], inplace=True)  # Drop geometry column

    return adf

