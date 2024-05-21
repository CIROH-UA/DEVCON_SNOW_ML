import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
import contextily as cx
from IPython.display import HTML
from tqdm import tqdm

def plot_single_date(cwd, area, date):
    """
    Create a single matplotlib plot of a single date model inference
        Args: 
            cwd (str): current working diectory. This should be the "National-Snow-Model" directory.
            area (str): Name of the region to model.
            date (str): 'YYYY-MM-DD' - Date of model infrence to plot.
    """
    #load SWE predictions up to the end_date
    preds=pd.read_csv(cwd+'//extrapolation//'+area+'//Predictions//submission_format_'+date+'.csv', index_col = [0])
    preds.index.names=['cell_id']

    #load feature dataframe & area shapefile
    Geo_df = pd.read_csv(cwd+'//extrapolation//'+area+'//'+area+'_Geo_df.csv', index_col='cell_id')
    shapefile_path = cwd+'//extrapolation//'+area+'//'+area+'.shp'
    gdf_shapefile = gpd.read_file(shapefile_path)

    # Make Geodataframe for plotting
    preds=Geo_df.join(preds)
    preds_gdf = gpd.GeoDataFrame(preds, geometry=gpd.points_from_xy(preds.Long, preds.Lat), crs="EPSG:4326")

    cmap = plt.cm.get_cmap('PuRd')  # Choose a colormap #'Blues' 'BuPu' 'jet' 'PuRd' 'viridis'
    
    # Create masks for zero and non-zero values
    mask_zero = preds_gdf[date] == 0
    mask_non_zero = preds_gdf[date] != 0

    date=date
    fig, ax = plt.subplots(figsize=(10, 10))
    preds_gdf[mask_non_zero].plot(ax=ax, column=date, marker='s', markersize=200, legend=True, cmap=cmap, alpha=1)
    gdf_shapefile.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
    cx.add_basemap(ax, crs=preds_gdf.crs.to_string(), source=cx.providers.Esri.WorldTerrain, alpha=0.5)

    # Set plot labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(date+' Estimated SWE [in]')

    # Display the plot
    plt.show()


def plot_multiple_dates(cwd, area, end_date, num_subplots):
    """
    Create a matplotlib subplot grid of multiple date model inference
        Args: 
            cwd (str): current working diectory. This should be the "Model" directory.
            area (str): Name of the region to model.
            end_date (str): 'YYYY-MM-DD' - Date of final model infrence to plot.
            num_subplots (int): Number of model inferences to plot. 
    """
     
    #load SWE predictions up to the end_date
    preds=pd.read_csv(cwd+'//extrapolation//'+area+'//Predictions//submission_format_'+end_date+'.csv', index_col = [0])
    preds.index.names=['cell_id']

    #load feature dataframe & area shapefile
    Geo_df = pd.read_csv(cwd+'//extrapolation//'+area+'//'+area+'_Geo_df.csv', index_col='cell_id')
    shapefile_path = cwd+'//extrapolation//'+area+'//'+area+'.shp'
    gdf_shapefile = gpd.read_file(shapefile_path)

    # Make Geodataframe for plotting
    preds=Geo_df.join(preds)
    preds_gdf = gpd.GeoDataFrame(preds, geometry=gpd.points_from_xy(preds.Long, preds.Lat), crs="EPSG:4326")

    # Calculate the number of rows and columns for the subplot layout
    num_rows = int(np.sqrt(num_subplots))
    num_cols = int(np.ceil(num_subplots / num_rows))

    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18,12))

    # Create a normalization instance for color mapping across subplots
    cmap = plt.cm.get_cmap('PuRd')  # Choose a colormap #'Blues' 'BuPu' 'jet' 'PuRd' 'viridis'
    norm = colors.Normalize(vmin=0, vmax=round(preds_gdf[preds_gdf.columns[4:-1]].max(axis=1).max()+5.1, -1))

    # Iterate over each column and create a subplot for each. To begin on a later date change the starting column slice. 
    for i, column in tqdm(enumerate(preds_gdf.columns[4:-1])):
        # Calculate the subplot position in the layout
        row = i // num_cols
        col = i % num_cols

        # Select the column data
        column_data = preds_gdf[column]
        
        # Plot the column data in the corresponding subplot
        preds_gdf.plot(ax=axes[row, col],column=column_data, marker='s', markersize=1, cmap=cmap, norm=norm)
        gdf_shapefile.plot(ax=axes[row, col], facecolor='none', edgecolor='black', linewidth=1)
        
        # Set the subplot title to the column name
        axes[row, col].set_title(column)

        # Add x and y-axis labels for the top-left subplot
        axes[row, col].xaxis.set_tick_params(labelbottom=False)
        axes[row, col].yaxis.set_tick_params(labelleft=False)
        if row == 0 and col == 0:
            axes[row, col].xaxis.set_tick_params(labelbottom=True, labelrotation=45)
            axes[row, col].yaxis.set_tick_params(labelleft=True)

    #remove empty plot at end
    for ax in range(1,7):
        axes.flat[-ax].set_visible(False)

    # fig = axes.get_figure()
    cax = fig.add_axes([0.0015, 0.3, 0.007, 0.4])  # Adjust the position and size of the colorbar axis
    # Create a colorbar for the entire plot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, cax=cax)
    # cbar.set_label('SWE (in)')
    cbar.ax.set_title('SWE (in)', fontsize=10)

    # sm._A = []
    # fig.colorbar(sm, cax=cax)

    # Adjust the spacing between subplots
    plt.tight_layout()
    # plt.savefig(cwd+'//extrapolation//'+area+'//Predictions//20.0//Predictions//'+area+'_'+end_date+'_estimated_SWE.png', bbox_inches='tight')

    plt.show()

# @staticmethod
def animate_SWE(cwd, area, end_date):
    """
    Create a matplotlib animation of multiple date model inferences.
        Args: 
            cwd (str): current working diectory. This should be the "Model" directory.
            area (str): Name of the region to model.
            end_date (str): 'YYYY-MM-DD' - Date of final model infrence to plot.
    """

    #load SWE predictions up to the end_date
    preds=pd.read_csv(cwd+'//extrapolation//'+area+'//Predictions//submission_format_'+end_date+'.csv', index_col = [0])
    preds.index.names=['cell_id']

    #load feature dataframe & area shapefile
    Geo_df = pd.read_csv(cwd+'//extrapolation//'+area+'//'+area+'_Geo_df.csv', index_col='cell_id')
    shapefile_path = cwd+'//extrapolation//'+area+'//'+area+'.shp'
    gdf_shapefile = gpd.read_file(shapefile_path)

    # Make Geodataframe for plotting
    preds=Geo_df.join(preds)
    preds_gdf = gpd.GeoDataFrame(preds, geometry=gpd.points_from_xy(preds.Long, preds.Lat), crs="EPSG:4326")
    
    # Set up the figure and subplots
    fig, ax = plt.subplots(figsize=(10,10))

    # Get the column names with the dates. To begin on a later date change the starting column slice.
    date_columns = preds_gdf.columns[4:-1].tolist()

    # Define the update function for each frame
    def update(frame):
        ax.clear()
        # Get the column name for the current frame
        date_column = '{}'.format(frame)
        # Create masks for zero and non-zero values
        mask_zero = preds_gdf[date_column] == 0
        mask_non_zero = preds_gdf[date_column] != 0

        # Plot non-zero values
        preds_gdf[mask_non_zero].plot(ax=ax, column=date_column, marker='s', markersize=260, cmap=cmap_ani, norm=norm, alpha=1)
        
        gdf_shapefile.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
        # current_data.plot(ax=ax)
        cx.add_basemap(ax, crs=preds_gdf.crs.to_string(), source=cx.providers.Esri.WorldTerrain, alpha=0.5)
        ax.set_title('{}'.format(date_column))
        ax.spines[['right', 'top']].set_visible(False)


    # Create a normalization instance for color mapping
    cmap_ani = plt.cm.get_cmap('PuRd')  # Choose a colormap #'Blues' 'BuPu' 'jet' 'PuRd' 'viridis'
    norm = colors.Normalize(vmin=0, vmax=round(preds_gdf[preds_gdf.columns[4:-1]].max(axis=1).max()+5.1, -1))
    cax = fig.add_axes([0.935, 0.3, 0.01, 0.4])  # Adjust the position and size of the colorbar axis
    # Create a colorbar for the entire plot
    sm = plt.cm.ScalarMappable(cmap=cmap_ani, norm=norm)
    cbar = plt.colorbar(sm, cax=cax)
    # cbar.set_label('SWE (in)')
    cbar.ax.set_title('Est. SWE (in)', fontsize=10)

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=date_columns, interval=1000)
    # plt.tight_layout()

#     # save animation - may need to conda install ffmpeg, pip install might not work
#     f = cwd+'//extrapolation//'+area+'//Predictions//20.0//Predictions//'+end_date+'_SWE.mp4'
#     print('Animation will be saved to', f)
#     # writervideo = animation.FFMpegWriter(fps=10)
#     writervideo = animation.FFMpegWriter(fps=10, metadata=dict(artist='D. Liljestrand'), bitrate=1800, extra_args=['-vcodec', 'libx264'])

#     # ani.save(f, writer=writervideo)
#     try:
#         ani.save(f, writer=writervideo)
#         print('Animation saved to', f)
#     except Exception as e:
#         print("Error saving animation:", e)

    # Show the animation
    # fig.show()
    plt.close()
    out = HTML(ani.to_jshtml())


    return out


def animate_SWE2(cwd, area, end_date):
    """
    Create a matplotlib animation of multiple date model inferences.
        Args: 
            cwd (str): current working directory. This should be the "Model" directory.
            area (str): Name of the region to model.
            end_date (str): 'YYYY-MM-DD' - Date of final model inference to plot.
    """

    # Load SWE predictions up to the end_date
    preds = pd.read_csv(cwd + '//extrapolation//' + area + '//Predictions//submission_format_' + end_date + '.csv', index_col=[0])
    preds.index.names = ['cell_id']

    # Load feature dataframe & area shapefile
    Geo_df = pd.read_csv(cwd + '//extrapolation//' + area + '//' + area + '_Geo_df.csv', index_col='cell_id')
    shapefile_path = cwd + '//extrapolation//' + area + '//' + area + '.shp'
    gdf_shapefile = gpd.read_file(shapefile_path)

    # Make Geodataframe for plotting
    preds = Geo_df.join(preds)
    preds_gdf = gpd.GeoDataFrame(preds, geometry=gpd.points_from_xy(preds.Long, preds.Lat), crs="EPSG:4326")
    
    # Set up the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Get the column names with the dates. To begin on a later date change the starting column slice.
    date_columns = preds_gdf.columns[4:-1].tolist()
    
    # Initialize a list to store maximum predictions over time
    max_preds = []
    
    # Load Mammoth Pass observed SWE values
    MHP_SWE = pd.read_csv(cwd + '//extrapolation//Data//Mammoth_Pass_SWE.csv', index_col=[0])
    
    # Create a normalization instance for color mapping
    cmap_ani = plt.cm.get_cmap('PuRd')  # Choose a colormap
    norm = colors.Normalize(vmin=0, vmax=round(preds_gdf[preds_gdf.columns[4:-1]].max(axis=1).max() + 5.1, -1))
    cax = fig.add_axes([0.935, 0.3, 0.01, 0.4])  # Adjust the position and size of the colorbar axis
    # Create a colorbar for the entire plot
    sm = plt.cm.ScalarMappable(cmap=cmap_ani, norm=norm)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.ax.set_title('Est. SWE (in)', fontsize=10)

    # Define the update function for each frame
    def update(frame):
        ax1.clear()
        ax2.clear()
 
        # Get the column name for the current frame
        # date_column = frame
        date_column = date_columns[frame]
        
        # Create masks for zero and non-zero values
        mask_zero = preds_gdf[date_column] == 0
        mask_non_zero = preds_gdf[date_column] != 0

        # Plot non-zero values
        preds_gdf[mask_non_zero].plot(ax=ax1, column=date_columns[frame], marker='s', markersize=260, cmap=cmap_ani, norm=norm, alpha=1)
        gdf_shapefile.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=1)
        cx.add_basemap(ax1, crs=preds_gdf.crs.to_string(), source=cx.providers.Esri.WorldTerrain, alpha=0.5)
        ax1.set_title(f'{date_column}')
        ax1.spines[['right', 'top']].set_visible(False)
        
        # Append the max prediction value for the current date
        max_pred = preds_gdf[date_column].max()
        max_preds.append(max_pred)

        # Plot the max prediction time series
        ax2.plot(date_columns[:len(max_preds)], max_preds, marker='o', color='r')
        ax2.plot(MHP_SWE.index[:len(max_preds)], MHP_SWE.MHP_SWE_in[:len(max_preds)], marker='o', color='b')
        ax2.set_title(f'Up to {date_column}') 
        ax2.legend(['Model', 'MHP'], loc='lower right', frameon=False)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Maximum SWE (in)')
        ax2.grid(True)
        
    def init():
        print('Initializing WY 2022')
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(date_columns), interval=1000, init_func=init, blit=False)
  
    plt.close()
    return HTML(ani.to_jshtml())