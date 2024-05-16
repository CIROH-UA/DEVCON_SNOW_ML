# Snow-Extrapolation - ML Algorithm Optimization for Operational Snow-Water-Equivalent Estimation

This workbook builds on the ongoing Cooperative Institute for Research-to-Operations in Hydrology ([CIROH](https://ciroh.ua.edu/)) machine learning snow modeling project to provide operational, near-real-time snow-water-equivalent (SWE) estimates for updating the state of the snowpack within the National Water Model ([NWM](https://water.noaa.gov/about/nwm)).

The [Getting started](./Getting%20Started.md) file will help new users create a virtual environment and install the correct packages using the [requirements.text](./requirements.txt).

## National-Snow-Model Summary

Snow-derived water is a critical hydrological component for characterizing the quantity of water available for domestic, recreation, agriculture, and power generation in the western United States.
Advancing the efficiency and optimization of these aspects of water resources management requires an enhanced characterization of the snow state variable, particularly the essential global inputs of snow-water-equivalent (SWE), peak SWE, and snowmelt onset for hydrological models.
While physically-based models that characterize the feedbacks and interactions between influencing factors predict SWE values well in homogeneous settings, these models exhibit limitations for CONUS-scale deployment due to challenges attributed to spatial resolution, landscape heterogeneity, and computational intensity. 
Leveraging a collaborative partnership between the Alabama Water Institute (AWI) at the University of Alabama (UA) and the University of Utah (UU), we address these limitations through the NSM as a data-driven machine learning (ML) platform with a modular structure to account for the heterogeneity of climate and topographical influences on SWE across the western United States.
The model consists of twenty-three regionally specific sub-models tailored to the unique topography and hydroclimate phenomena in the Western U.S., exhibiting an RMSE less than 8 cm and a coefficient of determination approaching 0.99 on predictions spanning the 2013-2017 training period.
The NSM pipeline assimilates nearly 700 snow telemetry (SNOTEL) and California Data Exchange Center (CDEC) sites and combines with processed lidar-derived terrain features for the prediction of a 1 km x 1 km SWE inference in critical snowsheds.

## Workbook Summary: Description and Goals

The purpose of this workbook is to provide users with an introduction to snow modeling through leveraging machine learning and observation data. After running the example notebook, users are encouraged to explore and create their own modifications to the code to refine SWE prediction performance in the Sierra Nevada mountains.
The current Sierra Nevada core ML algorithm is a deep neural network, and while it demonstrates satisfactory prediction skill, there still are large errors in predction and room for improvement identified by a rigorous evaluation of the model.
The most notable error appear at locations below 1,500m and above 2,900, ephemeral and alpine regions, respectively, and during the snow melt period.
We encourage users to explore other ML algorithms, such as other neural networks, Long-Short-Term-Memory, tree-based, and even simple regression algorithms such as Ordinary Least Squares.
Model evaluation will form a critical element of determing differences in algorithm performance, and the Standardized Snow Water Equivalent Evaluation Tool ([SSWEET](https://github.com/whitelightning450/Standardized-Snow-Water-Equivalent-Evaluation-Tool)) will serve as a standardized method to evaluate the different algorithm architectures.

## Creating Stable CONDA Environment

* Go to home directory
```
cd ~
```
* Create a envs directory
```
mkdir envs
```
* Create .condarc file and link it to a text file
```
touch .condarc

ln -s .condarc condarc.txt
```
* Add the below lines to the condarc.txt file
```
# .condarc
envs_dirs:
 - ~/envs
```
* Restart your server


## Files

* `.gitignore`
<br> Globally ignored files by `git` for the project.
* `environment.yml`
<br> `conda` environment description needed to run this project.
* `README.md`

### `scripts` and `requirements.txt`
All helper utilities and updated requirements.txt to replicate the working environemnt should be included in the project submissin package.


### Data

All data will be provided to participants with cloud access to either the CIROH publicly accessible Amazon Web Services (AWS) S3 storage or via Box.
Project data includes:
* NASA Airborne Snow Observator (ASO) LiDAR-derived SWE estimates
* Copernicus 90m Digitial Elevation Model (DEM)
* Natural Resources Conservation Service (NRCS) Snow Telemetry (SNOTEL) monitoring station SWE observations
* Visible Infrared Imaging Radiometer Suite (VIIRS) fraction snow covered area (fSCA)

The data folder contains pre-processed model training dataframes of approximately 7,000 1-km grid locations to limit the amount of time spent on data processing tasks.
While a key and essential component of any ML objective, data pre-processing and feature engineering are not a focus of this workbook.

### Existing methods

The current NSM uses the same 7-layer deep neural network for each of the 23 regions.
While regionally-optimized Light Gradient Boosted Models (LGBM) and Extreme Gradient Boosted (XGBoost) algorithms were part of the original model development phase, the standard 7-layer multilayered perceptron (MLP) neural network demonstrated higher overall prediction skill and efficiency than the tree-based models, notably during prolong snow-drought.
Existing model research and development explores and compares different algorithms, with the best-performing algorithm transitioning to become optimized.


### Tasks

The following tasks are expected to be completed before the Hack Week and will be made available to participants a week before the event:
* Create your own Github account if you do not have one
* Fork this repository to your personal account
* Download [Git Desktop](https://desktop.github.com/) and clone the repository to your machine. Note, if you are comfortable with the command line process using git, this is an acceptable approach
* Run through the provided steps in the Getting Started page to set up your appropriate virtual environment
* Successfully run the Sierra Nevada region(s) of the NSM
* Review and develop and understanding the model training and development steps
* Review ML literature and come prepared with algorithms/approaches to enter the Hack Week with explicit algorithm and coding goals.



