# Predictive Model for Temperature, Humidity, and Rainfall Forecasting

This repository contains the code, data artifacts, notebooks, figures, and selected outputs for an FYP on **next-day weather forecasting**. The project uses historical **ERA5-Land** weather data to predict **temperature**, **relative humidity**, and **precipitation**, with Hong Kong as the target test city and Bangkok, Ho Chi Minh City, and Kuala Lumpur used as source cities in selected experiments.

## Project Scope

The project studies whether different modelling strategies are effective for forecasting key weather variables under a cross-city setting. The workflow covers:

- ERA5-Land data download and preprocessing
- Daily aggregation and regional feature construction
- Exploratory data analysis and data cleaning
- Model training and evaluation
- Comparison of forecasting performance across targets and model families

## Repository Structure

```text
.
├─ src/
│  └─ data_pipeline/
│     ├─ 01_download_era5_land_hourly.py
│     ├─ 02_build_daily_from_nc.py
│     ├─ 03_build_regions_and_aggregate.py
│     └─ 04_make_features_and_targets.py
│
├─ notebooks/
│  ├─ FYP_DataCleaning_EDA.ipynb
│  └─ FYP_Modelling.ipynb
│
├─ data/
│  ├─ interim/
│  ├─ processed/
│  └─ regional_features_csv/
│
├─ outputs/
│  └─ modelling_outputs/
│
├─ figures/
│  ├─ eda/
│  └─ modelling/
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/acrylicruler/Predictive-Model-for-Temperature-Humidity-and-Rainfall-Forecasting.git
cd Predictive-Model-for-Temperature-Humidity-and-Rainfall-Forecasting
git lfs pull
