# Predictive Model for Temperature, Humidity, and Rainfall Forecasting

This repository contains the code, data artifacts, notebooks, figures, and selected outputs for **next-day weather forecasting**. The project uses historical **ERA5-Land** weather data to predict **temperature**, **relative humidity**, and **precipitation**, with Hong Kong as the target test city and Bangkok, Ho Chi Minh City, and Kuala Lumpur used as source cities in selected experiments.

## Project Scope

The project studies whether different modelling strategies are effective for forecasting key weather variables under a cross-city setting. The workflow covers:

- ERA5-Land data download and preprocessing
- Daily aggregation and regional feature construction
- Exploratory data analysis and data cleaning
- Model training and evaluation
- Comparison of forecasting performance across targets and model families
- Streamlit deployment for interactive Hong Kong next-day weather forecast visualisation

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
├─ deployment/
│  ├─ hk_weather_app_streamlit.py
│  ├─ hk_best_model_config.py
│  ├─ make_hk_app_predictions.py
│  ├─ make_hk_land_assets.py
│  └─ requirements.txt
│
├─ data/
│  ├─ interim/
│  ├─ processed/
│  └─ regional_features_csv/
│
├─ outputs/
│  └─ app/
│  └─ modelling_outputs/
│
├─ figures/
│  ├─ eda/
│  └─ modelling/
```

## How to Run

### 1. Clone the repository

```shell
git clone https://github.com/acrylicruler/Predictive-Model-for-Temperature-Humidity-and-Rainfall-Forecasting.git
cd Predictive-Model-for-Temperature-Humidity-and-Rainfall-Forecasting
git lfs pull
```

### 2. Create a virtual environment

```shell
python -m venv .venv
```

### 3. Activate the virtual environment on Windows

```shell
.venv\Scripts\activate
```

### 4. Install dependencies

```shell
pip install -r requirements.txt
```

### 5. Run the data pipeline

Run these scripts in order if you want to reproduce the workflow from raw ERA5-Land data:

```shell
python src/data_pipeline/01_download_era5_land_hourly.py
python src/data_pipeline/02_build_daily_from_nc.py
python src/data_pipeline/03_build_regions_and_aggregate.py
python src/data_pipeline/04_make_features_and_targets.py
```

### 6. Run the notebooks

```shell
jupyter notebook
```


### 7. Run the Streamlit App Locally

The Streamlit app provides an interactive deployment interface for visualising Hong Kong next-day weather forecasts.

Install the app-specific dependencies:

```shell
pip install -r deployment/requirements.txt
```

Run the app:
```shell
streamlit run deployment/hk_weather_app_streamlit.py
```

Then open and run these notebooks in order:

1. `notebooks/FYP_DataCleaning_EDA.ipynb`
2. `notebooks/FYP_Modelling.ipynb`

## Recommended Execution Order

```text
1. Clone repository
2. Pull Git LFS files
3. Create and activate virtual environment
4. Install dependencies
5. Run 01_download_era5_land_hourly.py
6. Run 02_build_daily_from_nc.py
7. Run 03_build_regions_and_aggregate.py
8. Run 04_make_features_and_targets.py
9. Run FYP_DataCleaning_EDA.ipynb
10. Run FYP_Modelling.ipynb
```

## Main Components

### `src/data_pipeline/`

- `01_download_era5_land_hourly.py` downloads hourly ERA5-Land data.
- `02_build_daily_from_nc.py` converts hourly data into daily point-level parquet files.
- `03_build_regions_and_aggregate.py` builds regional grids and aggregated regional daily datasets.
- `04_make_features_and_targets.py` creates engineered features and forecasting targets.

### `notebooks/`

- `FYP_DataCleaning_EDA.ipynb` performs data checks, cleaning, exploratory data analysis, and preparation of model-ready datasets.
- `FYP_Modelling.ipynb` performs model training, evaluation, comparison, and result generation.

### `data/`

Contains intermediate and processed data artifacts used by the notebooks and modelling workflow, including parquet files, geojson grids, and CSV datasets.

### `outputs/`

Contains selected modelling and deployment outputs.

- `outputs/modelling_outputs/` stores modelling results such as selected models, annotated sample days, feature importance summaries, and test comparison tables.
- `outputs/app/` stores the pre-generated Hong Kong prediction files used by the Streamlit app.

### `figures/`

Contains exported figures used for EDA and modelling analysis.

### `deployment/`

Contains the Streamlit deployment files for the Hong Kong weather forecast application.

- `hk_weather_app_streamlit.py` is the main Streamlit app file.
- `hk_best_model_config.py` stores the selected best model configuration used for app prediction generation.
- `make_hk_app_predictions.py` generates the prediction files used by the app.
- `make_hk_land_assets.py` prepares the Hong Kong geographic assets used for map-based visualisation.
- `requirements.txt` lists the app-specific Python dependencies.

## Notes

- Large files such as `.parquet`, `.ipynb`, and selected `.csv` files may be tracked using **Git LFS**. Run `git lfs pull` after cloning to retrieve them fully.
- If the required data artifacts are already included in the repository, the notebooks can be run directly without regenerating the full pipeline from scratch.
- If rerunning the raw download step, ensure ERA5/CDS API access is configured locally.
- Some scripts depend on outputs generated by earlier steps, so the execution order should be followed strictly.

## Author

Final Year Project repository for weather forecasting using ERA5-Land data.
