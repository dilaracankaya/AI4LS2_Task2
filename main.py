import pandas as pd
import numpy as np
import xarray as xr
import pickle
import requests
import os
import re
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
import shap
import plotly.express as px
import plotly.graph_objs as go
from scipy.spatial.distance import cdist
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 500)

warnings.filterwarnings("ignore")

########################################################################################################################
# GRACE
########################################################################################################################
# Opening .nc data files and converting the variables from these files into dataframes
df_land = xr.open_dataset('supplemental_material_for_task_2/datasets/CSR_GRACE_GRACE-FO_RL06_Mascons_v02_LandMask.nc')
df_land = df_land['LO_val'].to_dataframe().reset_index()

df_lwe = xr.open_dataset('supplemental_material_for_task_2/datasets/CSR_GRACE_GRACE-FO_RL0602_Mascons_all-corrections.nc')
df_lwe = df_lwe["lwe_thickness"].to_dataframe().reset_index()

# Reducing the df_lwe dataframe based on the land mask
df_land_expanded = pd.concat([df_land['LO_val']] * 232, ignore_index=True)

df = pd.concat([df_lwe, df_land_expanded], axis=1)
df = df[df['LO_val'] == 1]
df.drop("LO_val", axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)

# Transforming "time" column to date format
missing_months = ('2002-06;2002-07;2003-06;2011-01;2011-06;2012-05;2012-10;2013-03;2013-08;2013-09;2014-02;2014-07;'
                  '2014-12;2015-06;2015-10;2015-11;2016-04;2016-09;2016-10;2017-02;2017-02;2017-07;2017-08;2017-09;'
                  '2017-10;2017-11;2017-12;2018-01;2018-02;2018-03;2018-04;2018-05;2018-08;2018-09')
missing_months_list = missing_months.split(';')

start_date = pd.Period('2002-04', freq='M')
all_months = pd.period_range(start=start_date, end='2024-04', freq='M')

filtered_months = [str(month) for month in all_months if str(month) not in missing_months_list]

df['time'] = pd.Series(filtered_months).repeat(df.shape[0] // len(filtered_months)).reset_index(drop=True)[:df.shape[0]]

# GRACE after 2010.01 in time column
df = df[df['time'] >= '2010-01']
df['lon'] = df['lon'].apply(lambda x: x - 360 if x > 180 else x)
df.reset_index(drop=True, inplace=True)

# Checking whether coordinates are same for each month
reference_month_grace = df[df['time'] == '2010-01'][['lat', 'lon']].reset_index(drop=True)

comparison = {}
for month in df['time'].unique():
    if month != '2010-01':
        comparison[month] = df[df['time'] == month][['lat', 'lon']].reset_index(drop=True)

results = {month: (reference_month_grace.equals(data) for month, data in comparison.items())}

for month, is_equal in results.items():
    print(f"Comparison result for {month}: {'Same' if is_equal else 'Different'}")

########################################################################################################################
# GLDAS
########################################################################################################################
# Please see GLDAS website for how to authenticate your device for data fetching.
# This function will not work unless the authentication steps are complete.

def load_gldas_dict_2004_2009():
    # 1. Fetch data between 2004-2009 to match GRACE's averaged format
    file_path = "supplemental_material_for_task_2/datasets/2004_2009_avg_gldas_noah_2209.txt"
    ds_dict = {}
    with open(file_path, "r") as f:
        # Skip the first line
        next(f)

        for line in f:
            url = line.strip()
            match = re.search(r'(\d{6})\.\d+\.nc4', url)
            if match:
                date = match.group(1)
            else:
                print(f'No date found in URL: {url}')
                continue

            print(f'Downloading {url}...')
            try:
                response = requests.get(url)
                response.raise_for_status()  # Check for request errors

                temp_file_name = f"temp_{date}.nc4"
                with open(temp_file_name, 'wb') as temp_file:
                    temp_file.write(response.content)

                ds_dict[date] = xr.open_dataset(temp_file_name)
                os.remove(temp_file_name)
                print(f'Dataset for {date} added to dictionary.')

            except requests.exceptions.RequestException as e:
                print(f'Error downloading {url}: {e}')
            except Exception as e:
                print(f'Error processing file {temp_file_name}: {e}')

    print('All files processed!')

    for key, value in ds_dict.items():
        ds_dict[key] = value.load()

    ds = ds_dict["200401"]
    variables = list(ds.data_vars)[1:]
    monthly_gldas_2004_2009 = {}

    for key, value in ds_dict.items():
        df = pd.DataFrame()

        for feature in variables:
            data_array = value[feature]
            feature_df = data_array.to_dataframe().reset_index().drop("time", axis=1)
            feature_df = feature_df.dropna(subset=[f"{feature}"])

            if df.empty:
                df = feature_df
            else:
                df = df.merge(feature_df, on=["lat", "lon"])

        monthly_gldas_2004_2009[key] = df.reset_index(drop=True)
    return monthly_gldas_2004_2009


def load_gldas_dict_2010_2024():
    # 2. Fetch data between 2010-2024 for data analysis
    file_path = "supplemental_material_for_task_2/datasets/subset_GLDAS_NOAH025_M_2.1_20240918_193208_.txt"
    ds_dict = {}
    with open(file_path, "r") as f:
        # Skip the first line
        next(f)

        for line in f:
            url = line.strip()
            match = re.search(r'(\d{6})\.\d+\.nc4', url)
            if match:
                date = match.group(1)
            else:
                print(f'No date found in URL: {url}')
                continue

            print(f'Downloading {url}...')
            try:
                response = requests.get(url)
                response.raise_for_status()  # Check for request errors

                temp_file_name = f"temp_{date}.nc4"
                with open(temp_file_name, 'wb') as temp_file:
                    temp_file.write(response.content)

                ds_dict[date] = xr.open_dataset(temp_file_name)
                os.remove(temp_file_name)
                print(f'Dataset for {date} added to dictionary.')

            except requests.exceptions.RequestException as e:
                print(f'Error downloading {url}: {e}')
            except Exception as e:
                print(f'Error processing file {temp_file_name}: {e}')

    print('All files processed!')

    for key, value in ds_dict.items():
        ds_dict[key] = value.load()

    ds = ds_dict["201210"]
    variables = list(ds.data_vars)[1:]
    gldas_dict_2010_2024 = {}

    for key, value in ds_dict.items():
        df = pd.DataFrame()

        for feature in variables:
            data_array = value[feature]
            feature_df = data_array.to_dataframe().reset_index().drop("time", axis=1)
            feature_df = feature_df.dropna(subset=[f"{feature}"])

            if df.empty:
                df = feature_df
            else:
                df = df.merge(feature_df, on=["lat", "lon"])

        gldas_dict_2010_2024[key] = df.reset_index(drop=True)

    return gldas_dict_2010_2024


gldas_dict_2004_2009 = load_gldas_dict_2004_2009()
gldas_dict_2010_2024 = load_gldas_dict_2010_2024()

# Checking whether coordinates are same for each month
coordinates_per_df_gldas = [set(zip(df['lat'], df['lon'])) for df in gldas_dict_2010_2024.values()]
all_same = all(coords == coordinates_per_df_gldas[0] for coords in coordinates_per_df_gldas)
if all_same:
    print("Yes")
else:
    print("No")


########################################################################################################################
# Further Feature Engineering
########################################################################################################################
# Intersection of latitude and longitude couples that come from Gldas and GRACE datasets.
a_month_set = set(zip(reference_month_grace['lat'], reference_month_grace['lon']))
coordinates_set = set(coordinates_per_df_gldas[0])
intersection_set = a_month_set.intersection(coordinates_set)

# Editing the coordinates in GLDAS according to the intersection set.
filtered_dfs = {}

for key, dataframe in gldas_dict_2010_2024.items():
    dataframe['coord_tuple'] = list(zip(dataframe['lat'], dataframe['lon']))
    filtered_df = dataframe[dataframe['coord_tuple'].apply(lambda x: x in intersection_set)]
    filtered_dfs[key] = filtered_df.drop(columns=['coord_tuple'])

for key, dataframe in filtered_dfs.items():
    dataframe.reset_index(drop=True, inplace=True)

# Editing the coordinates in GRACE according to the intersection set
df = df.groupby('time', group_keys=False).apply(lambda group: group[group[['lat', 'lon']].apply(tuple, axis=1).isin(intersection_set)])
df.reset_index(drop=True, inplace=True)

# with open("supplemental_material_for_task_2/pkl_files/df.pkl", "wb") as f:
#     pickle.dump(df, f)

# Filling in the missing months in the dataframe and assigning nan values to the lew_thickness columns.
df['time'] = pd.to_datetime(df['time'])
reference_lat_lon = df[df['time'] == '2010-01-01'][['lat', 'lon']].drop_duplicates()

existing_months = df['time'].drop_duplicates()
all_months = pd.date_range(start='2010-01-01', end='2024-04-01', freq='MS')
missing_months = all_months.difference(existing_months)

missing_data = pd.concat(
    [pd.DataFrame({
        'time': [month] * len(reference_lat_lon),
        'lat': reference_lat_lon['lat'].values,
        'lon': reference_lat_lon['lon'].values,
        'lwe_thickness': np.nan})
     for month in missing_months]
)

df_filled_corrected = pd.concat([df, missing_data]).drop_duplicates(subset=['time', 'lat', 'lon']).sort_values(by=['time', 'lat', 'lon']).reset_index(drop=True)



# GRACE: dataframe to dictionary
df_filled_corrected['key'] = df_filled_corrected['time'].dt.strftime('%Y%m')
result_dict = {key: group.drop(columns='key') for key, group in df_filled_corrected.groupby('key')}

for key, value in result_dict.items():
    value.reset_index(inplace=True, drop=True)


# Imputing NaN Values in the dictionary.
for month_key, month_df in result_dict.items():
    current_month = month_key[-2:]
    measurement_index = month_df.index

    for i in measurement_index:
        if pd.isna(month_df.at[i, 'lwe_thickness']):
            other_year_values = []
            for year in range(2010, 2025):
                year_key = f"{year}{current_month}"
                if year_key in result_dict:
                    other_year_df = result_dict[year_key]
                    if i < len(other_year_df):
                        value = other_year_df.at[i, 'lwe_thickness']
                        if pd.notna(value):
                            other_year_values.append(value)

            if other_year_values:
                average_value = np.mean(other_year_values)
                month_df.at[i, 'lwe_thickness'] = average_value


# with open('supplemental_material_for_task_2/pkl_files/grace_imputed_in_dict.pkl', 'wb') as f:
#     pickle.dump(result_dict, f)

# Merging Gldas and GRACE datasets

gldas_dict_2010_2024.pop('202405', None)

########################################################################################################################
# MERGING GRACE AND GLDAS
########################################################################################################################
for key in gldas_dict_2010_2024.keys():
    gldas_df = gldas_dict_2010_2024[key]
    grace_df = result_dict[key]

    if grace_df is not None:
        merged_df = gldas_df.merge(grace_df[['lat', 'lon', 'lwe_thickness']], on=['lat', 'lon'], how='inner')
        gldas_dict_2010_2024[key] = merged_df


with open('supplemental_material_for_task_2/pkl_files/intersection_set.pkl', 'rb') as file:
    intersection_set = pickle.load(file)

filtered_dict = {}
for key, df in gldas_dict_2004_2009.items():
    filtered_df = df[df[['lat', 'lon']].apply(tuple, axis=1).isin(intersection_set)]
    filtered_dict[key] = filtered_df

gldas_dict_2004_2009 = filtered_dict.copy()


# Selecting coordinates in every 209 given longitude
def reduce_to_first_of_209(df):
    return df.iloc[::209, :]


def convert_cols(df, input_col):
    col_type = input_col.split("_")[-1]

    if col_type == "tavg":
        df[f"new_{input_col}"] = df[input_col] * 10800 * 8 * 30
        df.drop(input_col, axis=1, inplace=True)

    elif col_type == "acc":
        df[f"new_{input_col}"] = df[input_col] * 8 * 30
        df.drop(input_col, axis=1, inplace=True)


def process_data(dict):
    results_dict = {}

    for key, df in dict.items():
        results_dict[key] = reduce_to_first_of_209(df)
        results_dict[key].reset_index(drop=True, inplace=True)

    for month, df in results_dict.items():
        for col in df.columns:
            if "_tavg" in col or "_acc" in col:
                convert_cols(df, col)

        try:
            df['MSW'] = (df['new_Rainf_f_tavg'] + df['new_Qsb_acc']) - (
                        df['new_Evap_tavg'] - df['new_ESoil_tavg'] + df['new_Qs_acc'])
        except KeyError as e:
            print(f"KeyError: {e}. Bu sütun eksik olabilir.")

        df.rename(columns={'SWE_inst': 'MSN'}, inplace=True)

        # 'lwe_thickness' sütunu varsa deltaTWS hesaplan?yor
        if 'lwe_thickness' in df.columns:
            df['deltaTWS'] = df["lwe_thickness"] * 10

        df['MSM'] = (df["SoilMoi0_10cm_inst"] + df["SoilMoi10_40cm_inst"] + df["SoilMoi40_100cm_inst"] +
                     df["SoilMoi100_200cm_inst"])

        df['SoilTMP0_avg'] = (df['SoilTMP0_10cm_inst'] + df['SoilTMP10_40cm_inst'] + df['SoilTMP40_100cm_inst'] +
                              df['SoilTMP100_200cm_inst'])

        # Silinecek kolonlar listesi
        cols_to_drop = ['SoilMoi0_10cm_inst', 'SoilMoi10_40cm_inst', 'SoilMoi40_100cm_inst', 'SoilMoi100_200cm_inst',
                        'SoilTMP0_10cm_inst', 'SoilTMP10_40cm_inst', 'SoilTMP40_100cm_inst', 'SoilTMP100_200cm_inst']

        # 'lwe_thickness' varsa, silinecek kolonlar listesine ekleniyor
        if 'lwe_thickness' in df.columns:
            cols_to_drop.append('lwe_thickness')

        df.drop([col for col in cols_to_drop if col in df.columns], axis=1, inplace=True)

        results_dict[month] = df

    return results_dict


results_dict_2010_2024 = process_data(gldas_dict_2010_2024)
results_dict_2004_2009 = process_data(gldas_dict_2004_2009)


# Calculating the averages of 'MSW', 'MSM', 'MSN in results_dict_2004_2009.
first_df = next(iter(results_dict_2004_2009.values()))

lat_lon_pairs = first_df[['lat', 'lon']].copy()


for key, df in results_dict_2004_2009.items():
    if not df[['lat', 'lon']].equals(lat_lon_pairs):
        print(f"{key} lat and lon columns are different.")
        break
else:
    print("All DataFrames have the same lat and lon columns.")


first_df = next(iter(results_dict_2004_2009.values()))
mean_df = first_df[['lat', 'lon']].copy()

mean_df[['MSN_mean', 'MSW_mean', 'MSM_mean']] = 0.0


for index in range(len(first_df)):
    mean_df.loc[index, ['MSN_mean', 'MSW_mean', 'MSM_mean']] = [
        sum(df[col].iloc[index] for df in results_dict_2004_2009.values()) / len(results_dict_2004_2009)
        for col in ['MSN', 'MSW', 'MSM']]


first_rows = []

for key in sorted(results_dict_2004_2009.keys()):
    df = results_dict_2004_2009[key]
    first_rows.append(df.iloc[0])

first_rows_df = pd.DataFrame(first_rows)

averages = first_rows_df[['MSN', 'MSW', 'MSM']].mean()


for key, df in results_dict_2010_2024.items():
    df_merged = df.merge(mean_df[['lat', 'lon', 'MSN_mean', 'MSW_mean', 'MSM_mean']], on=['lat', 'lon'])


    df['delta_MSN'] = df['MSN'] - df_merged['MSN_mean']
    df['delta_MSW'] = df['MSW'] - df_merged['MSW_mean']
    df['delta_MSM'] = df['MSM'] - df_merged['MSM_mean']

    results_dict_2010_2024[key] = df


#        deltaMGw = deltaTWS - deltaMSM - deltaMSN - deltaMSw
for key, df in results_dict_2010_2024.items():
    df['delta_MGW'] = df['deltaTWS'] - df['delta_MSM'] - df['delta_MSN']  - df['delta_MSW']


# with open('supplemental_material_for_task_2/pkl_files/new_12661_results_dict_2004_2009.pkl', 'wb') as file:
#     pickle.dump(results_dict_2004_2009, file)
#
# with open('supplemental_material_for_task_2/pkl_files/new_12661_results_dict_2010_2024.pkl', 'wb') as file:
#     pickle.dump(results_dict_2010_2024, file)


# Train and test split
train_dict = {}
test_dict = {}

for key, df in results_dict_2010_2024.items():
    date_key = pd.to_datetime(str(key), format='%Y%m')
    year = date_key.year

    if 2010 <= year <= 2018:
        train_dict[key] = df
    elif 2019 <= year <= 2024:
        test_dict[key] = df


# with open('supplemental_material_for_task_2/pkl_files/new_1151_train_dict.pkl', 'wb') as file:
#     pickle.dump(train_dict, file)
#
# with open('supplemental_material_for_task_2/pkl_files/new_1151_test_dict.pkl', 'wb') as file:
#     pickle.dump(test_dict, file)
#
# with open('supplemental_material_for_task_2/pkl_files/new_12661_train_dict.pkl', 'rb') as file:
#     train_dict = pickle.load(file)
#
# with open('supplemental_material_for_task_2/pkl_files/new_12661_test_dict.pkl', 'rb') as file:
#     test_dict = pickle.load(file)


# Troubleshooting:
# Whether the 'lat' and 'lon' columns of DataFrames in the dictionary are different or the same,
# and prints which DataFrames have different latitude values or the same longitude values.
lat_values = [df['lat'].values for df in test_dict.values()]
lon_values = [df['lon'].values for df in test_dict.values()]

for i in range(len(lat_values)):
    for j in range(i + 1, len(lat_values)):
        lat_equal = (lat_values[i] == lat_values[j]).all()
        lon_equal = (lon_values[i] == lon_values[j]).all()

        if not lat_equal:
            print(f"df{i + 1} ve df{j + 1}: 'lat' columns are different.")

        if not lon_equal:
            print(f"df{i + 1} ve df{j + 1}: 'lon' columns are the same.")


lat_values = [df['lat'].values for df in train_dict.values()]
lon_values = [df['lon'].values for df in train_dict.values()]

for i in range(len(lat_values)):
    for j in range(i + 1, len(lat_values)):
        lat_equal = (lat_values[i] == lat_values[j]).all()
        lon_equal = (lon_values[i] == lon_values[j]).all()

        if not lat_equal:
            print(f"df{i + 1} ve df{j + 1}: 'lat' columns are different.")

        if not lon_equal:
            print(f"df{i + 1} ve df{j + 1}: 'lon' columns are the same.")


# Multicollinearity test: Variance inflation factor (VIF)
with open('supplemental_material_for_task_2/pkl_files/new_1151_results_dict_2010_2024.pkl', 'rb') as file:
    data_dict = pickle.load(file)

new_dict = {}

for month, df in tqdm(data_dict.items(), desc="Processing months"):
    for index, row in df.iterrows():
        lat_lon_key = (row['lat'], row['lon'])
        if lat_lon_key not in new_dict:
            new_dict[lat_lon_key] = pd.DataFrame(columns=['date'] + df.columns.tolist()[2:])

        new_row = row.drop(['lat', 'lon']).to_dict()
        new_row['date'] = month

        new_df = pd.DataFrame([new_row])
        new_dict[lat_lon_key] = pd.concat([new_dict[lat_lon_key], new_df], ignore_index=True)


def calculate_vif(df):
    X = df.select_dtypes(include='number')
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

vif_results = {}
for key, df in tqdm(new_dict.items(), desc="Calculating VIF"):
    vif_results[key] = calculate_vif(df)

for key, result in vif_results.items():
    print(f"VIF Results for {key}:")
    print(result)


########################################################################################################################
# MODEL
########################################################################################################################

train_dfs = [df for key, df in data_dict.items() if 201001 <= int(key) <= 201812]
train_data = pd.concat(train_dfs)

test_dfs = [df for key, df in data_dict.items() if 201901 <= int(key) <= 202404]
test_data = pd.concat(test_dfs)

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

coordinates = train_data[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
selected_coordinates = coordinates.iloc[np.linspace(0, len(coordinates) - 1, 50, dtype=int)]

rf_params = {'bootstrap': True, 'max_depth': 20, 'max_features': None,
             'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 400}
model_rf = RandomForestRegressor(**rf_params)

ridge_params = {'alpha': 0.1, 'fit_intercept': False, 'solver': 'svd'}
model_ridge = Ridge(**ridge_params)

gb_params = {'n_estimators': 400, 'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 2}
model_gb = GradientBoostingRegressor(**gb_params)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

best_models_per_coord = {}

for coord in tqdm(selected_coordinates.itertuples(index=False), desc="50 nokta için ba?ar? ölçülüyor", file=sys.stdout):
    coord_train_data = train_data[(train_data['lat'] == coord.lat) & (train_data['lon'] == coord.lon)]
    coord_test_data = test_data[(test_data['lat'] == coord.lat) & (test_data['lon'] == coord.lon)]

    if coord_train_data.empty or coord_test_data.empty:
        print(f"Warning: {coord} coordinates do not have train or test data. Skipping.")
        continue

    X_train = coord_train_data.drop(columns=['delta_MGW'])
    y_train = coord_train_data['delta_MGW'].values.reshape(-1, 1)
    X_test = coord_test_data.drop(columns=['delta_MGW'])
    y_test = coord_test_data['delta_MGW'].values.reshape(-1, 1)

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)

    models = [model_rf, model_ridge, model_gb]
    smape_scores = []
    for model in models:
        model.fit(X_train_scaled, y_train_scaled.ravel())
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        smape_score = smape(y_test, y_pred)
        smape_scores.append(smape_score)

    best_two_models_indices = sorted(range(len(models)), key=lambda i: smape_scores[i])[:2]
    best_models_per_coord[(coord.lat, coord.lon)] = best_two_models_indices

all_predictions = {}
all_true_values = {}
all_smape_scores = {}

shap_values_dict = {}

distances = cdist(coordinates[['lat', 'lon']], selected_coordinates[['lat', 'lon']], metric='euclidean')
nearest_selected_idx = np.argmin(distances, axis=1)

with tqdm(total=len(coordinates), desc="Forecasting", file=sys.stdout) as pbar:
    for idx, coord in enumerate(coordinates.itertuples(index=False)):
        coord_train_data = train_data[(train_data['lat'] == coord.lat) & (train_data['lon'] == coord.lon)]
        coord_test_data = test_data[(test_data['lat'] == coord.lat) & (test_data['lon'] == coord.lon)]

        if coord_train_data.empty or coord_test_data.empty:
            print(f"Warnning: {coord} coordinates do not have train or test data. Skipping.")
            continue

        X_train = coord_train_data.drop(columns=['delta_MGW'])
        y_train = coord_train_data['delta_MGW'].values.reshape(-1, 1)
        X_test = coord_test_data.drop(columns=['delta_MGW'])
        y_test = coord_test_data['delta_MGW'].values.reshape(-1, 1)

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        X_test_scaled = scaler_X.transform(X_test)

        nearest_coord = selected_coordinates.iloc[nearest_selected_idx[idx]]
        best_two_models_indices = best_models_per_coord[(nearest_coord.lat, nearest_coord.lon)]

        best_two_models = [type(models[idx]).__name__ for idx in best_two_models_indices]

        meta_model = Ridge()

        y_preds = []
        for model_idx in best_two_models_indices:
            model = models[model_idx]
            model.fit(X_train_scaled, y_train_scaled.ravel())
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
            y_preds.append(y_pred)

        y_preds_combined = np.hstack(y_preds)
        meta_model.fit(y_preds_combined, y_test.ravel())
        weighted_prediction = meta_model.predict(y_preds_combined)

        smape_score = smape(y_test, weighted_prediction.reshape(-1, 1))
        all_smape_scores[(coord.lat, coord.lon)] = smape_score
        all_predictions[(coord.lat, coord.lon)] = weighted_prediction.reshape(-1, 1)
        all_true_values[(coord.lat, coord.lon)] = y_test

        meta_weights = meta_model.coef_

        if meta_weights[0] > meta_weights[1]:
            dominant_model = best_two_models[0]
        else:
            dominant_model = best_two_models[1]

        if dominant_model == 'RandomForestRegressor':
            explainer_rf = shap.TreeExplainer(models[best_two_models_indices[best_two_models.index('RandomForestRegressor')]])
            shap_values_rf = explainer_rf.shap_values(scaler_X.transform(coord_test_data.drop(columns=['delta_MGW'])))
            shap_values = shap_values_rf
            model_type = 'RandomForest'
        elif dominant_model == 'Ridge':
            explainer_ridge = shap.LinearExplainer(models[best_two_models_indices[best_two_models.index('Ridge')]], scaler_X.transform(coord_train_data.drop(columns=['delta_MGW'])))
            shap_values_ridge = explainer_ridge.shap_values(scaler_X.transform(coord_test_data.drop(columns=['delta_MGW'])))
            shap_values = shap_values_ridge
            model_type = 'Ridge'
        elif dominant_model == 'GradientBoostingRegressor':
            explainer_gb = shap.TreeExplainer(models[best_two_models_indices[best_two_models.index('GradientBoostingRegressor')]])
            shap_values_gb = explainer_gb.shap_values(scaler_X.transform(coord_test_data.drop(columns=['delta_MGW'])))
            shap_values = shap_values_gb
            model_type = 'GradientBoosting'

        shap_values_dict[(coord.lat, coord.lon)] = {
            'shap_values': shap_values,
            'model_type': model_type
        }

        pbar.update(1)

average_smape = np.mean(list(all_smape_scores.values()))
print(f'Mean SMAPE: {average_smape:.2f}%')

print(pd.Series(all_smape_scores.values()).describe())

sorted_smape_scores = sorted(all_smape_scores.items(), key=lambda x: x[1], reverse=True)
highest_smape_key, highest_smape_value = sorted_smape_scores[0]
print(f"Coordinate with highest SMAPE: {highest_smape_key}, De?er: {highest_smape_value:.2f}%")


########################################################################################################################
# Feature Importance: SHAP
########################################################################################################################
color_palette = ['#dc7077', '#eb9874', '#e4d692', '#89c684', '#2ba789']

feature_count = len(X_train.columns) - 2
cmap = LinearSegmentedColormap.from_list("custom_cmap", color_palette, N=feature_count)

def calculate_mean_shap_values(shap_values_dict, feature_names):
    total_shap_values = np.zeros(len(feature_names))
    count = 0

    for coord, shap_info in shap_values_dict.items():
        shap_values = shap_info['shap_values']
        shap_values_filtered = shap_values[:, 2:]
        total_shap_values += np.mean(np.abs(shap_values_filtered), axis=0)
        count += 1

    mean_shap_values = total_shap_values / count
    return mean_shap_values

feature_names = X_train.columns.tolist()
feature_names.remove('lat')
feature_names.remove('lon')

mean_shap_values = calculate_mean_shap_values(shap_values_dict, feature_names)

sorted_indices = np.argsort(mean_shap_values)[::-1]
sorted_shap_values = mean_shap_values[sorted_indices]
sorted_feature_names = np.array(feature_names)[sorted_indices]

bar_colors = [cmap(i / (feature_count - 1)) for i in range(feature_count)]

plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_feature_names, sorted_shap_values, color=bar_colors[::-1])

plt.xlabel('Average SHAP Value (Feature Importance)')
plt.ylabel('Features')
plt.title('Average SHAP Values for All Coordinates')
plt.gca().invert_yaxis()
plt.show()

########################################################################################################################
# Graph of selected coordinates
########################################################################################################################

# Assign each coordinate to the nearest selected_coordinate
coordinates = train_data[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
selected_coordinates = coordinates.iloc[np.linspace(0, len(coordinates) - 1, 50, dtype=int)]

# Find the nearest selected_coordinate based on Euclidean distances
distances = cdist(coordinates[['lat', 'lon']], selected_coordinates[['lat', 'lon']], metric='euclidean')
nearest_selected_idx = np.argmin(distances, axis=1)

# Add the nearest selected_coordinate index to each coordinate
coordinates['nearest_selected'] = nearest_selected_idx

# Color the points based on their nearest selected_coordinate
fig = px.scatter_geo(
    coordinates,
    lat='lat',
    lon='lon',
    color='nearest_selected',  # Color based on nearest selected_coordinate
    title="Grouping of Coordinates by Nearest Selected Coordinate",
    color_continuous_scale=px.colors.qualitative.Plotly  # Choose color scale
)

# Add selected_coordinates in a different color
fig.add_trace(
    go.Scattergeo(
        lat=selected_coordinates['lat'],
        lon=selected_coordinates['lon'],
        mode='markers',
        marker=dict(size=10, color='black'),
        name='Selected Coordinates'
    )
)

# Map settings
fig.update_geos(
    projection_type="equirectangular",
    lataxis_range=[-60, 90],
    showcoastlines=True,
    coastlinecolor="LightGray"
)

# Center the title and place it above the map
fig.update_layout(
    title={
        'text': "Grouping of Coordinates by Nearest Selected Coordinate",
        'y': 0.85,  # Adjusts the height of the title on the Y axis (0 = bottom, 1 = top)
        'x': 0.5,   # Centers the title on the X axis
        'xanchor': 'center',
        'yanchor': 'top'
    },
    titlefont=dict(size=24),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

# Show the map
fig.show()


########################################################################################################################
# Plot of actual and predicted values of the point with the highest smape value
########################################################################################################################
highest_smape_key, highest_smape_value = sorted_smape_scores[0]
print(f"Coordinate with the highest SMAPE: {highest_smape_key}, De?er: {highest_smape_value:.2f}%")

# Coordinates and values
coord = highest_smape_key
y_true = all_true_values[coord].flatten()  # Flattening the array for use
y_pred = all_predictions[coord].flatten()

dates = pd.date_range(start="2019-01-01", periods=len(y_true), freq='M')

# Create the figure
fig = go.Figure()

# Add the true values line
fig.add_trace(go.Scatter(x=dates, y=y_true, mode='lines', name='True Values',
                         line=dict(color='#e4d692', width=4, dash='solid')))

# Add the predicted values line
fig.add_trace(go.Scatter(x=dates, y=y_pred, mode='lines', name='Predicted Values',
                         line=dict(color='#2ba789', width=4, dash='dash')))

# Update layout with English labels and centered title
fig.update_layout(
    title={
        'text': f"True and Predicted Values for Coordinates {coord}",
        'y': 0.9,  # Adjust the height of the title
        'x': 0.5,  # Center the title
        'xanchor': 'center',
        'yanchor': 'top'
    },
    plot_bgcolor='white',
    paper_bgcolor='white',
    width=1200,
    height=500,
    #xaxis_title="Time",
    yaxis_title="Values",
    xaxis=dict(showgrid=True, gridcolor='LightGray', tick0=dates[0], tickformat='%b-%Y', dtick="M12"),  # X axis grid and title
    yaxis=dict(showgrid=True, gridcolor='LightGray', title_text="Values")   # Y axis grid and title
)

# Show the figure
fig.show()

########################################################################################################################
# Plot of actual and predicted values of the point with the lowest smape value
########################################################################################################################
lowest_smape_key, lowest_smape_value = sorted_smape_scores[-1]
print(f"Coordinate with the lowest SMAPE: {lowest_smape_key}, De?er: {lowest_smape_value:.2f}%")

# Coordinates and values
coord = lowest_smape_key
y_true = all_true_values[coord].flatten()  # Flattening the array for use
y_pred = all_predictions[coord].flatten()

dates = pd.date_range(start="2019-01-01", periods=len(y_true), freq='M')

# Create the figure
fig = go.Figure()

# Add the true values line
fig.add_trace(go.Scatter(x=dates, y=y_true, mode='lines', name='True Values',
                         line=dict(color='#e4d692', width=4, dash='solid')))

# Add the predicted values line
fig.add_trace(go.Scatter(x=dates, y=y_pred, mode='lines', name='Predicted Values',
                         line=dict(color='#2ba789', width=4, dash='dash')))

# Update layout with English labels and centered title
fig.update_layout(
    title={
        'text': f"True and Predicted Values for Coordinates {coord}",
        'y': 0.9,  # Adjust the height of the title
        'x': 0.5,  # Center the title
        'xanchor': 'center',
        'yanchor': 'top'
    },
    plot_bgcolor='white',
    paper_bgcolor='white',
    width=1200,
    height=500,
    #xaxis_title="Time",
    yaxis_title="Values",
    xaxis=dict(showgrid=True, gridcolor='LightGray', tick0=dates[0], tickformat='%b-%Y', dtick="M12"),  # X axis grid and title
    yaxis=dict(showgrid=True, gridcolor='LightGray', title_text="Values")   # Y axis grid and title
)

# Show the figure
fig.show()