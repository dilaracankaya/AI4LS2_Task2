# IMPORTS
import os
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
from scipy.spatial import distance
from collections import Counter
import seaborn as sns
import itertools
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 500)
warnings.simplefilter('ignore', category=ConvergenceWarning)

# FUNCTIONS
def station_coordinates(input):
    """
    Creates a dataset consisting of measurement station IDs and their corresponding X and Y coordinates.

    Args:
        input: Directory of the measurement station CSV file.

    Returns:
        df: A DataFrame containing columns "x", "y", and "hzbnr01".
    """
    df = pd.read_csv(os.path.join("Ehyd", "datasets_ehyd", input, "messstellen_alle.csv"), sep=";")
    output_df = df[["x", "y", "hzbnr01"]].copy()
    output_df['x'] = output_df['x'].astype(str).str.replace(',', '.').astype("float32")
    output_df['y'] = output_df['y'].astype(str).str.replace(',', '.').astype("float32")
    return output_df

def to_dataframe(folder_path, tip_coordinates):
    """
    Processes CSV files in the specified folder, skipping header information and creating DataFrames
    from the section marked by "Werte". Converts "L�cke" (Gap) values to NaN and skips rows with
    invalid data or specific keywords.

    For each CSV file, it extracts data starting after the "Werte:" line, processes date and value
    columns, and stores each DataFrame in a dictionary where the key is derived from the filename.
    Additionally, it matches IDs with tip coordinates and returns a DataFrame containing matched coordinates.

    Args:
        folder_path (str): The directory path where the CSV files are located.
        tip_coordinates (pd.DataFrame): A DataFrame containing coordinates to be matched with the IDs.

    Returns:
        dict: A dictionary where keys are IDs (extracted from filenames) and values are DataFrames.
        pd.DataFrame: A DataFrame with matched coordinates based on IDs.
    """
    dataframes_dict = {}
    coordinates = pd.DataFrame()

    for filename in os.listdir(folder_path):
        try:
            if filename.endswith(".csv"):
                filepath = os.path.join(folder_path, filename)

                with open(filepath, 'r', encoding='latin1') as file:
                    lines = file.readlines()

                    # Find the starting index of the data section
                    start_idx = next((i for i, line in enumerate(lines) if line.startswith("Werte:")), None)
                    if start_idx is None:
                        continue  # Skip files that do not contain 'Werte:'

                    start_idx += 1
                    header_line = lines[start_idx - 1].strip()

                    # Skip files with 'Invalid' in the header line
                    if "Invalid" in header_line:
                        continue

                    data_lines = lines[start_idx:]

                    data = []
                    for line in data_lines:
                        if line.strip():  # Skip empty lines
                            try:
                                date_str, value_str = line.split(';')[:2]

                                # Try multiple date formats
                                try:
                                    date = datetime.strptime(date_str.strip(), "%d.%m.%Y %H:%M:%S").date()
                                except ValueError:
                                    try:
                                        date = datetime.strptime(date_str.strip(), "%d.%m.%Y %H:%M").date()
                                    except ValueError:
                                        continue

                                # Skip rows with invalid data or specific keywords
                                if any(keyword in value_str for keyword in ["F", "K", "rekonstruiert aus Version 3->"]):
                                    continue

                                # Convert value to float
                                try:
                                    value = np.float32(value_str.replace(',', '.'))
                                except ValueError:
                                    value = np.nan  # Assign NaN if conversion fails

                                data.append([date, value])

                            except Exception:
                                break

                    if data:  # Create DataFrame only if data exists
                        df = pd.DataFrame(data, columns=['Date', 'Values'])
                        df.drop(df.index[-1], inplace=True)  # Dropping the last row (2022-01-01)
                        df_name = f"{filename[-10:-4]}"

                        dataframes_dict[df_name] = df

                        # Convert keys to integers
                        int_keys = [int(key) for key in dataframes_dict.keys() if key.isdigit()]
                        coordinates = tip_coordinates[tip_coordinates['hzbnr01'].isin(int_keys)]

        except Exception:
            continue

    return dataframes_dict, coordinates

def to_global(dataframes_dict, prefix=''):
    """
    Adds DataFrames from a dictionary to the global namespace with optional prefix.

    Args:
        dataframes_dict (dict): A dictionary where keys are names (str) and values are DataFrames.
        prefix (str): An optional string to prefix to each DataFrame name in the global namespace.
    """
    for name, dataframe in dataframes_dict.items():
        globals()[f"{prefix}{name}"] = dataframe

def process_dataframes(df_dict):
    """
    Processes a dictionary of DataFrames by converting date columns, resampling daily data to monthly, and reindexing.

    Args:
        df_dict (dict): A dictionary where keys are DataFrame names and values are DataFrames.

    Returns:
        dict: The processed dictionary of DataFrames with date conversion, resampling, and reindexing applied.
    """
    for df_name, df_value in df_dict.items():
        df_value['Date'] = pd.to_datetime(df_value['Date'])

        if df_value['Date'].dt.to_period('D').nunique() > df_value['Date'].dt.to_period('M').nunique():
            df_value.set_index('Date', inplace=True)
            df_dict[df_name] = df_value.resample('MS').mean()

        else:
            df_value.set_index('Date', inplace=True)
            df_dict[df_name] = df_value

        all_dates = pd.date_range(start='1960-01-01', end='2021-12-01', freq='MS')
        new_df = pd.DataFrame(index=all_dates)
        df_dict[df_name] = new_df.join(df_dict[df_name], how='left').fillna("NaN")

    return df_dict

def process_and_store_data(folder, coordinates, prefix, points_list=None):
    data_dict, data_coordinates = to_dataframe(folder, coordinates)
    data_dict = process_dataframes(data_dict)

    for df_name, df in data_dict.items():
        df.astype('float32')

    to_global(data_dict, prefix=prefix)

    if points_list:
        data_dict = filter_dataframes_by_points(data_dict, points_list)
        data_coordinates = data_coordinates[data_coordinates['hzbnr01'].astype(str).isin(points_list)]

    return data_dict, data_coordinates

def filter_dataframes_by_points(dataframes_dict, points_list):
    """
    Filters a dictionary of DataFrames to include only those whose names are specified in a given CSV file.

    Args:
        dataframes_dict (dict): A dictionary where keys are names (str) and values are DataFrames.
        points_list (str): Path to a CSV file that contains the names (str) of the DataFrames to filter.

    Returns:
        dict: A filtered dictionary containing only the DataFrames whose names are listed in the CSV file.
    """
    filtered_dict = {name: df for name, df in dataframes_dict.items() if name in points_list}
    return filtered_dict

def save_to_pickle(item, filename):
    """
    Saves a dictionary to a pickle file.

    Args:
        data_dict (dict): The dictionary to save.
        filename (str): The path to the output pickle file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(item, f)


########################################################################################################################
# Creating Dataframes from given CSVs
########################################################################################################################

# Define paths and coordinates
groundwater_all_coordinates = station_coordinates("Groundwater")
precipitation_coordinates = station_coordinates("Precipitation")
sources_coordinates = station_coordinates("Sources")
surface_water_coordinates = station_coordinates("Surface_Water")

# Precipitation: Rain and Snow
precipitation_folders = [
    ("N-Tagessummen", "rain_"),
    ("NS-Tagessummen", "snow_")]

source_folders = [
    ("Quellsch�ttung-Tagesmittel", "source_fr_"),
    ("Quellleitf�higkeit-Tagesmittel", "conductivity_"),
    ("Quellwassertemperatur-Tagesmittel", "source_temp_")]

surface_water_folders = [
    ("W-Tagesmittel", "surface_water_level_"),
    ("WT-Monatsmittel", "surface_water_temp_"),
    ("Schwebstoff-Tagesfracht", "sediment_"),
    ("Q-Tagesmittel", "surface_water_fr_")]

# Groundwater Dictionary (Filtered to requested 487 points)
points = pd.read_csv(os.path.join("Ehyd", "datasets_ehyd", "gw_test_empty.csv"))
points_list = [col for col in points.columns[1:]]
filtered_groundwater_dict, filtered_gw_coordinates = process_and_store_data(
    os.path.join("Ehyd", "datasets_ehyd", "Groundwater", "Grundwasserstand-Monatsmittel"),
    groundwater_all_coordinates, "gw_", points_list)

gw_temp_dict, gw_temp_coordinates = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Groundwater", "Grundwassertemperatur-Monatsmittel"), groundwater_all_coordinates, "gwt_")
rain_dict, rain_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Precipitation", precipitation_folders[0][0]), precipitation_coordinates, "rain_")
snow_dict, snow_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Precipitation", precipitation_folders[1][0]), precipitation_coordinates, "snow_")
source_fr_dict, source_fr_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Sources", source_folders[0][0]), sources_coordinates, "source_fr_")
conduct_dict, conduct_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Sources", source_folders[1][0]), sources_coordinates, "conduct_")
source_temp_dict, source_temp_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Sources", source_folders[2][0]), sources_coordinates, "source_temp_")
surface_water_lvl_dict, surface_water_lvl_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Surface_Water", surface_water_folders[0][0]), surface_water_coordinates, "surface_water_lvl_")
surface_water_temp_dict, surface_water_temp_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Surface_Water", surface_water_folders[1][0]), surface_water_coordinates, "surface_water_temp_")
sediment_dict, sediment_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Surface_Water", surface_water_folders[2][0]), surface_water_coordinates, "sediment_")
surface_water_fr_dict, surface_water_fr_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Surface_Water", surface_water_folders[3][0]), surface_water_coordinates, "surface_water_fr_")

########################################################################################################################
# Gathering associated features for 487 stations
########################################################################################################################
def calculate_distance(coord1, coord2):
    return distance.euclidean(coord1, coord2)

def find_nearest_coordinates(gw_row, df, k=20):
    distances = df.apply(lambda row: calculate_distance(
        (gw_row['x'], gw_row['y']),
        (row['x'], row['y'])
    ), axis=1)
    nearest_indices = distances.nsmallest(k).index
    return df.loc[nearest_indices]

# Creating a dataframe that stores all the associated features of the 487 stations.
data = pd.DataFrame()
def add_nearest_coordinates_column(df_to_add, name, k, df_to_merge=None):
    if df_to_merge is None:
        df_to_merge = data  # Use the current value of 'data' as the default
    results = []

    # Find the nearest points according to the coordinates
    for _, gw_row in filtered_gw_coordinates.iterrows():
        nearest = find_nearest_coordinates(gw_row, df_to_add, k)
        nearest_list = nearest['hzbnr01'].tolist()
        results.append({
            'hzbnr01': gw_row['hzbnr01'],
            name: nearest_list
        })

    results_df = pd.DataFrame(results)

    # Debug: Check if 'hzbnr01' exists in both dataframes
    print("Columns in df_to_merge:", df_to_merge.columns)
    print("Columns in results_df:", results_df.columns)

    # Ensure that the column exists in both dataframes before merging
    if 'hzbnr01' in df_to_merge.columns and 'hzbnr01' in results_df.columns:
        # Merge operation
        df = df_to_merge.merge(results_df, on='hzbnr01', how='inner')

        # Debug: Birle?tirilmi? DataFrame'i yazd?rarak kontrol et
        print("Merged DataFrame:")
        print(df.head())
    else:
        raise KeyError("Column 'hzbnr01' does not exist in one of the dataframes.")

    return df

data = add_nearest_coordinates_column(gw_temp_coordinates, 'nearest_gw_temp', 1, df_to_merge=filtered_gw_coordinates)
data = add_nearest_coordinates_column(rain_coord, 'nearest_rain', 3, df_to_merge=data) # TODO burada data arguman? default oldugu icin silebiliriz.
data = add_nearest_coordinates_column(snow_coord, 'nearest_snow', 3, df_to_merge=data)
data = add_nearest_coordinates_column(source_fr_coord, 'nearest_source_fr', 1, df_to_merge=data)
data = add_nearest_coordinates_column(conduct_coord, 'nearest_conductivity', 1, df_to_merge=data)
data = add_nearest_coordinates_column(source_temp_coord, 'nearest_source_temp', 1, df_to_merge=data)
data = add_nearest_coordinates_column(surface_water_lvl_coord, 'nearest_owf_level', 3, df_to_merge=data)
data = add_nearest_coordinates_column(surface_water_temp_coord, 'nearest_owf_temp', 1, df_to_merge=data)
data = add_nearest_coordinates_column(sediment_coord, 'nearest_sediment', 1, df_to_merge=data)
data = add_nearest_coordinates_column(surface_water_fr_coord, 'nearest_owf_fr', 3, df_to_merge=data)
data.drop(["x", "y"], axis=1, inplace=True)

directory = 'Ehyd/pkl_files'
file_path = os.path.join(directory, 'data.pkl')
save_to_pickle(data, file_path)

########################################################################################################################
# Imputing NaN Values
########################################################################################################################
def nan_imputer(dict):
    new_dict = {}
    for df_name, df in dict.items():
        df_copy = df.copy(deep=True)  # Create a deep copy
        df_copy.replace('NaN', np.nan, inplace=True)
        first_valid_index = df_copy['Values'].first_valid_index()
        valid_values = df_copy.loc[first_valid_index:].copy()

        # Fill NaNs with the corresponding monthly means
        for month in range(1, 13):
            month_mean = valid_values[valid_values.index.month == month]['Values'].dropna().mean()
            valid_values.loc[valid_values.index.month == month, 'Values'] = valid_values.loc[
                valid_values.index.month == month, 'Values'].fillna(month_mean)

        # Update the copied DataFrame with filled values
        df_copy.update(valid_values)
        new_dict[df_name] = df_copy  # Store the modified copy

    return new_dict

filled_filtered_groundwater_dict = nan_imputer(filtered_groundwater_dict)
filled_gw_temp_dict = nan_imputer(gw_temp_dict)
filled_rain_dict = nan_imputer(rain_dict)
filled_snow_dict = nan_imputer(snow_dict)
filled_source_fr_dict = nan_imputer(source_fr_dict)
filled_source_temp_dict = nan_imputer(source_temp_dict)
filled_conduct_dict = nan_imputer(conduct_dict)
filled_surface_water_fr_dict = nan_imputer(surface_water_fr_dict)
filled_surface_water_lvl_dict = nan_imputer(surface_water_lvl_dict)
filled_surface_water_temp_dict = nan_imputer(surface_water_temp_dict)
filled_sediment_dict = nan_imputer(sediment_dict)

########################################################################################################################
# Adding lagged values and rolling means
########################################################################################################################
filled_dict_list = [filled_gw_temp_dict, filled_filtered_groundwater_dict, filled_snow_dict, filled_rain_dict,
                    filled_conduct_dict, filled_source_fr_dict, filled_source_temp_dict, filled_surface_water_lvl_dict,
                    filled_surface_water_fr_dict, filled_surface_water_temp_dict, filled_sediment_dict]

# otokorelasyon
# from pandas.plotting import autocorrelation_plot
#
# # Zaman serisi i�in auto-correlation grafi?i olu?turma
#
# plt.figure(figsize=(10, 6))
# autocorrelation_plot(filled_filtered_groundwater_dict["300111"]["Values"])
# plt.show()
#
# from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(filled_filtered_groundwater_dict["300111"]["Values"], lags=50)
# plt.title('Autocorrelation - ?lk 50 Lag')
# plt.xlabel('Lag')
# plt.ylabel('Autocorrelation')
# plt.show()

def add_lag_and_rolling_mean(df, window=6):
    """
    Adds lagged and rolling mean columns to a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        window (int, optional): The window size for calculating the rolling mean. Default is 6.

    Returns:
        pandas.DataFrame: The DataFrame with additional columns for lagged values and rolling means.
    """
    column_name = df.columns[0]
    df['lag_1'] = df[column_name].shift(1)
    df['lag_2'] = df[column_name].shift(2)
    df['lag_3'] = df[column_name].shift(3)

    df["rolling_mean_original"] = df[column_name].rolling(window=window).mean()

    for i in range(1, 4):
        df[f'rolling_mean_{window}_lag_{i}'] = df["rolling_mean_original"].shift(i)
    return df

for dictionary in filled_dict_list:
    for key, df in dictionary.items():
        dictionary[key] = add_lag_and_rolling_mean(df)

########################################################################################################################
# Zero Padding and changing the data type to float32
########################################################################################################################
for dictionary in filled_dict_list:
    for key, df in dictionary.items():
        df.fillna(0, inplace=True)
        df = df.astype(np.float32)
        dictionary[key] = df

# finalleri pickle'a alma
directory = 'Ehyd/pkl_files'

for dictionary in filled_dict_list:
    dict_name = [name for name in globals() if globals()[name] is dictionary][0]
    filename = os.path.join(directory, f'final_{dict_name}.pkl')
    save_to_pickle(dictionary, filename)

# Calling pickle files back from the directory
pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

for pkl_file in pkl_files:
    file_path = os.path.join(directory, pkl_file)
    with open(file_path, 'rb') as file:
        var_name = pkl_file[6:-4]
        globals()[var_name] = pickle.load(file)

########################################################################################################################
# LSTM-formatted dataframes and the .pkl file
########################################################################################################################
# 732 dataframe

def add_features_to_df(row, date, feature_type, feature_dict, monthly_df):
    """
    Bu fonksiyon, belirli bir veri t�r� i�in �zellikleri monthly_df'ye ekler.

    Args:
    row (Series): Veri sat?r?.
    date (Timestamp): �zerinde �al???lan tarih.
    feature_type (str): �zellik t�r� ('Rain', 'Sediment' vb.).
    feature_dict (dict): ?lgili �zellik verisini i�eren s�zl�k.
    monthly_df (DataFrame): Sonu�lar? ekleyece?iniz DataFrame.
    """
    # temp_df'yi her �a?r?da s?f?rlay?n
    temp_df = pd.DataFrame(index=monthly_df.index)

    if feature_type == 'Groundwater':
        a = [row['hzbnr01']]
    else:
        a = row[f'nearest_{feature_type.lower()}']

    for i, code in enumerate(a):
        if str(code) in feature_dict:
            feature_df = feature_dict[str(code)]

            if date in feature_df.index:
                for feature in feature_df.columns:
                    # Burada her feature'? do?ru ?ekilde isimlendiriyoruz
                    column_name = f'{feature_type}_{i + 1}_{feature}'
                    temp_df[column_name] = np.nan  # Ge�ici olarak s�tunlar? ekle

                    temp_df.loc[str(row['hzbnr01']), column_name] = feature_df.loc[date, feature]
            else:
                print(f"Tarih {date}, {feature_type} code {code} i�in bulunamad?.")
        else:
            print(f"{feature_type} code {code} {feature_type.lower()}_dict'te bulunamad?.")

    # temp_df'yi monthly_df'ye ekle
    monthly_df = pd.concat([monthly_df, temp_df], axis=1)

    return monthly_df

def get_first_n_items(dictionary, n=2):
    # Convert the dictionary items to a list and slice the first n items
    first_n_items = list(dictionary.items())[:n]
    return dict(first_n_items)

date_range = pd.date_range(start='1960-01-01', end='2021-12-01', freq='MS')
monthly_dfs = {}
for date in date_range:
    monthly_df = pd.DataFrame(index=data['hzbnr01'].astype(str))
    monthly_df.index.name = 'hzbnr01'

    for i, row in data.iterrows():
        monthly_df = add_features_to_df(row, date, 'Groundwater', filled_filtered_groundwater_dict, monthly_df)
        monthly_df = add_features_to_df(row, date, 'Gw_temp', filled_gw_temp_dict, monthly_df)
        monthly_df = add_features_to_df(row, date, 'Rain', filled_rain_dict, monthly_df)
        monthly_df = add_features_to_df(row, date, 'Snow', filled_snow_dict, monthly_df)
        monthly_df = add_features_to_df(row, date, 'Source_fr', filled_source_fr_dict, monthly_df)
        monthly_df = add_features_to_df(row, date, 'Conductivity', filled_conduct_dict, monthly_df)
        monthly_df = add_features_to_df(row, date, 'Source_temp', filled_source_temp_dict, monthly_df)
        monthly_df = add_features_to_df(row, date, 'Owf_level', filled_surface_water_lvl_dict, monthly_df)
        monthly_df = add_features_to_df(row, date, 'Owf_temp', filled_surface_water_temp_dict, monthly_df)
        monthly_df = add_features_to_df(row, date, 'Owf_fr', filled_surface_water_fr_dict, monthly_df)
        monthly_df = add_features_to_df(row, date, 'Sediment', filled_sediment_dict, monthly_df)


    # Sonu�lar? s�zl�?e ekleyin
    monthly_dfs[f'{date.year}-{date.month}'] = monthly_df

# �al??t?rmay? dene !!
save_to_pickle(monthly_dfs, "monthly_dfs.pkl")


########################################################################################################################
# Normalization
########################################################################################################################
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_dfs = [pd.DataFrame(scaler.fit_transform(df), columns=df.columns) for df in list_of_dfs]

# todo son bir defa daha datatype'i kontrol edelim
for key, value in monthly_dfs.items():
    print(value.dtypes)

########################################################################################################################
# LSTM Model
########################################################################################################################
# 1. Veri Haz?rl???

# DataFrame'leri numpy array'lerine d�n�?t�r�p birle?tirin
data = np.array([df.values for df in list_of_dfs])  # (720, 487, 57)

# 2. Pencereleme
def create_windows(data, window_size, forecast_horizon):
    X, y = [], []
    num_time_steps = data.shape[0]

    for start in range(num_time_steps - window_size - forecast_horizon + 1):
        end = start + window_size
        X.append(data[start:end, :, :])
        y.append(data[end:end + forecast_horizon, :, :])

    X = np.array(X)
    y = np.array(y)

    # X'in boyutlar?n? (batch_size, time_steps, features) haline getir
    X = X.reshape(X.shape[0], X.shape[1], -1)
    y = y.reshape(y.shape[0], y.shape[1], -1)

    return X, y


window_size = 12 # window size'? hala tam olarak anlayamad?m
forecast_horizon = 26 # �n�m�zdeli 26 ay? tahmin edece?iz
X, y = create_windows(data, window_size, forecast_horizon)  # X: (672, 12, 487, 5), y: (672, 26, 487, 5)

# 3. E?itim ve Test Setlerine B�lme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. LSTM Modelini Olu?turma ve E?itim
model = Sequential()
model.add(LSTM(units=57, return_sequences=True, input_shape=(window_size, data.shape[1] * data.shape[2])))
model.add(LSTM(units=57, return_sequences=True))
model.add(Dense(data.shape[2]))  # �?k?? katman?, tahmin edilmesi gereken s�tun say?s?na g�re ayarlanmal?
model.compile(optimizer='adam', loss='mse')

# Modeli e?itme
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

