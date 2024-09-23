
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import shap

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 500)

########################################################################################################################
# 1. GRACE VER?S?N? AYLIK DATAFRAMELERE D�N�?T�RME
########################################################################################################################

df_grace_filtered = pd.read_pickle("Grace/pkl_files/df_grace_filtered.pkl")

# 'date' s�tununu datetime format?na d�n�?t�rme (e?er gerekliyse)
df_grace_filtered['date'] = pd.to_datetime(df_grace_filtered['date'])
df_grace_filtered.info()


grace_filtered = df_grace_filtered.rename(columns={'date' : 'time'})

grace_monthly_dfs = {}

# Her bir ay i�in filtreleme ve ayr? bir DataFrame olu?turma
for month in grace_filtered['time'].dt.to_period('M').unique():
    # Ay? d�n�?t�rme ve ay? i�eren DataFrame'i olu?turma
    df_month = grace_filtered[grace_filtered['time'].dt.to_period('M') == month]

    # S�zl�?e ay? anahtar olarak ekleme
    grace_monthly_dfs[month] = df_month

# kaydet
with open(os.path.join('Grace', 'pkl_files', 'grace_monthly_dfs.pkl'), 'wb') as f:
    pickle.dump(grace_monthly_dfs, f)

########################################################################################################################
# 2. AYLIK GRACE VER?S?N? AYLIK GLDAS VER?S?NE S�TUN OLARAK EKLEME
########################################################################################################################
# gldas? a�
with open("Grace/pkl_files/monthly_gldas_dict_filtered_float16.pkl", "rb") as f:
    gldas_dict = pickle.load(f)

# gracei a�
with open("Grace/pkl_files/grace_monthly_dfs.pkl", "rb") as f:
    grace_dict = pickle.load(f)

# todo grace boylam?n? gldas format?na getir

# ?stenilen tarih aral???n? belirleme
start_date = pd.to_datetime('2010-01-01')
end_date = pd.to_datetime('2024-04-01')

# Tarih aral???nda bulunan aylar? belirleme
months_range = pd.date_range(start=start_date, end=end_date, freq='MS').to_period('M')

# Ayl?k verileri filtreleme ve birle?tirme
monthly_combined_dfs = {}

for month in months_range:
    # GRACE ve GLDAS verilerini filtreleme
    df_grace_month = grace_dict.get(month, pd.DataFrame())
    df_gldas_month = gldas_dict.get(month, pd.DataFrame())

    if not df_grace_month.empty and not df_gldas_month.empty:
        # 'time' s�tunlar?n? datetime format?na d�n�?t�rme (e?er gerekliyse)
        df_grace_month['time'] = pd.to_datetime(df_grace_month['time'])
        df_gldas_month['time'] = pd.to_datetime(df_gldas_month['time'])

        # GLDAS ve GRACE veri setlerini birle?tirme
        df_combined = pd.merge(df_gldas_month, df_grace_month[['time', 'lat', 'lon', 'lwe_thickness']],
                               on=['time', 'lat', 'lon'], how='left')

        # S�zl�?e ay? anahtar olarak ekleme
        monthly_combined_dfs[month] = df_combined

# burada bir kontrol etmeliyiz:
monthly_combined_dfs["2010-01"].head()
monthly_combined_dfs["2012-05"].head()
monthly_combined_dfs["2020-07"].head()

# Birle?tirilmi? verileri kaydetme (iste?e ba?l?)
with open('filtered_combined_monthly_data.pkl', 'wb') as file:
    pickle.dump(monthly_combined_dfs, file)

########################################################################################################################
# 3. DELTA GW ELDE ETMEK ?�?N DATAMIZA EKLEMEM?Z GEREK FEATURELAR
########################################################################################################################

#                ?MGw = ?TWS - ?MSM - ?MSN - ?MSw
#       Evap_tavg (April){kg/m2} = Evap_tavg (April){kg/m2/sec} * 10800{sec/3hr} * 8{3hr/day} * 30{days}
#                   Qs_acc (April){kg/m2} = Qs_acc (April){kg/m2/3hr} * 8{3hr/day} * 30{days}

#
# Bu form�lde,
#
# ?TWS: Belirli bir zaman aral???ndaki toplam su deposu de?i?imini (terajoule)
# ?MSM: Toprak nemi depolama de?i?imini (terajoule) = gldas["soil_moisture"]
# ?MSN: Kar depolama de?i?imini (terajoule) =gldas["snow_water_equivalent"]
# ?MGw: Yeralt? suyu depolama de?i?imini (terajoule)
# ?MSw: Y�zey suyu depolama de?i?imini (terajoule)
#      bunlar i�in Eda'n?n �nerileri:
#               MSw = (Rainf_f_tavg + Snowf_tavg + Qsm_acc + Qsb_acc) - (Evap_tavg + Qs_acc)
#               deltaTWS = ["lwe_thickness"]
#               MSN = SWE_inst


#

# S�zl�kteki her bir DataFrame i�in d�ng�
for month, df in monthly_combined_dfs.items():
    # Hesaplamay? ad?m ad?m kontrol et
    df['Rainf_f_tavg_m'] = df['Rainf_f_tavg'] * 10800 * 8 * 30
    df['Snowf_tavg_m'] = df['Snowf_tavg'] * 10800 * 8 * 30
    df['Evap_tavg_m'] = df['Evap_tavg'] *  10800 * 8 * 30
    df['Qsm_acc'] = df['Qsm_acc'] * 8 * 30
    df['Qs_acc'] = df['Qsc_acc'] * 8 * 30
    df['Qsb_acc'] = df['Qsb_acc'] * 8 * 30
    df['MSw'] = (df['Rainf_f_tavg_m'] + df['Snowf_tavg_m'] + df['Qsm_acc'] -
                 (df['Evap_tavg_m'] +  df['Qs_acc'] + df['Qsb_acc']))
    df.rename(columns={'lwe_thickness': 'deltaTWS'}, inplace=True)
    df['MSM'] = (df["SoilMoi0_10cm_inst"] + df["SoilMoi10_40cm_inst"] + df["SoilMoi40_100cm_inst"] +
                 df["SoilMoi100_200cm_inst"])
    df['MSN'] = df['SWE_inst']
    # G�ncellenmi? DataFrame'i s�zl�?e geri yaz (iste?e ba?l?)
    monthly_combined_dfs[month] = df

########################################################################################################################
# 4. EKLED???M?Z FEATURELARIN DELTA OLMASINI SA?LAMAK
########################################################################################################################

# Her bir DataFrame i�in i?lem yap?lacak d�ng�
for month, df in monthly_combined_dfs.items():
    # Tarih s�tununun DateTime format?nda oldu?unu varsay?yoruz
    # E?er de?ilse, �nce ?u sat?rla �evirmelisiniz: df['time'] = pd.to_datetime(df['time'])

    # 2004-2009 y?llar? aras?ndaki verileri filtreleme
    filtered_df = df[(df['time'].dt.year >= 2004) & (df['time'].dt.year <= 2009)]

    # Her ay i�in ortalamalar? hesaplama (Ocak, ?ubat, Mart... ?eklinde 12 ay i�in) bu yanl?? her ay i�in ortalama de?il tek bir ort kullancaz!!!
    monthly_avg = filtered_df.groupby(filtered_df['time'].dt.month)[['MSM', 'MSN', 'MSw']].mean()

    # Her sat?r?n ait oldu?u ay? bulma
    df['Month'] = df['time'].dt.month

    # ?MSM, ?MSN, ?MSw s�tunlar?n? olu?turma
    df['deltaMSM'] = df['MSM'] - df['Month'].map(monthly_avg['MSM'])
    df['deltaMSN'] = df['MSN'] - df['Month'].map(monthly_avg['MSN'])
    df['deltaMSw'] = df['MSw'] - df['Month'].map(monthly_avg['MSw'])

    # 'Month' s�tununu kald?rma
    df.drop(columns=['Month'], inplace=True)

    # G�ncellenen DataFrame'i s�zl�?e geri yazma (opsiyonel)
    monthly_combined_dfs[month] = df

########################################################################################################################
# 5. EKLED???M?Z FEATURELAR ?LE ?MGw = ?TWS - ?MSM - ?MSN - ?MSw ??LEM?N? GER�EKLE?T?RMEK
########################################################################################################################

# S�zl�kteki her bir DataFrame i�in d�ng�
for month, df in monthly_combined_dfs.items():
    # E?er gerekli s�tunlar varsa, deltaMGw s�tununu hesapla
    if all(col in df.columns for col in ['deltaTWS', 'deltaMSM', 'deltaMSN', 'deltaMSw']):
        # deltaMGw hesaplama
        df['deltaMGw'] = df['deltaTWS'] - df['deltaMSM'] - df['deltaMSN'] - df['deltaMSw']

    # G�ncellenen DataFrame'i tekrar s�zl�?e geri yaz
    monthly_combined_dfs[month] = df

########################################################################################################################
# 6. VER?M?Z? 2010 YILI SONRASI OLACAK ?EK?LDE G�NCELLEYEL?M:
########################################################################################################################

# S�zl�kteki her bir DataFrame i�in d�ng�
for month, df in monthly_combined_dfs.items():
    # Tarih s�tununun DateTime format?nda oldu?unu varsay?yoruz
    # E?er de?ilse, ?u sat?rla �evirebilirsiniz: df['time'] = pd.to_datetime(df['time'])

    # 2010 sonras? verileri filtreleme
    df = df[df['time'].dt.year > 2009]

    # G�ncellenen DataFrame'i tekrar s�zl�?e geri yaz
    monthly_combined_dfs[month] = df

########################################################################################################################
# 7. TRAIN - TEST AYRIMI
########################################################################################################################

# Train ve Test DataFrame'lerini saklamak i�in s�zl�kler olu?turuyoruz
train_dfs = {}
test_dfs = {}

# S�zl�kteki her bir DataFrame i�in d�ng�
for month, df in monthly_combined_dfs.items():
    # Tarih s�tununun DateTime format?nda oldu?unu varsay?yoruz
    # E?er de?ilse, ?u sat?rla �evirebilirsiniz: df['time'] = pd.to_datetime(df['time'])

    # Train dataset: 2010-2018 y?llar? aras? veriler
    train_df = df[(df['time'].dt.year >= 2010) & (df['time'].dt.year <= 2018)]

    # Test dataset: 2019-2024 y?llar? aras? veriler
    test_df = df[(df['time'].dt.year >= 2019) & (df['time'].dt.year <= 2024)]

    # Train ve test DataFrame'lerini ilgili s�zl�klere ekle
    train_dfs[month] = train_df
    test_dfs[month] = test_df

########################################################################################################################
# 8. MODEL FONKS?YONU
########################################################################################################################

# LSTM modeli olu?turma fonksiyonu
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))  # Tek bir �?k?? (regresyon i�in)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

########################################################################################################################
# 9. VER?Y? MODEL ?�?N KAZIRLAMA FONKS?YONU
########################################################################################################################

# Veriyi LSTM i�in haz?rlama fonksiyonu (her bir lat/lon i�in ayr?)
def prepare_data_for_lstm(df, target_col, time_steps=12):
    if target_col in df.columns:
        features = df.drop(columns=[target_col, 'lat', 'lon'])  # lat/lon haricindeki t�m �zellikler
        target = df[target_col]

        X, y = [], []
        for i in range(time_steps, len(df)):
            X.append(features.values[i - time_steps:i])
            y.append(target.values[i])

        return np.array(X), np.array(y)
    return None, None

########################################################################################################################
# 10. E??T?M VE TAHM?N
########################################################################################################################

# Her konum (lat/lon) i�in model e?itme ve tahmin yapma d�ng�s�
results = []  # Sonu�lar? saklamak i�in

for df_train, df_test in zip(train_dfs.values(), test_dfs.values()):
    lat = df_train['lat'].iloc[0]
    lon = df_train['lon'].iloc[0]

    # E?itim ve test verisini LSTM format?na uygun hale getiriyoruz
    X_train, y_train = prepare_data_for_lstm(df_train, target_col='deltaMGw')
    X_test, y_test = prepare_data_for_lstm(df_test, target_col='deltaMGw')

    if X_train is not None and X_test is not None:
        # Modeli olu?turuyoruz
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_lstm_model(input_shape)

        # Modeli e?itme
        print(f"Model e?itiliyor... Lat: {lat}, Lon: {lon}")
        model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

        # Test verisi �zerinde tahmin yapma
        predictions = model.predict(X_test)

        # Sonu�lar? kaydetme
        for pred, true_val in zip(predictions, y_test):
            results.append({
                'lat': lat,
                'lon': lon,
                'predicted_deltaMGw': pred[0],
                'true_deltaMGw': true_val
            })

results_df = pd.DataFrame(results)

########################################################################################################################
# 11. SONU�LARI G�RSELLE?T?RME
########################################################################################################################

# Bir konum i�in tahminleri ve ger�ek de?erleri kar??la?t?ral?m
def plot_predictions_for_location(results_df, lat, lon):
    location_df = results_df[(results_df['lat'] == lat) & (results_df['lon'] == lon)]

    plt.figure(figsize=(10, 6))
    plt.plot(location_df['true_deltaMGw'].values, label='Ger�ek deltaMGw', marker='o')
    plt.plot(location_df['predicted_deltaMGw'].values, label='Tahmin deltaMGw', marker='x')
    plt.title(f"Lat: {lat}, Lon: {lon} i�in Tahminler")
    plt.xlabel("Zaman Ad?m?")
    plt.ylabel("deltaMGw")
    plt.legend()
    plt.show()


# �rnek olarak bir konum i�in tahminleri �izelim
plot_predictions_for_location(results_df, lat=40.0, lon=30.0)

########################################################################################################################
# 12. SMAPE DE?ER? HESAPLAMA
########################################################################################################################

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


smape_results = []  # SMAPE sonu�lar?n? saklayaca??z

for (lat, lon), group_df in results_df.groupby(['lat', 'lon']):
    true_values = group_df['true_deltaMGw'].values
    predicted_values = group_df['predicted_deltaMGw'].values

    # SMAPE hesaplama
    smape_value = smape(true_values, predicted_values)

    # Sonu�lar? kaydetme
    smape_results.append({
        'lat': lat,
        'lon': lon,
        'smape': smape_value
    })

# SMAPE sonu�lar?n? DataFrame'e �evirme
smape_df = pd.DataFrame(smape_results)

average_smape = smape_df['smape'].mean()
print(f"Ortalama SMAPE: {average_smape}")

########################################################################################################################
# 13. FEATURE IMPORTANCE
########################################################################################################################

# 1. Permutasyon �nem Derecelendirmesi
# Permutasyon �nem derecelendirmesi, bir modelin tahmin g�c�n� �l�erek belirli �zelliklerin �nemini belirlemenizi sa?lar.
# �zelliklerin de?erlerini rastgele de?i?tirerek modelin performans?ndaki de?i?ikli?i inceler.


# LSTM modelini de?erlendirmek i�in �ncelikle bir sklearn regressor'a d�n�?t�rmeliyiz.
# Yani, tahminleri elde edip bir regresyon modeli olu?turmal?y?z.
# �rne?in, tahmin edilen y de?erlerini bir DataFrame'e koyup sklearn ile de?erlendirebiliriz.

# Tahmin edilen ve ger�ek de?erleri bir DataFrame'e koy
results_df = pd.DataFrame({'true': y_test, 'predicted': predictions.flatten()})

# �rnek olarak, feature �nem derecelendirmesini hesaplamak i�in bir regresyon modeli olarak RandomForest kullanabiliriz

# E?itim verisini haz?rlama
X_train_rf = X_train.reshape(X_train.shape[0], -1)  # 3D'den 2D'ye �evirme
X_test_rf = X_test.reshape(X_test.shape[0], -1)

# Random Forest modelini e?itme
rf_model = RandomForestRegressor()
rf_model.fit(X_train_rf, y_train)

# Permutasyon �nem derecelendirmesi
result = permutation_importance(rf_model, X_test_rf, y_test, n_repeats=30, random_state=0)

# Sonu�lar? yazd?rma
importance_df = pd.DataFrame({
    'feature': np.arange(X_train_rf.shape[1]),  # �zellik say?s?n? belirtin
    'importance': result.importances_mean
}).sort_values(by='importance', ascending=False)

print(importance_df)

# 2. SHAP De?erleri
# SHAP (SHapley Additive exPlanations) de?erleri, modelin tahminini a�?klamak i�in g��l� bir y�ntemdir.
# �zellikle karma??k modellerde (�rne?in LSTM) her bir �zelli?in tahmin �zerindeki etkisini anlaman?z? sa?lar.

# LSTM modelinin tahminlerini hesaplamak
explainer = shap.KernelExplainer(rf_model.predict, X_train_rf)
shap_values = explainer.shap_values(X_test_rf)

# ?lk �rnek i�in SHAP de?erlerini �izdirme
shap.initjs()
shap.summary_plot(shap_values, X_test_rf, feature_names=[f'Feature {i}' for i in range(X_test_rf.shape[1])])

########################################################################################################################
# 14. FEATURE IMPORTANCE G�RSELLE?T?RME
########################################################################################################################

# Permutasyon �nem derecelendirmesini g�rselle?tirme
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'].astype(str), importance_df['importance'])
plt.xlabel('�zellik �nemi')
plt.title('�zelliklerin �nemi (Permutasyon)')
plt.show()





