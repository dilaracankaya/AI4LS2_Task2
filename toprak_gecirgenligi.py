import geopandas as gpd
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 500)

# Path to the shapefile (only need to specify the .shp file)
shapefile_path = 'datasets/toprak_gecirgenligi/Durchlaessigkeit_BFW_Oesterreich.shp'

# Load the shapefile using GeoPandas
gdf = gpd.read_file(shapefile_path)

# Display the first few rows to understand the structure
print("First few rows of the GeoDataFrame:")
print(gdf.head())

# Save the attribute data (and geometry) to a CSV file
csv_output_path = 'datasets/toprak_gecirgenligi/Durchlaessigkeit_BFW_Oesterreich.csv'
gdf.to_csv(csv_output_path, index=False)

print(f"\nData saved to {csv_output_path}")

# Optionally, read the CSV back into a Pandas DataFrame
df = pd.read_csv(csv_output_path)

# Display the first few rows of the CSV data
print("\nFirst few rows of the DataFrame loaded from CSV:")
print(df.head())


df["federal_st"].nunique() # bizim 9 b�lge ile ayn?.
df["param"].value_counts() # todo bunlar da 9 b�lgenin kodu mu?
df["survey_yea"].nunique()

df["descriptio"].nunique()
df["descriptio"].value_counts()
"""
hoch ? y�ksek
m��ig ? orta
gering ? d�?�k
sehr hoch ? �ok y�ksek
sehr gering ? �ok d�?�k
m��ig bis hoch ? orta ile y�ksek aras?
gering bis m��ig ? d�?�k ile orta aras?
keine Angabe ? belirtilmemi?
sehr gering bis gering ? �ok d�?�k ile d�?�k aras?
hoch bis sehr hoch ? y�ksek ile �ok y�ksek aras?
"""

df["mapping_ar"].nunique()
df["mapping_ar"].value_counts()

df["beginlifes"].nunique()
df["beginlifes"].value_counts()

# Bunlar? dropluyorum ��nk� t�m sat?rlar ayn?
df["inspirei_1"].nunique() # 1
df["inspirei_2"].nunique() # todo bunu kontrol et
df["descript_1"].nunique() # todo bunu da

df["fid"].nunique() # hepsi unique. id asl?nda
df["inspireid_"].nunique() # fid ile ayn? asl?nda
df["id"].nunique() # TODO bu da ayn? m? yukar?dakiler ile? AT.4d283c4d-ddfc-4d27-b679-7d5881e66f85.SO.SoilObject. ifadesinden sonraki rakamlar? tek b?rak?p bak.


