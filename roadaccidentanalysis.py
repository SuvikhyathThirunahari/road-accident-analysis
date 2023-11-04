import numpy as npimport pandas as pd
import matplotlib.pyplot as pltimport seaborn as sns
from sklearn.feature_selection import SelectKBestfrom sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import chi2from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCAfrom imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_splitfrom sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifierfrom sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score!pip install dabl
import dabl#df = pd.read_csv("/kaggle/input/road-traffic-severity-classification/RTA Dataset.csv")
df = pd.read_csv("/content/rrrrr.csv")df.head()
print(df['Accident_severity'].value_counts())df['Accident_severity'].value_counts().plot(kind='bar')
df['Educational_level'].value_counts().plot(kind='bar')plt.figure(figsize=(6,5))
sns.catplot(x='Educational_level', y='Accident_severity', data=df)plt.xlabel("Educational level")
plt.xticks(rotation=60)plt.show()
df.describe()df.dtypes
df['Number_of_casualties'].fillna(0,inplace=True);df['Number_of_vehicles_involved'].fillna(0,inplace=True);
df['Number_of_casualties']=df['Number_of_casualties'].astype(int)df['Number_of_vehicles_involved']=df['Number_of_vehicles_involved'].astype(int)
dabl.plot(df, target_col='Accident_severity')df.corr()
print(df['Road_surface_type'].value_counts())
plt.figure(figsize=(6,5))
sns.countplot(x='Road_surface_type', hue='Accident_severity', data=df)plt.xlabel('Rode surafce type')
plt.xticks(rotation=60)plt.show
print(df['Road_surface_conditions'].value_counts())
plt.figure(figsize=(6,5))
sns.countplot(x='Road_surface_conditions', hue='Accident_severity', data=df)plt.xlabel('Rode condition type')
plt.xticks(rotation=60)plt.show
pivot_df = pd.pivot_table(data=df,
               index='Road_surface_conditions',               columns='Accident_severity',
               aggfunc='count')
fatal_df = pivot_df['Road_surface_type']fatal_df.fillna(0, inplace=True)
fatal_df['sum_of_injuries'] = fatal_df['Fatal injury'] + fatal_df['Serious Injury'] + fatal_df['Slight Injury']fatal_df
fatal_df_dry = (fatal_df.loc['Dry']/fatal_df.loc['Dry','sum_of_injuries'])*100fatal_df_dry
fatal_df_snow = (fatal_df.loc['Wet or damp']/fatal_df.loc['Wet or damp','sum_of_injuries'])*100
fatal_df_snow
df.groupby('Road_surface_conditions')['Accident_severity'].count()
df['Time'] = pd.to_datetime(df['Time'])
obj_cols = [col for col in df.columns if df[col].dtypes == 'object']obj_cols2 = [col for col in obj_cols if col != 'Accident_severity']
obj_cols2
new_df = df.copy()new_df['Hour_of_Day'] = new_df['Time'].dt.hour
n_df = new_df.drop('Time', axis=1)n_df
def count_plot(col):
    n_df[col].value_counts()
    # plot the figure of count plot    plt.figure(figsize=(5,5))
    sns.countplot(x=col, hue='Accident_severity', data=n_df)
    plt.xlabel(f'{col}')    plt.xticks(rotation=60)
    plt.show
for col in obj_cols:    count_plot(col)
plt.figure(figsize=(5,5))
sns.displot(x='Hour_of_Day', hue='Accident_severity', data=n_df)plt.show()
features = ['Day_of_week','Number_of_vehicles_involved','Number_of_casualties','Area_accident_occured',
           'Types_of_Junction','Age_band_of_driver','Sex_of_driver','Educational_level',           'Vehicle_driver_relation','Type_of_vehicle','Driving_experience','Service_year_of_vehicle','Type_of_collision',
           'Sex_of_casualty','Age_band_of_casualty','Cause_of_accident','Hour_of_Day']len(features)
featureset_df = n_df[features]
target = n_df['Accident_severity']
featureset_df.info()
feature_df = featureset_df.copy()
feature_df['Service_year_of_vehicle'] = feature_df['Service_year_of_vehicle'].fillna('Unknown')feature_df['Types_of_Junction'] = feature_df['Types_of_Junction'].fillna('Unknown')
feature_df['Area_accident_occured'] = feature_df['Area_accident_occured'].fillna('Unknown')feature_df['Driving_experience'] =
