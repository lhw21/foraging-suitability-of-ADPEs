### 1. Extraction of track points on the Antarctic Peninsula

```python
# This program is used to extract the track data 
# located near the Antarctic Peninsula
# and View data distribution

import pandas as pd
import numpy as np
import pygmt
import matplotlib.pyplot as plt

file = 'ADPE_ssm_by_id_predicted.csv'
df = pd.read_csv(file)
print(df.columns)
df1 = df[df['decimal_longitude']<=0]
df2 = df1[df1['decimal_longitude']>=-90]

df2['date'] = pd.to_datetime(df2['date'])
df2['Year'] = np.nan
df2['Month'] = np.nan

for i in range(len(df2)):
    df2.iloc[i,7] = df2.iloc[i,2].year
    df2.iloc[i,8] = df2.iloc[i,2].month
    
# df2['Year'].value_counts()
# df2['Month'].value_counts()

# Set the region for the plot to be slightly larger 
# than the data bounds.
region = [-67,-37,-73,-59]
fig = pygmt.Figure()
fig.basemap(region=region, projection="S-52.044/-90/12c", frame=True)
fig.coast(land="black", water="skyblue")
fig.plot(x=df2['decimal_longitude'], y=df2['decimal_latitude'], style="c0.3c", fill="white", pen="black")
fig.savefig('Track_AP.jpg',dpi=300)

# output data
df2.to_csv('AP_presence_data.csv')
```

### 2. Generate random background points

```python
# This program is used to generate background points
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy import distance
import datetime

file_i = 'AP_presence_data.csv'
df_loc = pd.read_csv(file_i)
df_loc.columns
lon = df_loc['decimal_longitude']
lat = df_loc['decimal_latitude']

# 25km away from the presence point
def gen_background_points(number, minx,maxx,miny,maxy,buffer_dis=25):
    points_list = []
    for i in range(number):
        print(i)
        valid = False
        while not valid:
            x = np.random.uniform(minx,maxx)
            y = np.random.uniform(miny,maxy)
            loc1 = (y, x)
            
            dis_list = []
            for j in range(len(lon)):
                loc2 = (lat[j],lon[j])
                dis_list.append(distance.great_circle(loc1,loc2).km)
            dis_arr = np.array(dis_list)
            
            if dis_arr.min()>buffer_dis:
                points_list.append(loc1)
                valid = True
                
    return points_list

minx,maxx,miny,maxy = -74,-37,-73,-59
background_points = gen_background_points(100000,minx,maxx,miny,maxy)

background_points = np.array(background_points)
np.save('background_points_AP.npy',background_points)
```

### 3. Tailoring the original ocean condition netcdf data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

# take the total rain as example
nc_file = 'G:/ecwmf_dat/ecwmf_era5_sl_monthly_tp_1940_2024.nc' 
ds = xr.open_dataset(nc_file,decode_times=True)

# select time span
ds = ds.sel(time=slice('1969-1-1','2021-12-1'))

var_name = 'tp'
msl = ds[var_name]
msl = msl.mean('expver')

# select spatial extent
lat1,lat2,lon1,lon2 = -82,-50,273,344
msl1 = msl.loc[:,lat2:lat1,lon1:lon2]

# select months
msl2 = msl1.loc[{"time": (msl.time.dt.month == 3) | (msl.time.dt.month == 2) | (msl.time.dt.month == 1) |
         (msl.time.dt.month == 10) | (msl.time.dt.month == 11) | (msl.time.dt.month == 12)}]

msl2.to_netcdf('G:/ecwmf_dat/ONDJFM_'+var_name+'_196901_202112.nc')
```

### 4. The 25km*25km grid points in the AP region were reconstructed (using the NSIDC Sea Ice Polar Stereographic South projection with a spatial resolution of 25km), and identify ocean data points within 25km of these grid points

#### 4.1. Computed distance matrix

```python
import pandas  as pd
import numpy as np
import netCDF4 as nc
import os
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from random import randint
import struct
from nsidc_polar_lonlat import nsidc_polar_lonlat
from nsidc_polar_ij import nsidc_polar_ij

# Function: Calculates the distance between two points
def haversine_distance(x1, y1, x2, y2):
    # Convert coordinates to radians
    lat1 = np.deg2rad(x1) # Latitude 1
    lon1 = np.deg2rad(y1) # Longitude 1
    lat2 = np.deg2rad(x2) # Latitude 2
    lon2 = np.deg2rad(y2) # Longitude 2

    # Calculate the difference between latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Apply the Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Calculate the distance between the two coordinates
    distance = 6371 * c
    return distance

# Read a 25km NISDC Antarctic sea ice data file 
file_name = 'nt_197811_n07_v1.1_s.bin'
file = open(file_name,'rb')
CONTENT=file.read()
flag = CONTENT[:300]
rawstring=CONTENT[300:]
data_raw = struct.unpack('B'*len(rawstring),rawstring)
file.close()
data=np.array(data_raw)
data=np.reshape(data,(332,316))
data=data.astype(float)
data[data>=251]=np.nan

# Select the data range for the Antarctic Peninsula region
data_select = data[65:166,25:125]
# Create two new arrays to store latitude and longitude
lon_arr = np.zeros(data_select.shape)*np.nan
lat_arr = np.zeros(data_select.shape)*np.nan
for i in range(data_select.shape[0]):
    i_index = i+65
    for j in range(data_select.shape[1]):
        j_index = j+25
        lonlat = nsidc_polar_ij(j_index, i_index, 25, -1)
        lon_arr[i,j] = lonlat[0]
        lat_arr[i,j] = lonlat[1]

# Computed distance matrixs
mete_data_file = './NDJF_tp.nc'
dat = nc.Dataset(mete_data_file)
lon_mete_arr = np.array(dat.variables['longitude'][:])
lat_mete_arr = np.array(dat.variables['latitude'][:])
distance = 25

lon_matrix,lat_matrix = np.meshgrid(lon_mete_arr,lat_mete_arr)
dis_matrix_list = np.zeros((lon_arr.shape[0],lon_arr.shape[1],lon_matrix.shape[0],lon_matrix.shape[1]))

for k in range(lon_arr.shape[0]):
    print(k)
    for t in range(lon_arr.shape[1]):
        lon_test = np.zeros(lon_matrix.shape)+lon_arr[k,t]
        lat_test = np.zeros(lon_matrix.shape)+lat_arr[k,t]
        dis_flag_temp = haversine_distance(lat_matrix,lon_matrix,lat_test,lon_test)
        dis_matrix_list[k,t,:,:] = dis_flag_temp < distance
        
np.save('nisdc_lon_arr_AP.npy',lon_arr)
np.save('nisdc_lat_arr_AP.npy',lat_arr)
np.save('nisdc_dis_matrix_AP.npy',dis_matrix_list)
```

### 5. Match ocean condition elements for each 25km grid point

```python
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nc_file_list = ['G:/ecwmf_dat/ONDJFM_siconc_196901_202112.nc',
 'G:/ecwmf_dat/ONDJFM_sst_196901_202112.nc']
var_name_list = ['siconc','sst']

# Read the latitude and longitude file and the distance matrix
nisdc_lon_matrix = np.load('nisdc_lon_arr_AP.npy')
nisdc_lat_matrix = np.load('nisdc_lat_arr_AP.npy')
nisdc_disflag_matrix = np.load('nisdc_dis_matrix_AP.npy')

for file_i in range(1,len(nc_file_list)):
    # Read the Marine data file
    nc_file = nc_file_list[file_i]
    ds = xr.open_dataset(nc_file,decode_times=True)
    var_name = var_name_list[file_i]
    print(nc_file)
    print(var_name)
    # Create a new array to store the extracted data
    data_ext_temp = np.zeros((len(ds.time),nisdc_lon_matrix.shape[0],nisdc_lon_matrix.shape[1]))*np.nan

    for i in range(len(ds.time)):
        print(i)
        test_ym = np.array(ds[var_name][i,:,:])
        for k in range(nisdc_lon_matrix.shape[0]):
            for t in range(nisdc_lon_matrix.shape[1]):
                data_ext_temp[i,k,t] = np.nanmean(test_ym[nisdc_disflag_matrix[k,t,:,:]>0])

    np.save('G:/ecwmf_dat/nisdc_ONDJFM_' + var_name + '_196901_202112.npy',data_ext_temp)
```

### 6. Matching sea ice, sea surface temperature and topographic data for track points and random background points (using random background points as an example)

#### 6.1. calculate the nearest point index of each random background point

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import sample

def haversine_distance(x1, y1, x2, y2):
    # Convert coordinates to radians
    lat1 = np.deg2rad(x1) # Latitude 1
    lon1 = np.deg2rad(y1) # Longitude 1
    lat2 = np.deg2rad(x2) # Latitude 2
    lon2 = np.deg2rad(y2) # Longitude 2

    # Calculate the difference between latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Apply the Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Calculate the distance between the two coordinates
    distance = 6371 * c
    return distance

bk_loc_file = 'background_points_AP.npy' 
bk_loc = np.load(bk_loc_file)

bk_loc_df = pd.DataFrame(bk_loc)
bk_loc_df.columns = ['lat', 'lon']

nisdc_lon = np.load('nisdc_lon_arr_AP.npy')
nisdc_lat = np.load('nisdc_lat_arr_AP.npy')
# Calculate the position of the nearest point from a point of latitude and longitude
loc_index_list = []
for i in range(len(bk_loc_df)):
    lon_t = bk_loc_df['lon'][i]
    lat_t = bk_loc_df['lat'][i]
    
    dis_matrix = haversine_distance(nisdc_lat, nisdc_lon, lat_t, lon_t)
    
    index = np.unravel_index(dis_matrix.argmin(), dis_matrix.shape)
    loc_index_list.append(index)
    
np.save('bk_loc_index_list.npy',loc_index_list)
```

#### 6.2. Randomly generates the year and month for random background points

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import sample

bk_loc_file = 'background_points_AP.npy' 
bk_loc = np.load(bk_loc_file)

bk_loc_df = pd.DataFrame(bk_loc)
bk_loc_df.columns = ['lat', 'lon']

bk_loc_df['Year'] = np.nan
bk_loc_df['Month'] = np.nan
# Now randomly match the background points 'Year' and 'Month'
# Random range of year []; Random range of month []
year_list = [1996,1997,1999,2000,2001,2002,2003,2004,
2005,2006,2007,2010,2012,2013]
month_list = [11,12,1,2]

for t in range(len(bk_loc_df)):
    print(t)
    bk_loc_df['Year'][t] = sample(year_list, 1)[0]
    bk_loc_df['Month'][t] = sample(month_list, 1)[0]
    
bk_loc_df.to_csv('absence_point_AP.csv')
```

#### 6.3. Match ocean conditions for random background points

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read point of presence
file1 = 'absence_point_AP.csv'
pre_df = pd.read_csv(file1)

# Reads data from the point of presence index
pre_loc_index = np.load('bk_loc_index_list.npy')

env_file_list = ['G:/ecwmf_dat/nisdc_ONDJFM_siconc_196901_202112.npy',
                 'G:/ecwmf_dat/nisdc_ONDJFM_sst_196901_202112.npy']

var_name_list = ['siconc','sst']

for file_i in range(len(env_file_list)):
    print(file_i)
    env_file = env_file_list[file_i]
    var_name = var_name_list[file_i]
    # Read environment data
    env = np.load(env_file)
    pre_df[var_name] = np.nan
    
    for i in range(len(pre_df)):
        year_i = pre_df['Year'][i]
        month_i = pre_df['Month'][i]
        index_i = pre_loc_index[i]
        if month_i == 1:
            time_index = 0 + (year_i-1969)*6
        if month_i == 2:
            time_index = 1 + (year_i-1969)*6
        if month_i == 3:
            time_index = 2 + (year_i-1969)*6        
        if month_i == 10:
            time_index = 3 + (year_i-1969)*6
        if month_i == 11:
            time_index = 4 + (year_i-1969)*6        
        if month_i == 12:
            time_index = 5 + (year_i-1969)*6        
        aim_point_dat = env[int(time_index), int(index_i[0]), int(index_i[1])]
        pre_df[var_name][i] = aim_point_dat
        
pre_df.to_csv('Random_background_points_matched.csv')
```

#### 6.4. Matched bathymetric data

```python
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic

def extracted_mete_etopo(lon_i, lat_i, lon_arr, lat_arr, aim_mete_arr):    
    lon_index = np.argmin(np.abs(lon_i - lon_arr))
    lat_index = np.argmin(np.abs(lat_i - lat_arr))
    region_dat = aim_mete_arr[lat_index, lon_index] 
    return region_dat

file = 'G:/ecwmf_dat/ETOPO1_Bed_g_gmt4.grd'
data = xr.open_dataset(file)
lon_arr = np.array(data.x)
lat_arr = np.array(data.y)
z = np.array(data.z)

loc_file = 'Random_background_points_matched.csv'
loc_df = pd.read_csv(loc_file)

ext_mete_list = []
for i in range(len(loc_df)):
    print(i)
    lon_i = loc_df['lon'][i]
    lat_i = loc_df['lat'][i]
    mete_arr = z
    ext_mete = extracted_mete_etopo(lon_i, lat_i, lon_arr, lat_arr, mete_arr)
    ext_mete_list.append(ext_mete)

loc_df['z'] = ext_mete_list
loc_df.to_csv('Random_background_points_matched.csv')
```

### 7. Build model

```python
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# ------Presence_points
file_pr = 'Presence_points_matched.csv'
pr_df = pd.read_csv(file_pr)
pr_df = pr_df[['siconc',  'sst',  'z']]
pr_df['flag'] = 1
pr_df = pr_df[pr_df['z']<0]

# ------Abesence_points
file_ab = 'Random_background_points_matched.csv'
ab_df = pd.read_csv(file_ab)
ab_df = ab_df[['siconc',  'sst',  'z']]
ab_df = ab_df.dropna()
ab_df = ab_df[ab_df['z']<0]
ab_df['flag'] = 0

# ------Sampling background point
ab_df_10000 = ab_df.sample(n=10000,random_state=1,replace = False)

# ------Integrate the data
total_df = ab_df_10000.append(pr_df)
total_df = shuffle(total_df)
features = ['sst','z','siconc']
X = total_df[features]
Y = total_df['flag']

# ------Divide the training data set and validation data set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# ------Train a random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ------Computational feature importance
feature_importances = rf.feature_importances_
feature_names = features
# ------Rank the importance of features
sorted_idx = np.argsort(feature_importances)
# ------Draw a feature importance bar chart
plt.figure(figsize=(10, len(feature_names)))
plt.barh(range(len(feature_names)), feature_importances[sorted_idx])
plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

y_pred_prob = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)

# ------Calculate the model evaluation index
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"AUC: {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# ------cross-verify
rfc_s = cross_val_score(rf,X,Y,cv=5)

# ------Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# ------Draw partial dependence curve
PartialDependenceDisplay.from_estimator(rf, X_train, [0,1,2])
plt.show()
```

### 8. Sensitivity test

#### 8.1. Build 100 models

```python
import os
import matplotlib.pyplot as plt
from random import sample
import numpy as np
import pandas as pd
import xarray as xr
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.inspection import PartialDependenceDisplay 
import joblib

def haversine_distance(x1, y1, x2, y2):
    lat1 = np.deg2rad(x1) 
    lon1 = np.deg2rad(y1) 
    lat2 = np.deg2rad(x2) 
    lon2 = np.deg2rad(y2) 

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = 6371 * c
    return distance

def extracted_mete_etopo(lon_i, lat_i, lon_arr, lat_arr, aim_mete_arr):    
    lon_index = np.argmin(np.abs(lon_i - lon_arr))
    lat_index = np.argmin(np.abs(lat_i - lat_arr))
    region_dat = aim_mete_arr[lat_index, lon_index] 
    return region_dat

for i in range(5,101):
    print(i)
    # ----Create a new folder
    path = 'G:/CMIP6_SST_SIC/second_download/sensitivity_test'
    model_dir = 'model_no_'+str(i)
    os.mkdir(path + './'+model_dir)
    outpath = path + './'+model_dir+'/'

    # ----Match the environmental factors and topographical data of the background points
    bk_loc_file = 'background_points_AP.npy' 
    bk_loc = np.load(bk_loc_file)

    bk_loc_df = pd.DataFrame(bk_loc)
    bk_loc_df.columns = ['lat', 'lon']

    bk_loc_df['Year'] = np.nan
    bk_loc_df['Month'] = np.nan

    year_list = [1996,1997,1999,2000,2001,2002,2003,2004,
    2005,2006,2007,2010,2012,2013]
    month_list = [11,12,1,2]

    for t in range(len(bk_loc_df)):
        bk_loc_df['Year'][t] = sample(year_list, 1)[0]
        bk_loc_df['Month'][t] = sample(month_list, 1)[0]

    bk_loc_df.to_csv(outpath+'absence_point_AP.csv')

    #----Match environment variables
    file1 = outpath+'absence_point_AP.csv'
    pre_df = pd.read_csv(file1)

    pre_loc_index = np.load('./bk_loc_index_list.npy')

    env_file_list = ['G:/ecwmf_dat/nisdc_ONDJFM_siconc_196901_202112.npy',
                     'G:/ecwmf_dat/nisdc_ONDJFM_sst_196901_202112.npy']

    var_name_list = ['siconc',
                    'sst']

    for file_i in range(len(env_file_list)):
        env_file = env_file_list[file_i]
        var_name = var_name_list[file_i]
        env = np.load(env_file)
        pre_df[var_name] = np.nan

        for i in range(len(pre_df)):
            year_i = pre_df['Year'][i]
            month_i = pre_df['Month'][i]
            index_i = pre_loc_index[i]
            if month_i == 1:
                time_index = 0 + (year_i-1969)*6
            if month_i == 2:
                time_index = 1 + (year_i-1969)*6
            if month_i == 3:
                time_index = 2 + (year_i-1969)*6        
            if month_i == 10:
                time_index = 3 + (year_i-1969)*6
            if month_i == 11:
                time_index = 4 + (year_i-1969)*6        
            if month_i == 12:
                time_index = 5 + (year_i-1969)*6        
            aim_point_dat = env[int(time_index), int(index_i[0]), int(index_i[1])]
            pre_df[var_name][i] = aim_point_dat

    pre_df.to_csv(outpath+'Random_background_points_matched.csv')

    # ----Matched bathymetric data
    file = 'G:/ecwmf_dat/ETOPO1_Bed_g_gmt4.grd'
    data = xr.open_dataset(file)

    lon_arr = np.array(data.x)
    lat_arr = np.array(data.y)
    z = np.array(data.z)

    loc_file = outpath+'Random_background_points_matched.csv'
    loc_df = pd.read_csv(loc_file)

    ext_mete_list = []
    for i in range(len(loc_df)):
    #     print(i)
        lon_i = loc_df['lon'][i]
        lat_i = loc_df['lat'][i]
        mete_arr = z
        ext_mete = extracted_mete_etopo(lon_i, lat_i, lon_arr, lat_arr, mete_arr)
        ext_mete_list.append(ext_mete)

    loc_df['z'] = ext_mete_list

        loc_df.to_csv(outpath+'Random_background_points_matched2.csv')

    # ----start modelling
    pr_df = pd.read_csv(file_pr)

    pr_df = pr_df[['siconc', 'sst','z']]
    pr_df['flag'] = 1
    pr_df = pr_df[pr_df['z']<0]

    file_ab = outpath+'Random_background_points_matched2.csv'
    ab_df = pd.read_csv(file_ab)
    ab_df = ab_df[['siconc', 'sst','z']]
    ab_df = ab_df.dropna()
    ab_df = ab_df[ab_df['z']<0]
    ab_df['flag'] = 0

    ab_df_10000 = ab_df.sample(n=10000,replace = False) 

    total_df = ab_df_10000.append(pr_df)
    total_df = shuffle(total_df)

    features = ['sst','z','siconc']
    X = total_df[features]
    Y = total_df['flag']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rfc_s = cross_val_score(rf,X,Y,cv=5)

    metric_df = pd.DataFrame({'AUC':auc,'Precision':precision,'Recall':recall,
                              'F1 Score':f1,'Accuracy':accuracy,'mean_cv5':np.mean(rfc_s)}, index=[0])
    metric_df.to_csv(outpath+model_dir+'_metric.csv')

    fig, ax = plt.subplots(figsize=(18, 6))
    PartialDependenceDisplay.from_estimator(rf, X_train, [0,1,2],ax=ax,line_kw={"color": "red"})
    plt.savefig(outpath+model_dir+'_pdp.jpg',dpi=300,bbox_inches='tight')

    model_path = outpath+model_dir+'.m'
    joblib.dump(rf, model_path)
```

### 9. Predicting Future Foraging Suitability

#### 9.1. Sample future SIC and SST data into 25km x 25km grid.

##### 9.1.1. Calculate the distance matrix (using UKESM-1-0-LL as an example)

```python
import matplotlib.pyplot as plt
import pandas  as pd
import numpy as np
import netCDF4 as nc
import os
from geopy.distance import geodesic
from random import randint

lon_arr = np.load('G:/ecwmf_dat/future_env/nisdc_lon_arr_AP.npy')
lat_arr = np.load('G:/ecwmf_dat/future_env/nisdc_lat_arr_AP.npy')

def haversine_distance(x1, y1, x2, y2):
    lat1 = np.deg2rad(x1) 
    lon1 = np.deg2rad(y1) 
    lat2 = np.deg2rad(x2) 
    lon2 = np.deg2rad(y2) 

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = 6371 * c
    return distance

mete_data_file = 'G:/CMIP6_SST_SIC/second_download/UKESM-1-0-LL/UKESM-1-0-LL_siconc_historical_ssp126_197901_210012.nc'
dat = nc.Dataset(mete_data_file)
lon_mete_arr = np.array(dat.variables['longitude'][:])
lat_mete_arr = np.array(dat.variables['latitude'][:])

distance = 25
lon_matrix,lat_matrix = lon_mete_arr,lat_mete_arr
dis_matrix_list = np.zeros((lon_arr.shape[0],lon_arr.shape[1],lon_matrix.shape[0],lon_matrix.shape[1]))

for k in range(lon_arr.shape[0]):
    print(k)
    for t in range(lon_arr.shape[1]):
        lon_test = np.zeros(lon_matrix.shape)+lon_arr[k,t]
        lat_test = np.zeros(lon_matrix.shape)+lat_arr[k,t]
        dis_flag_temp = haversine_distance(lat_matrix,lon_matrix,lat_test,lon_test)
        dis_matrix_list[k,t,:,:] = (dis_flag_temp == dis_flag_temp.min())
        if lat_arr[k,t] < lat_mete_arr.min(): # If the boundary is exceeded, it is set to np.nan
            dis_matrix_list[k,t,:,:] = np.zeros(dis_flag_temp.shape)
            
np.save('nisdc_lon_arr_AP.npy',lon_arr)
np.save('nisdc_lat_arr_AP.npy',lat_arr)
np.save('UKESM-1-0-LL_nisdc_dis_matrix_AP_future_env.npy',dis_matrix_list)
```

##### 9.1.2. Average the ocean conditions within a 25km radius.

```python
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nc_file_list = ['G:/CMIP6_SST_SIC/second_download/UKESM-1-0-LL/UKESM-1-0-LL_siconc_historical_ssp126_197901_210012.nc',
               'G:/CMIP6_SST_SIC/second_download/UKESM-1-0-LL/UKESM-1-0-LL_siconc_historical_ssp585_197901_210012.nc',
               'G:/CMIP6_SST_SIC/second_download/UKESM-1-0-LL/UKESM-1-0-LL_tos_historical_ssp126_197901_210012.nc',
               'G:/CMIP6_SST_SIC/second_download/UKESM-1-0-LL/UKESM-1-0-LL_tos_historical_ssp585_197901_210012.nc']

model_list = ['UKESM-1-0-LL_nisdc_','UKESM-1-0-LL_nisdc_','UKESM-1-0-LL_nisdc_','UKESM-1-0-LL_nisdc_']
ssp_list = ['ssp126','ssp585','ssp126','ssp585']
var_name_list = ['siconc','siconc','tos','tos']

# Read the latitude and longitude files and the distance matrix.
nisdc_lon_matrix = np.load('nisdc_lon_arr_AP.npy')
nisdc_lat_matrix = np.load('nisdc_lat_arr_AP.npy')
nisdc_disflag_matrix = np.load('UKESM-1-0-LL_nisdc_dis_matrix_AP_future_env.npy')

for file_i in range(len(nc_file_list)):
    # Read the ocean data file.
    nc_file = nc_file_list[file_i]
    print(nc_file)
    ds = xr.open_dataset(nc_file,decode_times=True)
    var_name = var_name_list[file_i]
    print(nc_file)
    print(var_name)
    # Create a new array to store the extracted data.
    data_ext_temp = np.zeros((len(ds.time),nisdc_lon_matrix.shape[0],nisdc_lon_matrix.shape[1]))*np.nan

    for i in range(len(ds.time)):
        print(i)
        test_ym = np.array(ds[var_name][i,:,:])
        for k in range(nisdc_lon_matrix.shape[0]):
            for t in range(nisdc_lon_matrix.shape[1]):
                data_ext_temp[i,k,t] = np.nanmean(test_ym[nisdc_disflag_matrix[k,t,:,:]>0])

    np.save('G:/CMIP6_SST_SIC/second_download/'+model_list[file_i] + var_name + '_'+ssp_list[file_i]+'_197901_210012.npy',data_ext_temp)
```

#### 9.2. Calculate trends

```python
import  pandas as pd
import scipy.stats as st
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
import seaborn as sns
import cmaps
import os

def cal_slope(x,y):
    result = st.linregress(x, y)
    slope = result.slope
    p_value = result.pvalue
    return slope, p_value

lon_aim = np.load('nisdc_lon_arr_AP.npy') 
lat_aim = np.load('nisdc_lat_arr_AP.npy') 

file1 = 'G:/CMIP6_SST_SIC/second_download/UKESM-1-0-LL_nisdc_tos_ssp585_197901_210012.npy'
prob_dat = np.load(file1)

interval = '_'
model_label = 'UKESM-1-0-LL'
var_label = 'tos'
ssp_label = 'ssp585'
month_label = '02'
time_lavel = '2015-2100'

hist_var = prob_dat[432:,:,:]
hist_var_01 = hist_var[1::12,:,:]
year = np.array(range(2015,2101))

trend_array = np.zeros((prob_dat.shape[1], prob_dat.shape[2]))*np.nan
pvalue_array = np.zeros((prob_dat.shape[1], prob_dat.shape[2]))*np.nan

for i in range(prob_dat.shape[1]):
    for j in range(prob_dat.shape[2]):
        result = cal_slope(year,hist_var_01[:,i,j])
        trend_array[i,j] = result[0]
        pvalue_array[i,j] = result[1]
        
np.save(model_label+interval+var_label+interval+month_label+interval+ssp_label+interval+time_lavel+'_trend.npy', trend_array)
np.save(model_label+interval+var_label+interval+month_label+interval+ssp_label+interval+time_lavel+'_pvalue.npy', pvalue_array)
```

#### 9.3. Reconstruct the Future Ocean Conditions Based on Trends

```python
# ----This program calculates future changes in sea ice based on pre-calculated trends of future changes.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_sic = 'G:/ecwmf_dat/nisdc_ONDJFM_siconc_196901_202112.npy'
sic_df = np.load(file_sic)
sic_df_01 = sic_df[1::6,:,:]
sic_df_01_2020 = sic_df_01[-2,:,:]

# The unit to pay attention to the trend of sea ice changes is %/year!!!
file_sictrend = './UKESM-1-0-LL/trend/UKESM-1-0-LL_siconc_02_ssp585_2015-2100_trend.npy'
sic_df_01_trend = np.load(file_sictrend)/100

sic_arr = np.zeros((80, 101, 100))*np.nan
for i in range(sic_arr.shape[0]):
    sic_arr[i,:,:] = sic_df_01_2020+(i+1)*sic_df_01_trend
    
sic_arr[sic_arr>1]=1
sic_arr[sic_arr<0]=0

sic_arr_all = np.zeros((122, 101, 100))*np.nan
sic_arr_all[0:42,:,:] = sic_df_01[10:-1,:,:]
sic_arr_all[42:,:,:] = sic_arr

interval = '_'
model_label = 'UKESM-1-0-LL'
var_label = 'sic'
ssp_label = 'ssp585'
month_label = '02'
time_lavel = '1979-2100'

np.save(model_label+interval+var_label+interval+month_label+interval+ssp_label+interval+time_lavel+'_for_predict.npy',sic_arr_all)
```

```python
# ----This program calculates future sea temperature changes based on the pre-calculated trends of future changes.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_sic = 'G:/ecwmf_dat/nisdc_ONDJFM_sst_196901_202112.npy'
sic_df = np.load(file_sic)
sic_df_01 = sic_df[1::6,:,:]
sic_df_01_2020 = sic_df_01[-2,:,:]

file_sictrend = './UKESM-1-0-LL/trend/UKESM-1-0-LL_tos_02_ssp585_2015-2100_trend.npy'
sic_df_01_trend = np.load(file_sictrend)

sic_arr = np.zeros((80, 101, 100))*np.nan

for i in range(sic_arr.shape[0]):
    sic_arr[i,:,:] = sic_df_01_2020+(i+1)*sic_df_01_trend
    
sic_arr_all = np.zeros((122, 101, 100))*np.nan
sic_arr_all[0:42,:,:] = sic_df_01[10:-1,:,:]
sic_arr_all[42:,:,:] = sic_arr

interval = '_'
model_label = 'UKESM-1-0-LL'
var_label = 'sst'
ssp_label = 'ssp585'
month_label = '02'
time_lavel = '1979-2100'

np.save(model_label+interval+var_label+interval+month_label+interval+ssp_label+interval+time_lavel+'_for_predict.npy',sic_arr_all)
```

#### 9.4. Predicting Future Foraging Suitability

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

sst_i = np.load('./UKESM-1-0-LL/for_predict/UKESM-1-0-LL_sst_12_ssp585_1979-2100_for_predict.npy')
siconc_i = np.load('./UKESM-1-0-LL/for_predict/UKESM-1-0-LL_sic_12_ssp585_1979-2100_for_predict.npy')
z_i = np.load('G:/ecwmf_dat/nisdc_ONDJFM_z_196901_202112.npy')

pro_i = np.zeros(sst_i.shape)*np.nan
pre_abs_i = np.zeros(sst_i.shape)*np.nan

index_i = np.zeros((sst_i.shape[1],sst_i.shape[2]))*np.nan
index_j = np.zeros((sst_i.shape[1],sst_i.shape[2]))*np.nan

for i in range(sst_i.shape[1]):
    index_i[i,:]  = i
    
for j in range(sst_i.shape[2]):
    index_j[:,j]  = j
    
k = 0
model_no = 'model_'+str(k+1)
rf = joblib.load('G:/ecwmf_dat/model1_100/'+model_no+'.m')

df_list = []
for t in range(sst_i.shape[0]):
    print(t)
    sst_i_df = sst_i[t,:,:].ravel()
    z_i_df = z_i[:,:].ravel()
    siconc_i_df = siconc_i[t,:,:].ravel()
    index_i_df = index_i.ravel()
    index_j_df = index_j.ravel()
    feature_df = pd.DataFrame({'sst':sst_i_df,
                              'z':z_i_df,
                              'siconc':siconc_i_df,
                              'index_i_df':index_i_df,
                              'index_j_df':index_j_df})
    feature_df = feature_df.dropna()

    y_pred_prob = rf.predict_proba(feature_df[['sst','z','siconc']])[:, 1]
    y_pred = rf.predict(feature_df[['sst','z','siconc']])
    feature_df['y_pred_prob'] = y_pred_prob
    feature_df['y_pred'] = y_pred
    df_list.append(feature_df)
    
for t_i in range(len(df_list)):
    print(t_i)
    df_temp = df_list[t_i]
    for line in range(len(df_temp)):
        index_line = int(df_temp.iloc[line,3])
        index_col = int(df_temp.iloc[line,4])
        prob_ti = df_temp.iloc[line,5]
        preb_abs_ti = df_temp.iloc[line,6]
        pro_i[t_i, index_line, index_col] = prob_ti
        pre_abs_i = preb_abs_ti
        
np.save('UKESM-1-0-LL_12_ssp585_1979-2100_for_predict_prob.npy',pro_i)
np.save('UKESM-1-0-LL_12_ssp585_1979-2100_for_predict_pre_abs.npy',pre_abs_i)
```

### 10. Calculating the southward migration of the foraging suitability

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lon_arr = np.load('G:/ecwmf_dat/future_env/nisdc_lon_arr_AP.npy')-360
lat_arr = np.load('G:/ecwmf_dat/future_env/nisdc_lat_arr_AP.npy')

flag_arr = np.zeros(lon_arr.shape)+1

flag_arr[(lon_arr<-74) | (lon_arr>-58) | (lat_arr>(0.375*(lon_arr+74)-66.5)) | (lat_arr<(0.375*(lon_arr+74)-70))] = 0

np.save('flag_arr.npy', flag_arr)
```



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file1 = './model_1_prob.npy'
file2 = './flag_arr.npy'
file3 = './nisdc_lat_arr_AP.npy'

pre_abs_dat = np.load(file1)
selected_region_matrix = np.load(file2)
lat_dat = np.load(file3)

region_prob = pre_abs_dat*selected_region_matrix
weight_lat = region_prob*lat_dat

latlist = []
for t in range(weight_lat.shape[0]):
    weight_lat_t = np.nansum(weight_lat[t,:,:])/np.nansum(region_prob[t,:,:])
    latlist.append(weight_lat_t)
    
weight_lat_df = pd.DataFrame(np.reshape(np.array(latlist), (int(weight_lat.shape[0]/6),6)))

weight_lat_df.columns = ['Jan','Feb','Mar','Oct','Nov','Dec']

weight_lat_df.index = list(range(1969,2022))
weight_lat_df.to_excel('weight_lat_df.xlsx')
```







