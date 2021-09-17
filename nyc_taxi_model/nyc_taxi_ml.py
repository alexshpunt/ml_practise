from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from shared.data_utility import *

import pandas as pd 
import numpy as np 
from os import listdir 
from os.path import isfile, join
from matplotlib.pyplot import xlim, ylim
from scipy import stats

from pylab import * 
import random 

project_data_path = 'data/nys_taxi'
trip_data_path = f'{project_data_path}/trip_data'
trip_flare_path = f'{project_data_path}/trip_fare'

def get_all_files(path):
    return [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv')]    

trip_data_files = get_all_files(trip_data_path)[:1]
trip_flare_files = get_all_files(trip_flare_path)[:1]

nrows = 1e6
trip_data = None
for f in trip_data_files:
    print(f)
    in_df = pd.read_csv(f, nrows=nrows)
    if not trip_data: trip_data = in_df 
    else: trip_data = trip_data.merge(in_df)

fare_data = None 
for f in trip_flare_files:
    print(f)
    in_df = pd.read_csv(f, nrows=nrows)
    if not fare_data: fare_data = in_df 
    else: fare_data = fare_data.merge(in_df)

fare_cols = [
    ' payment_type', ' fare_amount', ' surcharge', 
    ' mta_tax', ' tip_amount', ' tolls_amount', ' total_amount']
data = trip_data.join(fare_data[fare_cols])
data = data.drop(['medallion', 'hack_license'], axis=1)
del trip_data 
del fare_data 

def draw_tip_to_dist():
    data.plot(x='trip_time_in_secs', y=' tip_amount', kind='scatter', s=2, figsize=(8,8))
    xlim(0,5e3)
    ylim(0, 40)

def draw_triptime_to_totalamount():
    data.plot(x='trip_time_in_secs', y=' total_amount', kind='scatter', s=2, figsize=(8,8))
    xlim(0,5e3)
    ylim(0, 120)

def filter_outliers(data, column):
    threshold = 2
    outliers = np.where(np.abs(stats.zscore(column)) > threshold)[0]
    return data.loc[~data.index.isin(outliers)]

def draw_pickups():
    data.plot(x='pickup_latitude', y='pickup_longitude', kind='scatter', s=1, figsize=(16,8))
    xlim(40.6, 40.9)
    ylim(-74.05, -73.9)

def draw_dropoffs():
    data.plot(x='dropoff_latitude', y='dropoff_longitude', kind='scatter', s=1, figsize=(16,8))
    xlim(40.6, 40.9)
    ylim(-74.05, -73.9)

def draw_tipped_area(prefix):
    figure(figsize=(16,8))
    latitude = f'{prefix}_latitude'
    longitude = f'{prefix}_longitude'
    tipped = data[data['tipped'] == True]
    not_tipped = data[data['tipped'] == False]

    plot(tipped[latitude], tipped[longitude], 'b,')
    plot(not_tipped[latitude], not_tipped[longitude], 'r,')
    xlim(40.6, 40.9)
    ylim(-74.05, -73.9)

def draw_tipped_pickup_area(): draw_tipped_area("pickup")
def draw_tipped_dropoff_area(): draw_tipped_area("dropoff")

#0 tips are around 462k entries per 1e6 nrows 
tips_with_cash = data[data[' payment_type'] == 'CSH'][' tip_amount'].value_counts() 
#because there are no records about tips amount when paid in cash 
#we need to drop this data 
data = data[data[' payment_type'] != 'CSH'] 

#feature extraction
data['tipped'] = (data[' tip_amount'] > 0).astype('int')

#All the commented out features below were not important 
# data = data.join(cat_to_num(data[' payment_type']))
# data = data.join(cat_to_num(data['vendor_id']))
# data = data.join(cat_to_num(data['rate_code']))

data['trip_time_in_secs'][data['trip_time_in_secs'] < 1e-3] = -1
data['speed'] = data['trip_distance'] / data['trip_time_in_secs']

def extract_datetime(prefix):
    dtdf = pd.to_datetime(data[f'{prefix}_datetime'])
    data[f'{prefix}_day'] = dtdf.apply(lambda d: d.dayofweek)
    # data[f'{prefix}_week'] = dtdf.apply(lambda d: d.week)
    data[f'{prefix}_hour'] = dtdf.apply(lambda d: d.hour)
extract_datetime('pickup')
extract_datetime('dropoff')

feats_to_drop = [
    'store_and_fwd_flag', ' payment_type', 'vendor_id', 
    'rate_code', 'pickup_datetime', 'dropoff_datetime', 'trip_time_in_secs']
data = data.drop(feats_to_drop, axis=1)

M = len(data)
rand_idx = arange(M)
random.shuffle(rand_idx)
train_test_split = int(M*0.2)
train_idx = rand_idx[train_test_split:]
test_idx = rand_idx[:train_test_split]

dropped_feats = [' tip_amount', 'tipped', ' total_amount']
feats = [c for c in data.head() if c not in dropped_feats]

scaler = StandardScaler() 
data_scaled = scaler.fit_transform(data[feats])

#we should do this because we have missing indices 
#due to the fact the we removed the CSH payment type rows 
train_data = data.iloc[train_idx].loc[:,feats]
target_train_data = data.iloc[train_idx]['tipped']

test_data = data.iloc[test_idx].loc[:,feats]
target_test_data = data.iloc[test_idx]['tipped']

def train_predict_measure(model):
    model.fit(train_data, target_train_data)
    preds = model.predict_proba(test_data)[:,1]
    fpr, tpr, thr = roc_curve(target_test_data, preds)
    auc = roc_auc_score(target_test_data, preds)

    plot(fpr, tpr)
    plot(fpr, fpr)
    xlabel("False positive rate")
    ylabel("True positive rate")
    print(f"{type(model)} auc is {auc}")

# sgd = SGDClassifier(loss='modified_huber')
# train_predict_measure(sgd)

rf = RandomForestClassifier(n_estimators=100, n_jobs=10)
train_predict_measure(rf)

fi = zip(feats, rf.feature_importances_)
fi = sorted(fi, key=lambda x: -x[1])
fi = pd.DataFrame(fi, columns=['feature', 'importance'])
fi