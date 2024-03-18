from math import radians
from typing import Union
import numpy as np
import pandas as pd
import h3

def calculate_euclidean_distance(lat1: Union[int, float], lon1: Union[int, float], lat2: Union[int, float], lon2: Union[int, float]) -> float:
  """ Calculates euclidean distance between 2 points (lat1, lon1) & (lat2, lon2) """
  return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

def calculate_haversine_distance(lat1: Union[int, float], lon1: Union[int, float], lat2: Union[int, float], lon2: Union[int, float]) -> float:
    """ Calculates haversine distance (distance taking into account curvature of earth) 
    between 2 points (lat1, lon1) & (lat2, lon2) """
    R = 6372.8
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    dist = R*c
    return dist 

def calculate_average_distance_by_courier(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby('courier_id').agg({'dist_to_restaurant': 'mean', 'Hdist_to_restaurant': 'mean'})

def drop_nulls(X: pd.DataFrame) -> pd.DataFrame:
   return X.dropna()

def formatting_datetime_values(X: pd.DataFrame) -> pd.DataFrame:
    X['courier_location_timestamp']=  pd.to_datetime(X['courier_location_timestamp'])
    X['order_created_timestamp'] = pd.to_datetime(X['order_created_timestamp'])
    X['date_day_number'] = [d for d in X.courier_location_timestamp.dt.day_of_year]
    X['date_hour_number'] = [d for d in X.courier_location_timestamp.dt.hour]
    return X

def formatting_spatial_values(X: pd.DataFrame, resolution: int) -> pd.DataFrame:
    X['h3_index'] = [h3.geo_to_h3(lat, lon, resolution) for (lat, lon) in zip(X.courier_lat, X.courier_lon)]
    return X

# # #################################################

# def create_restaurant_ids(df):
#    df['restaurant_loc'] = df['restaurant_lat'].astype(str) + '_' + df['restaurant_lon'].astype(str)
#    encoder = LabelEncoder()
#    df['restaurant_id'] = encoder.fit_transform(df['restaurant_loc'])
#    return df, encoder

# def calc_dist(p1x, p1y, p2x, p2y):
#   p1 = (p2x - p1x)**2
#   p2 = (p2y - p1y)**2
#   dist = np.sqrt(p1 + p2)
#   return dist.tolist() if isinstance(p1x, collections.abc.Sequence) else dist

# def avg_dist_to_restaurants(courier_lat,courier_lon, restaurant_ids):
#   return np.mean([calc_dist(v['lat'], v['lon'], courier_lat, courier_lon) for v in restaurants_ids.values()])

# def calc_haversine_dist(lat1, lon1, lat2, lon2):
#   R = 6372.8    #3959.87433  this is in miles.  For Earth radius in kilometers use 6372.8 km
#   if isinstance(lat1, collections.abc.Sequence):
#     dLat = np.array([radians(l2 - l1) for l2,l1 in zip(lat2, lat1)])
#     dLon = np.array([radians(l2 - l1) for l2,l1 in zip(lon2, lon1)])
#     lat1 = np.array([radians(l) for l in lat1])
#     lat2 = np.array([radians(l) for l in lat2])
#   else:
#     dLat = radians(lat2 - lat1)
#     dLon = radians(lon2 - lon1)
#     lat1 = radians(lat1)
#     lat2 = radians(lat2)
#     a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
#     c = 2*np.arcsin(np.sqrt(a))
#     dist = R*c
#     return dist.tolist() if isinstance(lon1, collections.abc.Sequence) else dist

# def initiate_centroids(k, df):
#     '''
#     Select k data points as centroids
#     k: number of centroids
#     dset: pandas dataframe
#     '''
#     centroids = df.sample(k)
#     return centroids

# def centroid_assignation(df, centroids):
#   k = len(centroids)
#   n = len(df)
#   assignation = []
#   assign_errors = []
#   centroids_list = [c for i,c in centroids.iterrows()]
#   for i,obs in df.iterrows():
#     # Estimate error
#     all_errors = [eucl_dist( centroid['lat'],
#                             centroid['lon'],
#                             obs['courier_lat'],
#                             obs['courier_lon']) for centroid in centroids_list]

#     # Get the nearest centroid and the error
#     nearest_centroid =  np.where(all_errors==np.min(all_errors))[0].tolist()[0]
#     nearest_centroid_error = np.min(all_errors)

#     # Add values to corresponding lists
#     assignation.append(nearest_centroid)
#     assign_errors.append(nearest_centroid_error)

# def Encoder(df):
#   columnsToEncode = list(df.select_dtypes(include=['category','object']))
#   le = LabelEncoder()
#   for feature in columnsToEncode:
#       try:
#           df[feature] = le.fit_transform(df[feature])
#       except:
#           print('Error encoding '+feature)
#   return df