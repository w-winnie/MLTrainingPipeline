# This leverages sklearn pipeline, the idea is to fit the pipeline to the training data
# Then be able to save it and apply it on test or daily incoming unseen data

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from modules.ml.preparation.helper import *
import numpy as np
import pandas as pd
# from sklearn.cluster import KMeans
    
class DistanceTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.restaurant_encoder = LabelEncoder().fit(X[['restaurant_lat', 'restaurant_lon']].astype(str).agg('_'.join, axis=1))
        return self
    
    def transform(self, X, y=None):
        X['restaurant_id'] = self.restaurant_encoder.transform(X[['restaurant_lat', 'restaurant_lon']].astype(str).agg('_'.join, axis=1))
        X['dist_to_restaurant'] = X.apply(lambda row: calculate_euclidean_distance(row['courier_lat'], row['courier_lon'], row['restaurant_lat'], row['restaurant_lon']), axis=1)
        X['Hdist_to_restaurant'] = X.apply(lambda row: calculate_haversine_distance(row['courier_lat'], row['courier_lon'], row['restaurant_lat'], row['restaurant_lon']), axis=1)
        return X 

class DropNullsTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.dropna()
    
class AverageCalculatorTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # averages = X.groupby('courier_id','restaurant_id').agg({'euclidean_distance': 'mean', 'haversine_distance': 'mean'})
        # X_merged = pd.merge(X, averages, on='courier_id', suffixes=('', '_avg'))
        distances = []
        hdistances = []
        courier_ids = X['courier_id'].unique()
        for courier_id in courier_ids:
            courier_data = X[X['courier_id'] == courier_id]
            courier_lat = courier_data['courier_lat'].iloc[0]
            courier_lon = courier_data['courier_lon'].iloc[0]
            
            courier_distances = [calculate_euclidean_distance(courier_lat, courier_lon, restaurant_lat, restaurant_lon) 
                                 for restaurant_lat, restaurant_lon in zip(X['restaurant_lat'], X['restaurant_lon'])]
            distances.append(np.mean(courier_distances))

            courier_Hdistances = [calculate_haversine_distance(courier_lat, courier_lon, restaurant_lat, restaurant_lon)
                                 for restaurant_lat, restaurant_lon in zip(X['restaurant_lat'], X['restaurant_lon'])]
            hdistances.append(np.mean(courier_Hdistances))
        
        avg_distances_df = pd.DataFrame({'courier_id': courier_ids, 'avg_dist_to_restaurants': distances, 'avg_Hdist_to_restaurants': distances})
        X = pd.merge(X, avg_distances_df, on='courier_id', how='left')
        return X
    
# class KMeansTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, k):
#         self.k = k
    
#     def fit(self, X, y=None):
#         self.kmeans = KMeans(n_clusters=self.k, random_state=1)
#         self.kmeans.fit(X)
#         return self
    
#     def transform(self, X):
#         X['Five_Clusters_embedding'] = self.kmeans.predict(X)
#         X['Five_Clusters_embedding_error'] = self.kmeans.labels_
#         return X
    
class CentroidAssignmentTransform(BaseEstimator, TransformerMixin):
    def __init__(self,k):
        self.k = k
        # pass

    def fit(self, X, y=None):
        X_restaurants = X[['restaurant_lat', 'restaurant_lon']].drop_duplicates()
        self.centroids = X_restaurants.sample(self.k)
        return self

    def transform(self, X):
        assignation = []
        assign_errors = []
        centroids_list = [c for i, c in self.centroids.iterrows()]

        for i, obs in X.iterrows():
            all_errors = [calculate_euclidean_distance(centroid['restaurant_lat'], centroid['restaurant_lon'],
                                         obs['courier_lat'], obs['courier_lon']) for centroid in centroids_list]

            nearest_centroid = np.argmin(all_errors)
            nearest_centroid_error = np.min(all_errors)

            assignation.append(nearest_centroid)
            assign_errors.append(nearest_centroid_error)

        k = len(centroids_list)
        X['Five_Clusters_embedding'] = assignation
        X['Five_Clusters_embedding_error'] = assign_errors
        # X[f'Clusters_embedding_{k}'] = assignation
        # X[f'Clusters_embedding_error_{k}'] = assign_errors
        return X

class OrderBusynessTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.index_list = [(i, d, hr) for (i, d, hr) in zip(X['h3_index'], X['date_day_number'], X['date_hour_number'])]
        self.set_indexes = list(set(self.index_list))
        self.dict_indexes = {label: self.index_list.count(label) for label in self.set_indexes}
        self.restaurants_counts_per_h3_index = {a: len(b) for a, b in zip(X.groupby('h3_index')['restaurant_id'].unique().index, X.groupby('h3_index')['restaurant_id'].unique())}
        return self
    
    def transform(self, X, y=None):
        X['orders_busyness_by_h3_hour'] = [self.dict_indexes[i] for i in self.index_list]
        X['restaurants_per_index'] = [self.restaurants_counts_per_h3_index[h] for h in X['h3_index']]
        return X

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_encode=[]):
        self.columns_to_encode = columns_to_encode
        self.label_encoders_ = {}

    def fit(self, X, y=None):
        for col in self.columns_to_encode:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders_[col] = le
        return self

    def transform(self, X, y=None):
        for col, le in self.label_encoders_.items():
            X[col] = le.transform(X[col])
        return X

data_prep_pipeline = Pipeline([
    # ('remove_null_rows', FunctionTransformer(drop_nulls))  
    ('remove_null_rows', DropNullsTransform()),
    # ('restaurant_encoder', CustomLabelEncoder(['restaurant_loc'])),
    ('calculate_distances', DistanceTransform()), 
    # ('group_by_courier', FunctionTransformer(calculate_average_distance_by_courier))   
    ('group_by_courier', AverageCalculatorTransform()),
    ('assign_centroid', CentroidAssignmentTransform(k=5)),
    ('handle_datetime_values', FunctionTransformer(formatting_datetime_values)),
    ('handle_spatial_values', FunctionTransformer(formatting_spatial_values, kw_args={'resolution': 7})),
    ('order_busyness', OrderBusynessTransform()),
    ('h3_encoder', CustomLabelEncoder(['h3_index']))
])