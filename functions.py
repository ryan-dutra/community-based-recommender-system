import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import pairwise_distances


def split_in_k_folds(X: pd.DataFrame, y: pd.DataFrame, k: int, test_ratio: float):   # TO DO: TROCAR PELO K-FOLD DO SKLEARN (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html) // (https://surprise.readthedocs.io/en/stable/model_selection.html#surprise.model_selection.split.RepeatedKFold)
    """
    This function splits the original  
    dataset into k different folds
    """
    data = {}
    for i in range(k):
      X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                          test_size=test_ratio, random_state=i)
      data.update({f'X_train{i}': X_train, f'X_test{i}': X_test, f'y_train{i}': y_train, f'y_test{i}': y_test, })
    return data

def get_number_of_keys(dictionary: dict):
  counter = 0
  for key in dictionary:
      if 'X_train' in key:
          counter = counter + 1
  return counter


def get_user_item_matrices(data: dict):
  matrices={}
  for i in range(get_number_of_keys(data)):
    df = data[f'X_train{i}'].copy()
    df['rating'] = data[f'y_train{i}'].copy()
    ratings_matrix = df.pivot_table(index=['user_id'],columns=['item_id'],values='rating')
    ratings_matrix.fillna(0, inplace = True)
    matrices[f'Pair{i}'] =  ratings_matrix
  return matrices

def get_adjacency_matrices(matrices_dict: dict, metric_dist: str, threshold: int):
  data={}
  for i in range(len(matrices_dict.items())):
    similarity_matrix = 1 - pairwise_distances(matrices_dict[f'Pair{i}'].to_numpy(), metric=metric_dist)
    np.fill_diagonal(similarity_matrix, 0)

    adjacency_matrix = similarity_matrix.copy()
    adjacency_matrix[similarity_matrix > np.percentile(similarity_matrix, threshold)] = 1
    adjacency_matrix[similarity_matrix <= np.percentile(similarity_matrix, threshold)] = 0

    data[f'AM{i}'] = adjacency_matrix
  return data