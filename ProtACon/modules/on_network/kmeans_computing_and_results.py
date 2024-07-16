#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__email__ = 'renatoeliasy@gmail.com'
__author__ = 'Renato Eliasy'

import pandas as pd
import numpy as np

import logging
import sklearn.cluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_clusters_label(
        dataset: pd.DataFrame,  # dataset where to perform the analysis
        cluster_feature: pd.core.series.Series | int = 4,  # the feature to cluster
        cluster_method: str = 'kmeans++',  # as default kmeans++
        # option to scale data based on the method ['std', 'minMAX', others...]
        scaler_option: str = None,
        scaled_data_return: bool = False
) -> tuple[tuple[list[str], ...], pd.DataFrame]:  # returned type data
    '''
    the function has the purpouse to apply a certain cluster to a dataframe and obtain the labels
    -----------

    Parameters:
    ------------
    cluster_method : sklearn.cluster
        the cluster method to apply, as default it will be kmeans++
    dataset : pd.DataFrame
        a DataFrame to start with, assured to be all of number data, 
        set the label of clusters as index column, with set_index method
        e.g. in the dataframe with AA_pos : C(0), P(1), K(2), L(3)...
        the clusters will be in the format ( [C(0), K(2), ...], [P(1), ... ], [L(3), ...])
    cluster_feature : pd.core.series.Series
        the feature to clusterize, it has to be a column of the dataset, otherwise it can produce a warning; for example if i want to find matches of i-feature i gonna apply the kmeans to the n-i.th features
    scaler_option : str 
        the option to scale the data, as default is None, other options are ['std', 'minMAX']
    scaled_data_return : bool
        if True, return the scaled data, else the original data
    Returns:
    ----------
    tuple(): external structure to organize data
    lists: clustee_label1, cluster_label2, .... as many as the number of different type of cluster in the cluster feature
    '''
    feature_in_this_dataset = True
    cluster_method = sklearn.cluster.kmeans_plusplus

    # FIXME check sulle feature

    if (type(cluster_feature) != int) and not (cluster_feature.name in dataset.columns):
        logging.warning('the feature {0} is not in the {1}, it can produce some unexpected result if the feature proposed is not linked to this DataFrame'.format(
            cluster_feature.name, dataset))
        feature_in_this_dataset = False

    # check sul metodo di clustering

        # raise AttributeError('please check the method of clustering to apply')

    # measure the number of element in the cluster_feature, depending its type
    if type(cluster_feature) == int:
        n_clusters = cluster_feature
    else:
        n_clusters = len(set(cluster_feature.values))

    # remove the column from dataframe to get as a trigger of the kmeans
    if feature_in_this_dataset:
        dataset = dataset.drop(columns={cluster_feature.name})

    # check sul dataframe, non deve contenere valori tipo stringa, dopo la rimozione della colonna da cui prendere i cluster, nel caso essi siano in forma di stringa:
    for row in dataset.values:
        for element in row:
            if type(element) == str:
                logging.error(
                    'the {0} contain str-type values'.format(dataset))
        # raise TypeError('please handle the DataFrame to have only float or int type of data')

    # as default
    scaled_df = dataset.values
    # scale data?
    if scaler_option == 'std':
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(dataset)

    elif scaler_option == 'minMAX':
        scaler = MinMaxScaler()
        scaled_df = scaler.fit_transform(dataset)

    if scaled_data_return:
        data = scaled_df
    elif not scaled_data_return:
        data = dataset.values

    new_dataset = pd.DataFrame(
        data, columns=dataset.columns, index=dataset.index)

    # kmeans method initialization
    cluster_method = KMeans(
        init='k-means++', n_clusters=n_clusters, n_init='auto')
    # fit data on kmeans and get labels
    cluster_method.fit(scaled_df)

    label_groups = []
    for _ in range(n_clusters):
        label_groups.append([])
    # put the labels information in the dataframe col named cluster_group
    new_dataset['cluster_group'] = cluster_method.labels_

    for index, row in new_dataset.iterrows():
        label_groups[int(row['cluster_group'])].append(index)

    combined_results = (tuple(label_groups), new_dataset)

    return combined_results


def dictionary_from_tuple(list_of_labels: tuple[list[str], ...]
                          ) -> dict:
    """
    it get the dictionary of elements appartaining to different labels
    this function is performed specifically to work with the get_clusters_label funciton in the same module
    to have the label and the node expressed in a better way, and easier to access to in some point of view:
    Parameters:
    ----------
    list_of_labels: tuple[list[str], ...]
        the tuple of lists containing the labels of the clusters

    Returns:    
    -------
    dict_labels: dict
        the dictionary of the labels with the elements in the list
    """
    label_dict = {}
    for label_index, listed in enumerate(list_of_labels):
        for aa in listed:
            label_dict[aa] = label_index + 1
    return label_dict
