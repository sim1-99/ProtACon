#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__email__ = 'renatoeliasy@gmail.com'
__author__ = 'Renato Eliasy'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PCA_computing_and_results as PCA_results


def plot_histogram_pca(percentage_var: tuple[float, ...],
                       best_features: tuple[str, ...],
                       protein_name: str
                       ) -> None:
    """
    This function generate an histogram whose 
    x-axis is the PCAs components and y-axis is the percentage of variations
    since it require a lot of space there will be a legend with the top 3 components best corresponding

    Parameters:
    ----------
    percentage_var: tuple[float,...]
        the percentage variations as results of explained_variance_ratio method
    best_features: tuple[str,...]
        the most compatible feature for each of the PCAs
    protein_name: str
        the name of the protein whose histogram is computed on

    Returns:
    -------
    None, but plot a figure

    """

    labels = ['PC' + str(i) for i in range(1, len(percentage_var)+1)]
    plt.bar(x=range(1, len(percentage_var)+1),
            height=percentage_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Components')
    plt.title('PCA components of {0}'.format(protein_name))
    plt.legend(['PC1-> {0}\nPC2-> {1}\nPC3-> {2}'.format(best_features[0],
               best_features[1], best_features[2])])
    plt.show()
    return None


def plot_pca_2d(pca_dataframe: pd.DataFrame,  # dataframe from which take the components
                protein_name: str,  # the name of the protein whose plot refers to
                # the features to show on the plot corresponting to the PC1 and PC2 most compatible components
                best_features: tuple[str, ...],
                # the amount of compatibility of the feature and the component
                percentage_var: tuple[float, ...],
                color_map: pd.Series = False
                ) -> None:
    """
    it plot a scatter plot using the first 2 PCAs components as axis of reference

    Parameters:
    ----------
    pca_dataframe: pd.DataFrame
        the dataframe of the PCAs
    protein_name: str
        the name of the protein whose plot refers to
    best_features: tuple[str,...]
        the features to show on the plot corresponting to the PC1 and PC2 most compatible components
    percentage_var: tuple[float, ...]
        the percentage variations as results of explained_variance_ratio method
    color_map: pd.Series
        the color map to be used for the scatter plot to cluster the points, as default is false
    Returns:
    -------
    None, but plot a scatter plot 2d

    """
    labels = ['PC' + str(i) for i in range(1, len(percentage_var)+1)]
    for label in labels:
        if label not in pca_dataframe.columns:
            raise ValueError(
                'The dataframe must have the same columns as the labels of PC1, PC2...')
    x_values = pca_dataframe.PC1
    y_values = pca_dataframe.PC2

    fig, ax = plt.subplot(figsize=(10, 8))
    if not color_map:
        scatter = ax.scatter(x_values, y_values, color='blue')

    else:
        scatter = ax.scatter(x_values, y_values, c=color_map, cmap='viridis')
        if color_map == x_values:
            cbar = plt.colorbar(scatter, location='bottom')
        elif color_map == y_values:
            cbar = plt.colorbar(scatter, location='left')

    cbar.set_label('{0}'.format(str(color_map)))

    plt.title('PCAs Scatter Plot of {0}'.format(protein_name))
    plt.xlabel('PC1-> {0} : {1}'.format(best_features[0], percentage_var[0]))
    plt.ylabel('PC2-> {0} : {1}'.format(best_features[1], percentage_var[1]))
    plt.show()
    return None


def plot_pca_3d(pca_dataframe: pd.DataFrame,  # dataframe from which take the components
                protein_name: str,  # the name of the protein whose plot refers to
                # the features to show on the plot corresponting to the PC1 and PC2 most compatible components
                best_features: tuple[str, ...],
                # the amount of compatibility of the feature and the component
                percentage_var: tuple[float, ...],
                color_map: pd.Series = False
                ) -> None:
    """
    it plot a scatter plot using the first 2 PCAs components as axis of reference

    Parameters:
    ----------
    pca_dataframe: pd.DataFrame
        the dataframe of the PCAs
    protein_name: str
        the name of the protein whose plot refers to
    best_features: tuple[str,...]
        the features to show on the plot corresponting to the PC1 and PC2 most compatible components
    percentage_var: tuple[float, ...]
        the percentage variations as results of explained_variance_ratio method
    color_map: pd.Series
        the color map to be used for the scatter plot to cluster the points, as default is false
    Returns:
    -------
    None, but plot a scatter plot 3D
    """
    labels = ['PC' + str(i) for i in range(1, len(percentage_var)+1)]
    for label in labels:
        if label not in pca_dataframe.columns:
            raise ValueError(
                'The dataframe must have the same columns as the labels of PC1, PC2...')
    x_values = pca_dataframe.PC1
    y_values = pca_dataframe.PC2
    z_values = pca_dataframe.PC3
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    if not color_map:
        scatter = ax.scatter(x_values, y_values, z_values, color='blue')
    else:
        scatter = ax.scatter(x_values, y_values, z_values,
                             c=color_map, cmap='viridis')

    plt.title(f'PCA 3D-Scatter Plot of {protein_name} protein')
    ax.set_xlabel('{0} -{1}%'.format(best_features[0], percentage_var[0]))
    ax.set_ylabel('{0} -{1}%'.format(best_features[1], percentage_var[1]))
    ax.set_zlabel('{0} -{1}%'.format(best_features[2], percentage_var[2]))
    plt.show()
    return None
