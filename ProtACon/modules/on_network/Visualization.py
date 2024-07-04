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


def plot_pca_2d(

) -> None:
    """
    it plot a scatter plot using the first 2 PCAs components as axis of reference
    """
    pass


def plot_pca_3d(

) -> None:
    """
    it plot a scatter plot using the first 3 PCAs components as axis of reference
    """
    pass
