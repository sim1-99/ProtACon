#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__email__ = 'renatoeliasy@gmail.com'
__author__ = 'Renato Eliasy'

'''
the main purpouse of this script is to compute the pca on a dataframe and
build another dataframe with the PCAs component, then collect the results in a tuple
composed by the dataframe of the PCAs and the most compatible feature for each of the PCAs
'''

from ProtACon.modules.basics import CA_Atom
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


features_dataframe_columns = ('AA_Name', 'AA_Coords', 'AA_Hydropathy', 'AA_Volume', 'AA_Charge_Density', 'AA_RCharge_density',
                              'AA_Charge', 'AA_PH', 'AA_iso_PH', 'AA_Hydrophilicity', 'AA_Surface_accessibility',
                              'AA_ja_transfer_energy_scale', 'AA_self_Flex', 'AA_local_flexibility', 'AA_secondary_structure',
                              'AA_aromaticity', 'AA_human_essentiality', 'AA_web_group')


def main(df_prepared_for_pca: pd.DataFrame
         ) -> tuple[pd.DataFrame, tuple[str, ...], tuple[int, ...]]:
    '''
    It generate a dataframe with sample the same of the sample of the df_prepared_for_pca dataframe
    As the main function of this script it give all the necessary to visualize the results in a graph
    which include:
    - the dataframe of the PCAs
    - the most compatible feature for each of the PCAs
    - the percentage variations as results of explained_variance_ratio method

    Parameters:
    ----------
    df_prepared_for_pca: pd.DataFrame
        the dataframe to be used for the PCA
    n_component: int
        the number of components to be displayed from the PCA

    Returns:
    -------
    tuple[pd.DataFrame, tuple[str, ...], tuple[int, ...]]
        the dataframe of the PCAs,
        the most compatible feature for each of the PCAs,
        the percentage variations as results of explained_variance_ratio method
    '''

    # FIXME a control over the df_prepared_for_pca TO LIMIT THE CONTROL OF DTYPES float or int in content not in index and columns
    if not (all(isinstance(x, int) for x in df_prepared_for_pca.values.flatten()) or all(isinstance(x, float) for x in df_prepared_for_pca.values.flatten())):
        raise ValueError(
            'The dataframe must have only float or int type data\nCheck ')

    scaled_data = StandardScaler().fit_transform(df_prepared_for_pca)
    pca = PCA()  # NOTE you can fix the n_component to 2 or more however this is not necessary
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)

    percentage_variations = np.round(
        pca.explained_variance_ratio_*100, decimals=1)

    most_compatible_components = []
    for array in pca.components_:
        most_compatible_components.append(
            df_prepared_for_pca.columns[np.argmax(array)])

    pca_component_labels = [
        'PC' + str(i) for i in range(1, len(percentage_variations) + 1)]
    pca_dataframe = pd.DataFrame(
        pca_data, index=df_prepared_for_pca.index, columns=pca_component_labels)

    return (pca_dataframe, most_compatible_components, percentage_variations)
