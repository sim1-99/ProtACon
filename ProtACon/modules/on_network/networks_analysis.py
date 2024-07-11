#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__email__ = 'renatoeliasy@gmail.com'
__author__ = 'Renato Eliasy'

import numpy as np

'''
this script analyze the amminoacids in the protein, it also enhance some selected features
through colors

'''


def collect_results_about_partitions(homogeneity: float,
                                     completeness: float
                                     ) -> tuple[float, ...]:
    '''
    starting from homogeneity and completeness it returns a triplet tuple to collect
    and compute calculation about the V-measure:

    Parameters:
    ----------
    homogeneity: float
        the homogeneity score
    completeness: float
        the completeness score

    Returns:
    -------
    partitions_results : tuple[float,...]
        the V-measure, the homogeneity and the completeness
    '''
    V_measure = (homogeneity*completeness)/np.sum(homogeneity, completeness)
    V_measure = np.round(V_measure, 2)
    partitions_results = (V_measure, homogeneity, completeness)
    return partitions_results
