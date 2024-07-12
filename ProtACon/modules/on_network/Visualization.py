#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__email__ = 'renatoeliasy@gmail.com'
__author__ = 'Renato Eliasy'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PCA_computing_and_results as PCA_results

import plotly.graph_objects as go
import igraph as ig
from miscellaneous import assign_color_to


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

# Following the spatial visualization of the protein in the space,
# with feature enhanced by color


def plot_protein_chain_3D(feature_dataframe: pd.DataFrame,
                          edge_list1: list[tuple[int, int]],
                          edge_list2: list[tuple[int, int]],
                          color_map: str = None,
                          edge_list3: list = [],
                          protein_name: str = None
                          ) -> None:
    """
    it works with a dataframe and one or more list of edge link
    NOTE that the elements in the link in lists must have the same 
    notation to access in the dataframe

    Parameters:
    ----------
    feature_dataframe: pd.DataFrame
        the dataframe of the features
    FIXME add an option to get edge list of index by edge_list of string type named AAs
    edge_list1: list
        the first list of edges, consider mandatory it has to be the index of nodes
    edge_list2: list
        the second list of edges
    color_map: str
        the color map to be used for the scatter plot to cluster the points, as default is None
    edge_list3: list
        the third list of edges
    protein_name: str
        the name of the protein whose plot refers to
    Returns:
    -------
    None but it plot a graph
    """

    feature_to_be_in_df = ['AA_pos',]
    for feature in feature_to_be_in_df:
        if feature_dataframe.columns.str.contains(feature):
            raise ValueError(
                'the dataframe do not contain\nthe necessary features to plot this graph')
    if not feature_dataframe.columns.str.contains(color_map):
        print('unable to find the feature selected in color_map between the dataframe features\nthe color is set to be the same')
        color_map = 'blue'

    separated_components = False
    singular_components = ('X', 'Y', 'Z')
    for coord in singular_components:
        if any(feature_dataframe.columns.str.upper().contains(coord)):
            list_of_cols = []
            separated_components = True
            for k in separated_components:
                for el in feature_dataframe.columns.str.upper().contains(k)*feature_dataframe.columns:
                    if el:
                        list_of_cols.append(str(el))

            if len(list_of_cols) != 3:
                raise ValueError(
                    'avoid repetition of columns name containing XYZ')
        break

    df_feature_dict = feature_dataframe.to_dict(orient='records')

    nodes = [AA_dict for AA_dict in df_feature_dict]

    N = len(nodes)
    trace_1edges = edge_list1
    trace_2edges = edge_list2
    if len(edge_list3):
        trace_3edges = edge_list3
        Xe3, Ye3, Ze3 = [], [], []

    G = ig.Graph(trace_1edges, directed=False)
    # labels stay for the name of the node
    labels = [node['AA_pos'] for node in nodes]
    if color_map == 'blue':
        node_color = ['blue' for _ in range(len(nodes))]
    else:
        list_of_items = [element for element in feature_dataframe.color_map]
        color_dict = assign_color_to(discrete_list_of=list_of_items)
        node_color = [color_dict[el] for el in list_of_items]

    if separated_components:
        Xn = [AA[list_of_cols[0]] for AA in df_feature_dict]
        Yn = [AA[list_of_cols[1]] for AA in df_feature_dict]
        Zn = [AA[list_of_cols[2]] for AA in df_feature_dict]
        edge_layout = zip(Xn, Yn, Zn)
    else:
        edge_layout = [AA['AA_Coords'] for AA in df_feature_dict]
        Xn = [edge_layout[k][0] for k in range(N)]
        Yn = [edge_layout[k][1] for k in range(N)]
        Zn = [edge_layout[k][2] for k in range(N)]

    Xe1, Ye1, Ze1 = [], [], []
    Xe2, Ye2, Ze2 = [], [], []

    for edge in trace_1edges:
        Xe1 += [edge_layout[edge[0]][0], edge_layout[edge[1]][0], None]
        Ye1 += [edge_layout[edge[0]][1], edge_layout[edge[1]][1], None]
        Ze1 += [edge_layout[edge[0]][2], edge_layout[edge[1]][2], None]

    for edge in trace_2edges:
        Xe2 += [edge_layout[edge[0]][0], edge_layout[edge[1]][0], None]
        Ye2 += [edge_layout[edge[0]][1], edge_layout[edge[1]][1], None]
        Ze2 += [edge_layout[edge[0]][2], edge_layout[edge[1]][2], None]

    if len(edge_list3):
        for edge in trace_3edges:
            Xe3 += [edge_layout[edge[0]][0], edge_layout[edge[1]][0], None]
            Ye3 += [edge_layout[edge[0]][1], edge_layout[edge[1]][1], None]
            Ze3 += [edge_layout[edge[0]][2], edge_layout[edge[1]][2], None]

    trace1 = go.Scatter3d(
        x=Xe1,
        y=Ye1,
        z=Ze1,
        mode='lines',
        line=dict(color='red', width=5),
        hoverinfo='none'
    )

    trace2 = go.Scatter3d(
        x=Xe2,
        y=Ye2,
        z=Ze2,
        mode='lines',
        line=dict(color='blue', width=5),
        hoverinfo='none'
    )

    if len(edge_list3):
        trace3 = go.Scatter3d(
            x=Xe3,
            y=Ye3,
            z=Ze3,
            mode='lines',
            line=dict(color='green', width=5),
            hoverinfo='none'
        )
    else:
        trace3 = None

    trace4 = go.Scatter3d(
        x=Xn,
        y=Yn,
        z=Zn,
        mode='markers',
        name='AmminoAcids',
        marker=dict(symbol='circle',
                    size=6,
                    color=node_color,
                    colorscale='Viridis',
                    line=dict(color='rgb(50,50,50)', width=0.5)
                    ),
        text=labels,
        hoverinfo='text'
    )

    layout = go.Layout(
        title=f"Protein: {protein_name}, color: {color_map}",
        width=1000,
        height=1000,
        showlegend=False,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, title=''),
        ),
        margin=dict(t=100),
        hovermode='closest',
        annotations=[
            dict(
                showarrow=False,
                text="author: {__author__}",
                xref='paper',
                yref='paper',
                x=0,
                y=0.1,
                xanchor='left',
                yanchor='bottom',
                font=dict(size=14)
            )
        ]

    )
    data = [trace1, trace2, trace3, trace4]  # FIXME if trace3
    fig = go.Figure(data=data, layout=layout)
    fig.show()
    return None
