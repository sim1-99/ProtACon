#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__email__ = 'renatoeliasy@gmail.com'
__author__ = 'Renato Eliasy'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ProtACon.modules.on_network import PCA_computing_and_results as PCA_results

import plotly.graph_objects as go
import igraph as ig
from ProtACon.modules.miscellaneous import assign_color_to, get_AA_features_dataframe, CA_Atom
from Collect_and_structure_data import get_indices_from_str
import networkx as nx
from typing import Mapping
import logging
from pathlib import Path


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


def plot_protein_chain_3D(CA_Atoms: tuple[CA_Atom, ...],
                          edge_list1: list[tuple[int, int]] | list[tuple[str, str]],
                          edge_list2: list[tuple[int, int]] | list[tuple[str, str]],
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
        the dataframe of the features:
        - AA_Name
        - AA_pos
        - AA_iso_PH
        - AA_PH
        - AA_Coords or AA_Xcoords, AA_Ycoords, AA_Zcoords, 
        - AA_Charge
        - AA_Hydrophobicity
        - AA_Hydrophilicity
        - AA_Volume
        - AA_self_Flexibility
        - AA_JA_in->out_E.transfer
        - AA_EM_surface.accessibility
        - AA_local_flexibility
        - aromaticity
        - secondary_structure
        - vitality

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

    # NOTE complete to the necessary features
    feature_dataframe = get_AA_features_dataframe(CA_Atoms=CA_Atoms)
    feature_to_be_in_df = ['AA_pos', 'AA_Coords']
    for feature in feature_to_be_in_df:
        if feature_dataframe.columns.str.contains(feature):
            raise ValueError(
                'the dataframe do not contain\nthe necessary features to plot this graph')
    # in case the color_mapping is not defined in the dataframe
    if not feature_dataframe.columns.str.contains(color_map):
        if isinstance(color_map, dict):
            pass  # NOTE ADD CODE HERE
        else:
            logging.warning(
                'unable to find the feature selected in color_map between the dataframe features\nthe color is set to be the same')
            color_map = 'blue'

    # FIXME TO BE REMOVED SINCE THE DATAFRAME IS GIVEN
    separated_components = False
    singular_components = ('XCOORD', 'YCOORD', 'ZCOORD')
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

    # trasform the dataframe in a dictionary of records, easier to access to
    df_feature_dict = feature_dataframe.to_dict(orient='records')

    nodes = [AA_dict for AA_dict in df_feature_dict]

    N = len(nodes)
    # control if edge lists are index or str: FOR EDGE_LIST1
    trace_1edges = edge_list1
    edge_in_str = False
    for source, target in edge_list1:
        if type(source) != type(target):
            raise ValueError(
                'the edge list must have the same type of elements')
        else:
            if isinstance(source, str):
                edge_in_str = True
    if edge_in_str:
        trace_1edges = get_indices_from_str(list=edge_list1,
                                            dataframe_x_conversion=feature_dataframe,
                                            column_containing_key='AA_pos')
    # for EDGE_LIST2
    trace_2edges = edge_list2
    edge_in_str = False
    for source, target in edge_list2:
        if type(source) != type(target):
            raise ValueError(
                'the edge list must have the same type of elements')
        else:
            if isinstance(source, str):
                edge_in_str = True
    if edge_in_str:
        trace_2edges = get_indices_from_str(list=edge_list2,
                                            dataframe_x_conversion=feature_dataframe,
                                            column_containing_key='AA_pos')
    # for EDGE LIST 3
    if len(edge_list3):
        trace_3edges = edge_list3
        edge_in_str = False
        for source, target in edge_list3:
            if type(source) != type(target):
                raise ValueError(
                    'the edge list must have the same type of elements')
            else:
                if isinstance(source, str):
                    edge_in_str = True
        if edge_in_str:
            trace_3edges = get_indices_from_str(list=edge_list3,
                                                dataframe_x_conversion=feature_dataframe,
                                                column_containing_key='AA_pos')

    G = ig.Graph(trace_1edges, directed=False)

    # labels stay for the name of the node
    labels = [node['AA_pos'] for node in nodes]

    if color_map == 'blue':
        node_color = ['blue' for _ in range(N)]
    else:
        list_of_items = [element for element in feature_dataframe.color_map]
        color_dict = assign_color_to(discrete_list_of=list_of_items)
        node_color = [color_dict[el] for el in list_of_items]

    if separated_components:
        # fix coords of nodes
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
            Xe3, Ye3, Ze3 = [], [], []
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
    if trace3:
        data = [trace1, trace2, trace3, trace4]
    else:
        data = [trace1, trace2, trace4]

    fig = go.Figure(data=data, layout=layout)
    fig.show()
    return None

# NOTE better to use it directly in the main as see results of...


def network_layouts(network_graph: nx.Graph,
                    # the attribute of the nodes to map the color
                    node_layout: tuple[str, ...],
                    edge_layout: tuple[str, ...],
                    clusters_color_group: dict = False,  # use the dict to get the map of colors
                    label: tuple[str, int] = False
                    ) -> dict:
    """
    a function to draw a network graph, as a base see draw_networkx; draw_netwokx_nodes; draw_networx_edges
    Parameters:
    ----------
    network_graph: nx.Graph
        the network graph to be drawn, it contains the following attributes,at least for nodes and edges:
        - NODES: 'AA_Name', 'AA_Coords', 'AA_Hydropathy', 'AA_Volume', 'AA_Charge', 'AA_PH', 'AA_iso_PH', 'AA_Hydrophilicity', 'AA_Surface_accessibility',
                        'AA_ja_transfer_energy_scale', 'AA_self_Flex', 'AA_local_flexibility', 'AA_secondary_structure', 'AA_aromaticity', 'AA_human_essentiality'

        - EDGES: 'lenght', 'stability', 'contact_in_sequence'
    node_colors: tuple[str, ...]
        the attribute of the nodes to map: ['color', 'size',... ]
    edge_layout : tuple[str, ...]
        the attribute of the edges to map: ['color', 'style',...]
    clusters_color_group: dict
        the dictionary of the color mapping of the nodes it hase to be the format: {'C(0)' : 1 , 'P(1)' : 2 ....} 
        it could be the kmeans cluster or the louvain partitions, or any other dict of kind
    label : tuple[str, ...]
        the label of the nodes in graph
        if True: font_weight = 'bold' | 
                 font_size = int()
        if False: no label

    """
    list_of_nodes_attributes = list(
        network_graph.nodes(data=True))[0][1].keys()
    for node_feature in node_layout:
        if not node_feature in list_of_nodes_attributes:
            raise AttributeError(
                'the selected feature is not in the list of attribute of nodes')

    for edge_feature in edge_layout:
        if not edge_feature in list(network_graph.edges(data=True))[0][2].keys():
            raise AttributeError(
                'the selected feature is not in the list of attribute of edges')
    if not clusters_color_group:
        node_color = [network_graph.nodes[node][node_layout[0]]
                      for node in network_graph.nodes]
    else:
        node_color = [clusters_color_group[node]
                      for node in network_graph.nodes]
    node_size = [network_graph.nodes[node][node_layout[1]]
                 for node in network_graph]
    edge_color = [[network_graph.get_edge_data(
        u, v)[edge_layout[0]] for u, v in network_graph.edges]]

    if edge_layout[1].lower() == 'contact_in_sequence':
        style = ['solid' if network_graph.get_edge_data(
            u, v)['contact_in_sequence'] else 'dashed' for u, v in network_graph.edges]

    elif edge_layout[1].lower() == 'stability':
        style = ['solid' if network_graph.get_edge_data(
            u, v)['stability'] >= 0. else 'dashed' for u, v in network_graph.edges]

    elif edge_layout[1].lower() == 'lenght':
        style = ['solid' if network_graph.get_edge_data(
            u, v)['lenght'] < 7. else 'dashed' for u, v in network_graph.edges]

    if len(edge_layout) == 3:
        width = [network_graph.get_edge_data(
            u, v)[edge_layout[2]] for u, v in network_graph.edges]
    if len(edge_layout) > 3:
        raise AttributeError('too many features for the edge layout')
    # NOTE if possible increase the features nodes
    if len(node_layout) > 2:
        raise AttributeError('too many features for the node layout')
    if label:
        with_labels = True
        font_weight = label[0]
        font_size = label[1]
    options = {
        'node_color': node_color,
        'node_size': node_size,
        'edge_color': edge_color,
        'style': style,
        'width': width,
        'with_labels': with_labels,
        'font_weight': font_weight,
        'font_size': font_size,
        'with_labels': True,
        'font_weight': str(label[0]),
        'font_size': label[1]
    }
    # NOTE remember to use **option in nx.draw
    return options


def main():
    plot_dir = Path(__file__).resolve().parents[2]
    print(plot_dir)


if __name__ == '__main__':
    main()
