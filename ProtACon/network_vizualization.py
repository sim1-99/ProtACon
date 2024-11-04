#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__email__ = 'renatoeliasy@gmail.com'
__author__ = 'Renato Eliasy'

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ProtACon.modules.on_network.networks_analysis import rescale_0_to_1
from ProtACon.modules.on_network import PCA_computing_and_results as PCA_results
from ProtACon.modules.on_network import Collect_and_structure_data
from ProtACon.modules.on_network import networks_analysis as netly
import plotly.graph_objects as go
import igraph as ig
from ProtACon import config_parser
from ProtACon.modules.miscellaneous import assign_color_to, get_AA_features_dataframe, CA_Atom, get_var_name
from ProtACon.modules.on_network.Collect_and_structure_data import get_indices_from_str, generate_index_df
import networkx as nx
from typing import Mapping
import logging
from pathlib import Path
import os


def plot_histogram_pca(percentage_var: tuple[float, ...],
                       best_features: tuple[str, ...],
                       protein_name: str,
                       save_option: bool = False
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
    save_option: bool
        the option to save the plot as default is False

    Returns:
    -------
    None, but plot a figure

    """
    config = config_parser.Config("config.txt")
    folder_name = config.get_paths()
    networks_path = folder_name["NET_FOLDER"]
    folder_path = Path(__file__).resolve().parents[1]/networks_path

    protein_name = protein_name.upper()

    labels = ['PC' + str(i) for i in range(1, len(percentage_var)+1)]
    plt.bar(x=range(1, len(percentage_var)+1),
            height=percentage_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Components')
    plt.title('PCA components of {0}'.format(protein_name))
    plt.legend(['PC1-> {0}\nPC2-> {1}\nPC3-> {2}'.format(best_features[0],
               best_features[1], best_features[2])])

    save_path = folder_path/protein_name/'PCAs_components.png'
    save_path.parent.mkdir(exist_ok=True, parents=True)
    if save_option:
        for i in range(3):
            if os.path.isfile(str(save_path)):
                save_path = folder_path / protein_name / \
                    f'PCAs_components({i}).png'
            else:
                plt.savefig(save_path)
    plt.show()

    return None


def plot_pca_2d(pca_dataframe: pd.DataFrame,  # dataframe from which take the components
                protein_name: str,  # the name of the protein whose plot refers to
                # the features to show on the plot corresponting to the PC1 and PC2 most compatible components
                best_features: tuple[str, ...],
                # the amount of compatibility of the feature and the component
                percentage_var: tuple[float, ...],
                color_map: pd.Series | dict = False,
                save_option: bool = False
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
        the color map to be used for the scatter plot to cluster the points:
        it can be  'pca_dataframe.PC1', 'pca_dataframe.PC2' or a dictionary having for nodes the index of pca_dataframe as default is false
    save_option: bool
        the option to save the plot as default is False
    Returns:
    -------
    None, but plot a scatter plot 2d

    """
    config = config_parser.Config("config.txt")
    folder_name = config.get_paths()
    networks_path = folder_name["NET_FOLDER"]
    folder_path = Path(__file__).resolve().parents[1]/networks_path
    protein_name = protein_name.upper()

    labels = ['PC' + str(i) for i in range(1, len(percentage_var)+1)]
    for label in labels:
        if label not in pca_dataframe.columns:
            raise ValueError(
                'The dataframe must have the same columns as the labels of PC1, PC2...')
    x_values = pca_dataframe.PC1
    y_values = pca_dataframe.PC2

    fig, ax = plt.subplots(figsize=(10, 8))

    if not color_map:
        scatter = ax.scatter(x_values, y_values, color='blue')
        cbar = plt.colorbar(scatter, location='top')

    elif isinstance(color_map, dict):
        if set(color_map.keys()) != set(pca_dataframe.index):
            raise ValueError(
                'the dictionary must have the same keys as the index of the dataframe')
        else:
            c_map = [color_map[el] for el in pca_dataframe.index]
            scatter = ax.scatter(x_values, y_values, c=c_map, cmap='viridis')
            cbar = plt.colorbar(scatter, location='top')
            color_map = c_map
    else:
        scatter = ax.scatter(x_values, y_values, c=color_map, cmap='viridis')
        if color_map == x_values:
            cbar = plt.colorbar(scatter, location='bottom')
        elif color_map == y_values:
            cbar = plt.colorbar(scatter, location='left')
    cbar_name = get_var_name(color_map)  # FIXME
    # FIXME better to chose another wawy to assign name to color_map
    cbar.set_label('{0}'.format(cbar_name))

    plt.title('PCAs Scatter Plot of {0}'.format(protein_name))
    plt.xlabel('PC1-> {0} : {1}'.format(best_features[0], percentage_var[0]))
    plt.ylabel('PC2-> {0} : {1}'.format(best_features[1], percentage_var[1]))

    save_path = folder_path / protein_name.upper() / "PCA_2D.png"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    if save_option:
        for i in range(3):
            if os.path.isfile(str(save_path)):
                save_path = folder_path / protein_name / f'PCA_2D({i}).png'
            else:
                plt.savefig(save_path)
    plt.show()

    return None


def plot_pca_3d(pca_dataframe: pd.DataFrame,  # dataframe from which take the components
                protein_name: str,  # the name of the protein whose plot refers to
                # the features to show on the plot corresponting to the PC1 and PC2 most compatible components
                best_features: tuple[str, ...],
                # the amount of compatibility of the feature and the component
                percentage_var: tuple[float, ...],
                color_map: pd.Series = False,
                save_option: bool = False
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
    save_option: bool
        the option to save the plot as default is False
    Returns:
    -------
    None, but plot a scatter plot 3D
    """
    config = config_parser.Config("config.txt")
    folder_name = config.get_paths()
    networks_path = folder_name["NET_FOLDER"]
    folder_path = Path(__file__).resolve().parents[1]/networks_path
    protein_name = protein_name.upper()

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
    labels = []
    for index in pca_dataframe.index:
        labels.append(str(index))
    if not color_map:
        scatter = ax.scatter(x_values, y_values, z_values,
                             color='blue')
    else:
        if isinstance(color_map, dict):
            if set(color_map.keys()) != set(pca_dataframe.index):
                raise ValueError(
                    'the dictionary must have the same keys as the index of the dataframe')
            else:
                c_map = [color_map[el] for el in pca_dataframe.index]
                scatter = ax.scatter(x_values, y_values, z_values,
                                     c=c_map, cmap='viridis')
        else:
            scatter = ax.scatter(x_values, y_values, z_values,
                                 c=color_map, cmap='viridis')
    plt.legend()
    plt.title(f'PCA 3D-Scatter Plot of {protein_name} protein')
    ax.set_xlabel('{0} -{1}%'.format(best_features[0], percentage_var[0]))
    ax.set_ylabel('{0} -{1}%'.format(best_features[1], percentage_var[1]))
    ax.set_zlabel('{0} -{1}%'.format(best_features[2], percentage_var[2]))

    save_path = folder_path / protein_name.upper() / "PCA_3D.png"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    if save_option:
        for i in range(3):
            if os.path.isfile(str(save_path)):
                save_path = folder_path / protein_name / f'PCA_3D({i}).png'
            else:
                fig.savefig(save_path)
    plt.show()

    return None

# Following the spatial visualization of the protein in the space,
# with feature enhanced by color


def plot_protein_3D(CA_Atoms: tuple[CA_Atom, ...],
                    edge_list1: list[tuple[int, int]
                                     ] | list[tuple[str, str]],
                    edge_list2: list[tuple[int, int]
                                     ] | list[tuple[str, str]],
                    color_map: str | dict = '',
                    edge_list3: list = [],
                    protein_name: str = None,
                    save_option: bool = False
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
    None but it plot a graphe 
    """
    trace_1edges, trace_2edges, trace_3edges = '', '', ''
    feature_dataframe = get_AA_features_dataframe(CA_Atoms=CA_Atoms)
    node_label = generate_index_df(CA_Atoms=CA_Atoms)
    feature_dataframe['AA_pos'] = node_label

    if not feature_dataframe.columns.str.contains(color_map).any():
        if isinstance(color_map, dict):
            node_color = [color_map[node] for node in node_label]
        else:
            logging.warning(
                'unable to find the feature selected in color_map between the dataframe features\nthe color is set to be the same')
            color_map = 'blue'

    # trasform the dataframe in a dictionary of records, easier to access to
    df_feature_dict = feature_dataframe.to_dict(orient='records')

    nodes = [AA_dict for AA_dict in df_feature_dict]
    df_positional = feature_dataframe.set_index(feature_dataframe['AA_pos'])
    dict_coordinates_positional = df_positional['AA_Coords'].to_dict()

    N = len(nodes)
    # control if edge lists are index or str: FOR EDGE_LIST1
    edge_lists = [edge_list1, edge_list2, edge_list3]

    for source, target in edge_list1:
        if type(source) != type(target):
            raise ValueError(
                'the edge list must have the same type of elements')
        elif isinstance(source, str):
            trace_1edges = get_indices_from_str(list=edge_list1,
                                                dataframe_x_conversion=feature_dataframe,
                                                column_containing_key='AA_pos')
            break
        elif isinstance(source, int):
            trace_1edges = edge_list1
            break

    for source, target in edge_list2:
        if type(source) != type(target):
            raise ValueError(
                'the edge list must have the same type of elements')
        elif isinstance(source, str):
            trace_2edges = get_indices_from_str(list=edge_list2,
                                                dataframe_x_conversion=feature_dataframe,
                                                column_containing_key='AA_pos')
            break
        elif isinstance(source, int):
            trace_2edges = edge_list2
            break

    # for EDGE LIST 3
    if len(edge_list3):
        for source, target in edge_list3:
            if type(source) != type(target):
                raise ValueError(
                    'the edge list must have the same type of elements')
            elif isinstance(source, str):
                trace_3edges = get_indices_from_str(list=edge_list3,
                                                    dataframe_x_conversion=feature_dataframe,
                                                    column_containing_key='AA_pos')
                break
            elif isinstance(source, int):
                trace_3edges = edge_list3
                break

    traces_edges = []

    G = ig.Graph(trace_1edges, directed=False)

    # labels stay for the name of the node
    labels = node_label

    if color_map == 'blue':
        node_color = ['blue' for _ in range(N)]
    elif isinstance(color_map, dict):
        node_color = [color_map[el] for el in labels]
    else:
        list_of_items = [element for element in feature_dataframe.color_map]
        color_dict = assign_color_to(discrete_list_of=list_of_items)
        if not color_dict:
            node_color = ['blue' for _ in range(N)]
            logging.warning('too much element to map')
        else:
            node_color = [color_dict[el] for el in list_of_items]

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

    trace_node = go.Scatter3d(
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
        data = [trace1, trace2, trace3, trace_node]
    else:
        data = [trace1, trace2, trace_node]

    fig = go.Figure(data=data, layout=layout)

    config = config_parser.Config("config.txt")
    path_name = config.get_paths()
    networks_path = path_name["NET_FOLDER"]
    folder_path = Path(__file__).resolve().parent/networks_path
    save_path = folder_path / protein_name.upper()/"3D_protein_chain.png"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    if save_option:
        for i in range(3):
            if os.path.isfile(str(save_path)):
                save_path = folder_path / protein_name / \
                    f'3D_protein_chain({i}).png'
            else:
                fig.savefig(save_path)

    """ if save_path.isfile():
        path = folder_path / protein_name.upper() / "3D_protein_chain(1).png"
        save_path = os.path.join(os.getcwd(), path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    if save_option:
        fig.savefig(Path(save_path))"""
    fig.show()

    return None


def plot_protein_chain_3D(CA_Atoms: tuple[CA_Atom, ...],  # 2.0 version
                          edge_list1: list[tuple[int, int]
                                           ] | list[tuple[str, str]],
                          edge_list2: list[tuple[int, int]
                                           ] | list[tuple[str, str]] = [],
                          color_map: str | dict = '',
                          color_feature: str | pd.Series = '',
                          edge_list3: list = [],
                          protein_name: str = None,
                          save_option: bool = False
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
    None but it plot a graphe 
    """
    G = ig.Graph()
    feature_dataframe = get_AA_features_dataframe(CA_Atoms=CA_Atoms)

    # set indiced as C(0)
    node_label = generate_index_df(CA_Atoms=CA_Atoms)
    feature_dataframe['AA_pos'] = node_label
    positional_df = feature_dataframe.set_index(feature_dataframe['AA_pos'])

    # trasform the dataframe in a dictionary of records, easier to access to
    df_feature_dict = feature_dataframe.to_dict(orient='records')
    node_coordinates_dict = positional_df['AA_Coords'].to_dict()

    # control if edge lists are index or str adn append resutls in type_list
    edge_lists = [edge_list1, edge_list2, edge_list3]
    type_list = ['int', 'int', 'int']
    for i, edges in enumerate(edge_lists):
        for source, target in edges:
            if type(source) != type(target):
                raise ValueError(
                    f'the edge list{edges} must have the same type of elements')
            elif isinstance(source, str):
                type_list[i] = 'str'
                break
            elif isinstance(source, int):
                type_list[i] = 'int'
                break

    # ricava le liste degli edges in modalità di interi, che provengano da liste di stringhe o num
    num_edge_lists = edge_lists
    for i, edges in enumerate(edge_lists):
        if type_list[i] == 'str':
            num_edge_lists[i] = get_indices_from_str(list=edges,
                                                     dataframe_x_conversion=feature_dataframe,
                                                     column_containing_key='AA_pos')
        else:
            num_edge_lists[i] = edges

    trace_edges1 = num_edge_lists[0]
    trace_edges2 = num_edge_lists[1]
    trace_edges3 = num_edge_lists[2]

    tot_edges = []
    for el in edge_lists:
        tot_edges.extend(el)

    # set color of nodes depending color map
    if not color_map and not color_feature:
        node_colors = ['blue' for _ in range(len(CA_Atoms))]

    elif isinstance(color_map, dict):
        # as node labels.keys() are C(0) format
        node_colors = [color_map[el] for el in node_label]

    elif isinstance(color_feature, pd.Series):
        if color_map:
            mapping = getattr(plt.cm, color_map)
        else:
            mapping = plt.cm.viridis
        node_colors = mapping(rescale_0_to_1(
            color_feature.values.astype(float)))

    elif isinstance(color_feature, str):
        if not (color_feature in feature_dataframe.columns.astype(str)):
            raise ValueError(
                f'the selected feature {color_feature} is not in the dataframe columns {feature_dataframe.columns}\n reconsider to use the dataframe columns itself, instead the name of column')
        else:
            if color_map:
                mapping = getattr(plt.cm, color_map)
            else:
                mapping = plt.cm.viridis
            node_colors = mapping(rescale_0_to_1(
                feature_dataframe[color_feature].values.astype(float)))
    else:
        mapping = plt.cm.viridis
        node_colors = mapping(rescale_0_to_1(
            feature_dataframe['AA_local_isoPH'].values.astype(float)))
        logging.info(
            msg='something off however it sssing color based on local_isoPH')

    # get nodes coordinates edges in an rgb way
    node_coordinates_dict
    Xn = [node_coordinates_dict[node][0] for node in node_label]
    Yn = [node_coordinates_dict[node][1] for node in node_label]
    Zn = [node_coordinates_dict[node][2] for node in node_label]

    Xe1, Ye1, Ze1 = [], [], []
    Xe2, Ye2, Ze2 = [], [], []
    Xe3, Ye3, Ze3 = [], [], []
    node_coords = list(node_coordinates_dict.values())
    edge_colors1, edge_colors2, edge_colors3 = [], [], []

    for edge in trace_edges1:
        Xe1 += [node_coords[edge[0]][0], node_coords[edge[1]][0], None]
        Ye1 += [node_coords[edge[0]][1], node_coords[edge[1]][1], None]
        Ze1 += [node_coords[edge[0]][2], node_coords[edge[1]][2], None]
        edge_colors1.append(calculate_rgb_edge_color(
            edge=edge, edge_list1=trace_edges1, edge_list2=trace_edges2, edge_list3=trace_edges3))

    for edge in trace_edges2:
        Xe2 += [node_coords[edge[0]][0], node_coords[edge[1]][0], None]
        Ye2 += [node_coords[edge[0]][1], node_coords[edge[1]][1], None]
        Ze2 += [node_coords[edge[0]][2], node_coords[edge[1]][2], None]
        edge_colors2.append(calculate_rgb_edge_color(
            edge=edge, edge_list1=trace_edges1, edge_list2=trace_edges2, edge_list3=trace_edges3))

    for edge in trace_edges3:
        Xe3 += [node_coords[edge[0]][0], node_coords[edge[1]][0], None]
        Ye3 += [node_coords[edge[0]][1], node_coords[edge[1]][1], None]
        Ze3 += [node_coords[edge[0]][2], node_coords[edge[1]][2], None]
        edge_colors3.append(calculate_rgb_edge_color(
            edge=edge, edge_list1=trace_edges1, edge_list2=trace_edges2, edge_list3=trace_edges3))

    trace1 = go.Scatter3d(
        x=Xe1,
        y=Ye1,
        z=Ze1,
        mode='lines',
        line=dict(color='rgb(0,0,255)', width=5),
        hoverinfo='none'
    )
    trace2 = go.Scatter3d(
        x=Xe2,
        y=Ye2,
        z=Ze2,
        mode='lines',
        line=dict(color='rgb(0,255,0)', width=5),
        hoverinfo='none'
    )
    trace3 = go.Scatter3d(
        x=Xe3,
        y=Ye3,
        z=Ze3,
        mode='lines',
        line=dict(color='rgb(255,0,0)', width=5),
        hoverinfo='none'
    )

    trace_node = go.Scatter3d(
        x=Xn,
        y=Yn,
        z=Zn,
        mode='markers',
        name='amminoacid',
        marker=dict(symbol='circle',
                    size=6,
                    color=node_colors,  # NOTE controlla per colorscale
                    line=dict(color='rgb(50,50,50)', width=0.5)
                    ),
        text=node_label,
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

    data = [trace1, trace2, trace3, trace_node]
    fig = go.Figure(data=data, layout=layout)
    config = config_parser.Config("config.txt")
    path_name = config.get_paths()
    networks_path = path_name["NET_FOLDER"]
    folder_path = Path(__file__).resolve().parent/networks_path
    save_path = folder_path / protein_name.upper()/"3D_protein_chain.png"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    if save_option:
        for i in range(3):
            if os.path.isfile(str(save_path)):
                save_path = folder_path / protein_name / \
                    f'3D_protein_chain({i}).png'
            else:
                fig.savefig(save_path)
    fig.show()
    pass


def calculate_rgb_edge_color(edge: tuple[int, int],
                             edge_list1: list[tuple[int, int]] = [],
                             edge_list2: list[tuple[int, int]] = [],
                             edge_list3: list[tuple[int, int]] = [],
                             ) -> str:
    '''
    it get the string necessary to get the color in rgb format to traces
    Parameters
    ----------
    edge : tuple[int, int]
        the edge to be colored
    edge_list1 : list[tuple[int, int]]
        the first list of edges
    edge_list2 : list[tuple[int, int]]
        the second list of edges
    edge_list3 : list[tuple[int, int]]
        the third list of edges
    '''
    r, g, b = 0, 0, 0
    if np.any(edge == edge_list1):
        r = 255
    if np.any(edge == edge_list2):
        g = 255
    if np.any(edge == edge_list3):
        b = 255
    if r+b+g == 0:
        return 'rgb(50,50,50)'

    return f'rgb({r},{g},{b})'


def draw_network(network_graph: nx.Graph,
                 # the attribute of the nodes to map the color
                 node_color: list[float] | str = 'AA_local_isoPH',
                 node_size: list[float] | str = 'AA_Volume',
                 edge_color: str = 'instability',
                 edge_style: str = 'contact_in_sequence',
                 pos: dict | str = 'kk',  # as default kamadakawaii
                 clusters_color_group: dict = {},  # use the dict to get the map of colors
                 label: tuple[str, int] = ('bold', 10),
                 save_option: bool = False
                 ) -> None:
    """
    plot a network based on certain feature:

    Parameters
    ----------
    network_graph: nx.Graph
        the network graph to be drawn, it contains the following attributes,at least for nodes and edges:
        - NODES: 'AA_Name', 'AA_Coords', 'AA_Hydropathy', 'AA_Volume', 'AA_Charge', 'AA_PH', 'AA_iso_PH', 'AA_Hydrophilicity', 'AA_Surface_accessibility',
                        'AA_ja_transfer_energy_scale', 'AA_self_Flex', 'AA_local_flexibility', 'AA_secondary_structure', 'AA_aromaticity', 'AA_human_essentiality'

        - EDGES: 'lenght', 'instability', 'contact_in_sequence'
    node_color: 
        or a list of float/int to assing to each node in G.nodes() or a feature attribute in G.nodes()
    node_size:
        or a list of float/int to assing to each node in G.nodes() or a feature attribute in G.nodes()
    edge_color:
        the feature to color the edges, preset
    edge_style:
        the feature to style the edges, preset
    pos: dict | str = 
        the position of the nodes as a list of {node : (x, y), ...}
    clusters_color_group: dict
        the dictionary of the color mapping of the nodes it hase to be the format: {'C(0)' : 1 , 'P(1)' : 2 ....}
    save_option : bool
        the option to save the plot as default is False
    Return
    ------
    a plot network

    """
    config_file_path = Path(__file__).resolve().parent/"config.txt"
    config = config_parser.Config(config_file_path)
    cutoffs = config.get_cutoffs()
    instability_cutoff = cutoffs['INSTABILITY_CUTOFF']
    stability_cutoff = cutoffs['STABILITY_CUTOFF']
    contact_cutoff = cutoffs['DISTANCE_CUTOFF']

    list_of_nodes_attributes, _ = netly.get_node_atttribute_list(
        G=network_graph)
    list_of_edge_attributes, _ = netly.get_edge_attribute_list(G=network_graph)
    # TODO controlla che i node ed edge sono tra gli attributi
    node_layout = [node_color, node_size]

    for feature in node_layout:
        # if set(list_of_nodes_attributes) != set(list_of_nodes_attributes.append(feature)):
        if str(feature) not in [str(a) for a in list_of_nodes_attributes]:
            raise AttributeError(
                f'the selected feature {feature} is not in the list of attribute of nodes: {list_of_nodes_attributes}')

    edge_layout = [edge_style, edge_color]
    for feature in edge_layout:

        if str(feature) not in list_of_edge_attributes and feature != '':
            raise AttributeError(
                f'the selected feature {feature} is not in the list of attribute of edges {list_of_edge_attributes}')

    for node in clusters_color_group.keys():  # controllo formato cluster
        if node not in network_graph.nodes():
            raise ValueError(
                f'the node {node} in the clusters_color_group is not in the network_graph')
    if len(clusters_color_group.keys()):
        node_color_layout = [clusters_color_group[node]
                             for node in network_graph.nodes()]
    else:
        node_color_layout = [network_graph.nodes[node][node_color]
                             for node in network_graph.nodes()]

    node_size_layout = [network_graph.nodes[node][node_size]
                        for node in network_graph.nodes()]

    # edges informations
    if edge_style.lower() == 'contact_in_sequence':
        style = ['solid' if network_graph.get_edge_data(
            u, v)['contact_in_sequence'] else 'dashed' for u, v in network_graph.edges]

    elif edge_style.lower() == 'instability':
        style = ['solid' if network_graph.get_edge_data(
            u, v)['instability'] <= float(instability_cutoff) else 'dashed' for u, v in network_graph.edges]

    elif edge_style.lower() == 'lenght':
        style = ['solid' if network_graph.get_edge_data(
            u, v)['lenght'] < float(contact_cutoff) else 'dashed' for u, v in network_graph.edges]

    if edge_color != '':

        edge_color_layout = [float(edgedata[edge_color])
                             for _, _, edgedata in network_graph.edges(data=True)]
    else:
        edge_color_layout = ['black' for _ in network_graph.edges]

    # pos
    if pos == 'kk':  # more layouts?
        pos = nx.kamada_kawai_layout(network_graph)
    elif isinstance(pos, dict):
        for k, v in pos.items():
            if len(v) != 2:
                raise ValueError(
                    f'the position of the node {k} is not in the right format')

    node_options = {
        'node_color': rescale_0_to_1(node_color_layout),
        'node_size': rescale_0_to_1(node_size_layout)*500,
        'edgecolors': ['black' for _ in network_graph.nodes()],

    }
    edge_options = {
        'style': style,
        'edge_color': edge_color_layout,
        'edge_cmap': plt.cm.viridis,
    }

    label_options = {
        'labels': {n: n for n in network_graph.nodes},
        'font_weight': label[0],
        'font_size': label[1]
    }
    print(f'# of nodes: {len(network_graph.nodes())}')
    print(f'len of node_color: {len(rescale_0_to_1(node_color_layout))}')
    print(f'len of node_size: {len(rescale_0_to_1(node_size_layout))}')
    print(f'# of edges: {len(network_graph.edges())}')
    print(f'len of edge_style: {len(style)}')
    print(f' edge_color: {True in edge_color_layout}')
    plt.figure(figsize=(12, 12))

    nx.draw_networkx_nodes(network_graph, pos, **node_options)
    nx.draw_networkx_edges(network_graph, pos, **edge_options)
    if label_options:
        nx.draw_networkx_labels(network_graph, pos, **label_options)

    # TODO salva il network
    path_name = config.get_paths()
    networks_path = path_name["NET_FOLDER"]
    folder_path = Path(__file__).resolve().parent/networks_path
    title = f'nc_{node_color}_es_{edge_style}'
    save_path = folder_path / f'network_{title}.png'
    save_path.parent.mkdir(exist_ok=True, parents=True)
    if save_option:
        for i in range(3):
            if os.path.isfile(save_path):
                save_path = folder_path / f'network_graph({i}).png'
            else:
                plt.savefig(save_path)
    # disegan il network
    plt.show()
    plt.close()
    return None
