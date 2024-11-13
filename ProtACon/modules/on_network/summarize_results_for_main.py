#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__email__ = 'renatoeliasy@gmail.com'
__author__ = 'Renato Eliasy'

'''
this script is useful to reassume the work on network, 
both for analysis and for visualization
its means is to slim the code to use in the __main__.py script
it works with
- the results of the network
- the result of kmeans
- the result of louvain
'''

from ProtACon.modules.on_network import networks_analysis, kmeans_computing_and_results, PCA_computing_and_results, Attention_map_from_networks, Collect_and_structure_data
from ProtACon.modules.miscellaneous import CA_Atom, get_AA_features_dataframe
from ProtACon.modules.contact import generate_distance_map, generate_instability_map
from ProtACon import network_vizualization as netviz
from ProtACon import process_instability, process_contact
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure
# Define the results for kmeans


def get_kmeans_results(
        CA_Atoms: tuple[CA_Atom, ...],

) -> tuple[
    pd.DataFrame,  # the updated dataframe
    dict,  # kmeans_labels
    np.ndarray,  # the attention map associated to km clusters
]:
    '''
    It give the results of the kmeans analysis
    Parameters:
    ----------
    CA_Atoms: tuple[CA_Atom,...]
        the tuple of the CA_Atom objects

    Returns:
    -------
    tuple[pd.DataFrame, tuple[int,...]]
        the updated dataframe
        the kmeans_labels
    '''
    feature_df = get_AA_features_dataframe(CA_Atoms=CA_Atoms)
    feature_df['AA_pos'] = Collect_and_structure_data.generate_index_df(
        CA_Atoms=CA_Atoms)
    feature_df.set_index('AA_pos', inplace=True)
    if ('AA_Name' in feature_df.columns):
        feature_df.drop(columns=['AA_Name'], inplace=True)
    kmeans_labels,  new_df = kmeans_computing_and_results.get_clusters_label(
        dataset=feature_df,
        cluster_feature=feature_df['AA_web_group'],
        columns_to_remove='AA_Coords'
    )
    km_labels_dict = new_df['cluster_group'].to_dict()
    # get the attention map for kmean( 1 for intra link comm, 0 for infra link communities)
    km_attention_map = np.zeros((len(CA_Atoms), len(CA_Atoms)))
    for i, AA_i in enumerate(km_labels_dict.keys()):
        for j, AA_j in enumerate(km_labels_dict.keys()):
            if km_labels_dict[AA_i] == km_labels_dict[AA_j]:
                km_attention_map[i, j] = 1

    return (new_df, km_labels_dict, km_attention_map)


def get_partition_results(CA_Atoms: tuple[CA_Atom, ...],
                          df: pd.DataFrame | dict,
                          ) -> tuple[float, float, float]:
    '''
    the funciton has the pourpose to calculate the parameter of homogeneity, completness, vmeasure 
    of the partition, considering as ground truth : base_df.AA_web_group, respecting the df.columns

    Parameters:
    ----------
    CA_Atoms: tuple[CA_Atom,...]
        the tuple of the CA_Atom objects
    df: pd.DataFrame
        the dataframe to be used for the partition analysis

    Returns:
    -------
    tuple[float, float, float]
        the homogeneity, the completness, the vmeasure  

    '''
    kmeans_columm_label = 'cluster_group'
    louvain_columns_label = 'louvain_community'
    base_df = get_AA_features_dataframe(CA_Atoms=CA_Atoms)
    ground_truth = base_df['AA_web_group'].values
    if isinstance(df, pd.DataFrame):
        if kmeans_columm_label in df.columns and louvain_columns_label in df.columns:
            raise ValueError('The dataframe must have up to one cluster label')
        elif kmeans_columm_label in df.columns:
            km_homo, km_compl, km_vm = homogeneity_completeness_v_measure(
                labels_true=ground_truth, labels_pred=df.cluster_group.values)
            return km_homo, km_compl, km_vm
        elif louvain_columns_label in df.columns:
            louvain_homo, louvain_compl, louvain_vm = homogeneity_completeness_v_measure(
                labels_true=ground_truth, labels_pred=df.louvain_community.values)
            return louvain_homo, louvain_compl, louvain_vm
        else:
            raise ValueError(
                f'something wrong in {df}\nplease control if the cluster is in {df.columns}')
    elif isinstance(df, dict):
        homo, compl, vm = homogeneity_completeness_v_measure(
            labels_pred=tuple(df.values()), labels_true=ground_truth)
        return homo, compl, vm

# summarize the steps to get the complete nx.Graph rapresentation of the protein


def prepare_complete_graph_nx(CA_Atoms: tuple[CA_Atom, ...],
                              binary_map: bool | np.ndarray = False,
                              ) -> tuple[nx.Graph, float]:
    '''
    from the CA_Atoms list it's in need:
    - the AA_dataframe
    - the the instability value from DIWV dict
    - the distancies between AAs
    - the bool to see if they are in contact or not
    it give back the complete graph and the resolution for the louvain communities computing
    '''
    node_name_for_Graph = Collect_and_structure_data.generate_index_df(
        CA_Atoms=CA_Atoms)
    instability_df = Collect_and_structure_data.get_dataframe_from_nparray(base_map=generate_instability_map(
        CA_Atoms=CA_Atoms), index_str=node_name_for_Graph, columns_str=node_name_for_Graph)
    distance_df = Collect_and_structure_data.get_dataframe_from_nparray(base_map=generate_distance_map(
        CA_Atoms=CA_Atoms), index_str=node_name_for_Graph, columns_str=node_name_for_Graph)

    if isinstance(binary_map, bool):
        list_of_edges = []
        for i, AA_i in enumerate(node_name_for_Graph):
            for j, AA_j in enumerate(node_name_for_Graph):
                if i < j:
                    list_of_edges.append((AA_i, AA_j))
    else:
        list_of_edges, _ = Collect_and_structure_data.get_list_of_edges(
            base_map=binary_map, CA_Atoms=CA_Atoms)

    weights_for_edges = Collect_and_structure_data.get_weight_for_edges(
        list_of_edges=list_of_edges,
        base_df=distance_df,
        instability_df=instability_df)
    nx_graph, resolution = Collect_and_structure_data.get_the_Graph_network(
        CA_Atoms=CA_Atoms, edges_weight_list=weights_for_edges)

    return (nx_graph, resolution)


# Define the results for louvain
def get_louvain_results(CA_Atoms: tuple[CA_Atom, ...],
                        base_Graph: nx.Graph,
                        resolution: float,
                        threshold: float = None,  # optional
                        threshold_type: str = 'zero',  # optional
                        # optional
                        edge_weights_combination: tuple[float,
                                                        float, float] | dict = False
                        ) -> tuple[nx.Graph, dict, np.ndarray]:
    '''
    It give the results of the louvain analysis
    Parameters:
    ----------
    CA_Atoms: tuple[CA_Atom,...]
        the tuple of the CA_Atom objects
    base_Graph: nx.Graph
        the base graph to be used for the louvain analysis
    edge_weights_combination: tuple[float,float,float] | dict
        the weights to be used on the edge for the community modulation:
        if a dict was given it has to have the keys: {'contact_in_sequence' : float
                                                      'lenght' : float
                                                      'instability' : float}
    threshold: float
        the threshold to be used for the proximity graph, to filter the edge considering a certain threshold
        it can be Not specified if the map used for the base graph it was already filtred
    Threshold_type : str
        the type of threshold to be used, it can be 'zero' or 'abs', zero is not included if 'zero' is selected 
    Returns:
    -------
    tuple[nx.Graph, tuple[int,...], np.ndarray]
        the updated Graph
        the louvain_labels
        the attention map associated to intra link of communities
    '''

    # assessing the weight on edge, respecting the type of data
    # of edge_weights_combination:
    if not edge_weights_combination:
        edge_weights = networks_analysis.weight_on_edge()
    elif isinstance(edge_weights_combination, tuple):
        edge_weights = networks_analysis.weight_on_edge(
            contact=edge_weights_combination[0],
            lenght=edge_weights_combination[1],
            instability=edge_weights_combination[2]
        )
    elif isinstance(edge_weights_combination, dict):
        for k in ['contact_in_sequence', 'lenght', 'instability']:
            if k not in edge_weights_combination.keys():
                raise KeyError(
                    f"the key {k} is missing in the edge_weights_combination dict")

        edge_weights = edge_weights_combination
    # add 'weight_combination' as attribute to node on graph
    new_graph = networks_analysis.add_weight_combination(G=base_Graph,
                                                         weight_to_edge=edge_weights)

    # if a threshold was given, filter the graph:
    if threshold != None:
        for source, target in new_graph.edges:
            if threshold_type.lower() == 'abs':
                if not abs(new_graph.get_edge_data(source, target)['weight_combination']) < threshold:
                    new_graph.remove_edge(source, target)
            elif threshold_type.lower() == 'zero':
                if not (0 < new_graph.get_edge_data(source, target)['weight_combination'] < threshold):
                    new_graph.remove_edge(source, target)
            else:
                raise ValueError('The threshold type is not recognized')

    # add the louvain community attribute to each node in the graph:
    final_Graph, louvain_communities = networks_analysis.add_louvain_community_attribute(G=new_graph,
                                                                                         weight_of_edges='weight_combination',
                                                                                         resolution=resolution)
    louvain_attention_map = Attention_map_from_networks.binary_map_from_clusters(
        proximity_graph=final_Graph)
    # finally return both the graph with weight_combination attributes on edges and louvain_community on nodes:
    return (final_Graph, louvain_communities, louvain_attention_map)


def plot_the_3D_chain(CA_Atoms: tuple[CA_Atom, ...],
                      protein_name: str = 'vattelapesca',
                      first_feature_edge: str = 'contact',
                      second_feature_edge: str = 'sequential',
                      third_feature_edge: str = 'instability',
                      node_colors: str | dict = '',
                      df_col_feature: str | pd.Series = '',
                      save_option: bool = False,
                      ) -> None:
    '''
    the plot function try to handle the big data to plot the 3d chain
    in netviz.plot_protein_chain_3D...
    Parameter
    ---------
    CA_Atoms: tuple[CA_Atom,...]
        the tuple of the CA_Atom objects
    first_feature_edge: str
        the first feature to be used on the edge
    second_feature_edge: str
        the second feature to be used on the edge
    third_feature_edge: str
        the third feature to be used on the edge
    first_feature_node: str
        the first feature to be used on the node

    '''
    positional_aa = Collect_and_structure_data.generate_index_df(
        CA_Atoms=CA_Atoms)
    _, _, bin_con_map = process_contact.main(CA_Atoms=CA_Atoms)
    contact_edges = np.argwhere(bin_con_map == 1)  # in int format
    _, _, bin_inst_map = process_instability.main(CA_Atoms=CA_Atoms)
    instability_edges = np.argwhere(bin_inst_map == 1)  # in int format
    sequential_edges = []
    for i in range(0, len(positional_aa)-1):
        sequential_edges.append((i, i+1))  # in nit format
    features_edge = [first_feature_edge,
                     second_feature_edge, third_feature_edge]
    edges_feature = [contact_edges, sequential_edges, instability_edges]
    for n, feature in enumerate(features_edge):
        if feature == 'instability':
            edges_feature[n] = instability_edges
        elif feature == 'contact':
            edges_feature[n] = contact_edges
        elif feature == 'sequential':
            edges_feature[n] = sequential_edges

    netviz.plot_protein_chain_3D(CA_Atoms=CA_Atoms,
                                 edge_list1=edges_feature[0],
                                 edge_list2=edges_feature[1],
                                 edge_list3=edges_feature[2],
                                 color_map=node_colors,
                                 color_feature=df_col_feature,
                                 protein_name=protein_name,
                                 save_option=save_option)
    # FIXME rifare da qui la funzione di plot 3D considerando le modifiche su net-work
    # dato che troppo complessa da spacchettare avendo il edge_layout da cui prendere gli estremidegli edges
    # è più facile se ricominciamo tenendo conto delle modifiche piuttosto che non cambiare quello che già c'è
