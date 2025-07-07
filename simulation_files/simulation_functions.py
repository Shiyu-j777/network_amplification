import networkx as nx
import pandas as pd


def generate_spontaneous_transition(beta_ei = 0.2, beta_ir = 0.1, beta_rs = 0):
    """
    function to generate spontaneous transition graph
    Input:
        beta_ei(float): exposure to infected probability (incubation period)
        beta_ir(float): infected to recover probability
        beta_rs(float): recover to susceptible probability
    Output:
        spontaneous_graph
    """
    spontaneous_graph = nx.DiGraph()
    spontaneous_graph.add_node('S')
    spontaneous_graph.add_edge('E', 'I', rate = beta_ei)
    spontaneous_graph.add_edge('I', 'R', rate = beta_ir)
    spontaneous_graph.add_edge('R', 'S', rate = beta_rs)

    return spontaneous_graph


def generate_edge_based_transition(beta_se = 0.05, weight_column = "weight"):
    """
    function to generate edge_reliant transition graph
    Input:
        beta_se(float): susceptible to exposure probability (transmission)
    Output:
        edge_reliant_graph
    """
    edge_reliant_graph = nx.DiGraph()
    edge_reliant_graph.add_edge(('I', 'S'), ('I', 'E'), rate = beta_se, weight_label = weight_column)
     
    return edge_reliant_graph

def concatenate_dictionary(list_of_dict):
    """
    function to concatenate a list of dictionary with identical keys and generate a dataframe
    Input:
        list of dictionary with identical keys
    Output:
        output_DF: an output dataframe
    """

    output_DF = dict()

    keys = list(list_of_dict[0].keys())

    ### initialize dict
    for key in keys:
        output_DF[key] = []

    for i in range(len(list_of_dict)):
        current_dict = list_of_dict[i]
        for key in keys:
            output_DF[key] = output_DF[key] + list(current_dict[key])
    
    output_DF = pd.DataFrame.from_dict(output_DF)

    return output_DF