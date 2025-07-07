### Code for simulating one data with a given parameter set
### This code will be updated later to move on clusters

import network_generation as netgen

from copy import deepcopy

import networkx as nx

import EoN

import simulation_functions as simfun

import numpy as np

from collections import defaultdict

import pandas as pd

import os

from datetime import datetime


if __name__ == "__main__":

    group = int(snakemake.params[0])
    beta_se = float(snakemake.params[1]) ## susceptible to exposure rate after contacting with infected individuals
    beta_ei = float(snakemake.params[2]) ## exposure to contagious/infected rate
    beta_rs = float(snakemake.params[3]) ## recover to susceptible rate (i.e. lost protection rate)
    population = int(snakemake.params[4])
    file_name = snakemake.output[0]

    num_rep = 5
    
    company_1K = 18
    school_1K = 0.4
    store_1K = 0.4
    grocery_discount = 1/40

    ### Edge specific parameters
   

    ## T
    T = 500

    ### Independent Transition parameters
    beta_ir = 0.1 ## infection to recover rate
    
    
    ### seed percentage
    seed_pct = 0.05

    alphas = [-4, -2, -1, 0, 1, 1.5, 2, 2.5, 3, 3.2, 3.5, 3.8, 4, 4.5, 5]

    spontaneous_transition = simfun.generate_spontaneous_transition(beta_ei, beta_ir, beta_rs)
    edge_reliant_transition = simfun.generate_edge_based_transition(beta_se)

    output = []
     
    
    for replicate in range(num_rep):
        base_city = netgen.city_network_sim(population, \
                                               n_company_per_1K_pop=company_1K, n_school_per_1K_pop=school_1K,\
                                              n_store_per_1K_pop= store_1K,
                                              grocery_discount = grocery_discount)
        now_big = datetime.now()
        

        for alpha in alphas:
            now = datetime.now()
            
            ### generate city
            current_city = deepcopy(base_city)
            
            current_city.preferential_attachment(alpha = alpha, beta = 0, gamma = 1, target_increase=0.5)
            
            current_city_graph = current_city.to_graph()

            ### simulate: The code was copied from Alec's code
            N = len(current_city_graph.nodes)
            
            seed_nodes = np.random.choice(N, size = int(np.round(N * seed_pct, decimals = 0)), replace = False)
            
            initial_condition = defaultdict(lambda : "S")
            
            for seed_initial_node in seed_nodes:
                initial_condition[seed_initial_node] = "I"
                
            return_statuses = ('S', 'E', 'I', 'R')
            
            results = EoN.Gillespie_simple_contagion(current_city_graph,spontaneous_transition, \
                                                     edge_reliant_transition, initial_condition, return_statuses, \
                                                        tmax = T, return_full_data=True)
            
            time_periods = len(results.t())

            output.append({
            'group':np.tile(group, time_periods),
            'replicate':np.tile(replicate, time_periods),
            't': results.t(),
            "alpha": list(np.tile(alpha, time_periods)),
            "beta_se": list(np.tile(beta_se, time_periods)),
            "N": list(np.tile(N, time_periods)),
            'S': results.S(),
            'I': results.I(),
            'R': results.R()})
            print("Replicate ", replicate, "Alpha=", alpha, "Simulation Completed, Time is: ", datetime.now() - now)
        print("Replicate", replicate, "Completed, Time is: ", datetime.now() - now_big)

        
        
    output = simfun.concatenate_dictionary(output)
        
    output.to_parquet(file_name)

    