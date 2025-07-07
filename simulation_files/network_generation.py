import numpy as np
import utils
from collections import Counter

import networkx as nx
import pandas as pd


class city_network_sim():
    def __init__(self, N, partner_prob = 0.8, max_num_child = 3, 
                 n_company_per_1K_pop = 18, n_store_per_1K_pop = 0.4,
                 n_school_per_1K_pop = 0.4, employment_rate = 0.6, enrollment_rate = 0.75, grocery_discount = 1/40):
        """
        Generate a city network
        Input:
            N(int): the number of nodes in the network
            alpha(float): the skewness of the random tie degress distribution to the network
            beta (float): an adjustment factor to the degree distribution probability distribution intercept
            gamma (float): adjustment factor for degree distribution within the power
            partner_prob (float): the probability of an individual having the next one as partner, default is 0.8
            max_num_child (int): maximum number of children that could be uniformly sampled
            n_company_per_1K_pop (float): number of companies per 1K population, will be rounded
            n_store_per_1K_pop (float): number of stores per 1K population, will be rounded
            n_school_per_1K_pop (float): number of schools per 1K population, will be rounded
            employment_rate (float): employment rate of the adults
            enrollment_rate (float): enrollment rate of children
            grocery_discount (float): ratio of contagion likelihood between grocery shopping versus working/studying
        Fields: 
            N is the number of nodes
            nodelist is the list of nodes
            node_attr is the a pd.dataframe that comes from a dictionary that contains 
                adult info and foci info (family, company, school,grocery)
            edgelist is pd.dataframe that comes from a dict with a tuple of nodes (from, to) mapped to weight,\
                from_node number should be strictly smaller than to_node number
            S is the list of susceptible individuals (initialized as all individuals)
            E is the list of exposed individuals
            I is the list of infected individuals
            R is the list of recovered individuals 
        """

        self.N = N
        self.nodelist = np.arange(N)
        self.node_attr = {"node": self.nodelist,\
                          "have_been_infected":[False] * N} 
        ## edgelist naming convention: smaller number, larger number, weight
        self.edgelist = dict()

        ## generate family info
        adult_attribute, household_record, household_edgelist = self._gen_family(N, partner_prob,max_num_child)
        
        self.node_attr["adult"] = adult_attribute
        self.node_attr["household"] = household_record
        self.edgelist.update(household_edgelist)
        del(adult_attribute)
        del(household_record)
        del(household_edgelist)

        ## generate foci - work
        workplace, workplace_edgelist = self._gen_workplaces(round(N/1000*n_company_per_1K_pop), decay_factor=1/2, employment_rate= employment_rate)
        self.node_attr["workplace"] = workplace
        self.edgelist = self._outer_join_sum_two_edgelists(self.edgelist, workplace_edgelist)
        del(workplace)
        del(workplace_edgelist)

        ## generate foci - shopping (one adult in household visit one grocery store)

        grocery_store, grocery_edgelist = self._gen_grocery_stores(round(N/1000*n_store_per_1K_pop), grocery_discount)
        self.node_attr["grocery_store"] = grocery_store
        self.edgelist = self._outer_join_sum_two_edgelists(self.edgelist, grocery_edgelist)
        del(grocery_store)
        del(grocery_edgelist)
        
        ## generate foci - education (every child in household is enrolled in a school with a dropout rate of )
        schools, school_edgelist = self._gen_schools(round(N/1000*n_school_per_1K_pop), enrollment_rate = enrollment_rate)
        self.node_attr["school"] = schools
        self.edgelist = self._outer_join_sum_two_edgelists(self.edgelist, school_edgelist)
        del(schools)
        del(school_edgelist)
       
    def initialize_infection(self):
        """
        initialize the original infection of the city
        """
        pass
                           
    
    def _gen_family(self, N, partner_prob,max_num_child):
        """
        Generate a family relation network
        Input: 
            N: number of nodes in the network
            partner_prob: the chance that the first household adult has a partner
        Return:
            adult_attribute (list(bool)): a list that speficies whether the member is a child or adult
            household_record (list(int)): a list that specifies which family an individual belongs to
            family_edgelist(list(int)): a dict that has the from_node, to_node, weight
        """

        adult_attribute = []
        household_record = []
        household_edgelist = dict()

        ## generate family child relationship
        current_node = 0
        current_household = -1 ## will start from 0

        while current_node < N:
            ## The first person must be an adult
            adult_attribute.append(True) 
            ## assign a new household number and add to the member list
            current_household += 1
            household_record.append(current_household)
            current_household_members = [current_node]

            ### match a potential 2nd adult in the household
            ### if there still exists unassigned individuals and matched successfully
            if current_node < (N-1) and \
                np.random.choice([False, True], p = [1-partner_prob, partner_prob]): 
                current_node += 1
                adult_attribute.append(True) 
                household_record.append(current_household)
                current_household_members.append(current_node)
            
            ### generate the number of children in the household if
            ### There are still unassigned individuals
            if current_node < (N-1):
                current_max_child_N = min(N - 1 - current_node, max_num_child)
                child_N  = np.random.choice(current_max_child_N+1)

                ### Assign children info if the household has children
                if child_N > 0:
                    child_adult_info, child_household_num, child_nodes = self._gen_child_attribute(child_N, current_household, current_node)
                    adult_attribute += child_adult_info
                    household_record += child_household_num
                    current_household_members += child_nodes

                ## advance the current_node pointer accordingly
                current_node += child_N

            
            
            ## if there are more than one household member update edgelist
            if len(current_household_members) > 1: 
                current_household_edgelist = self._gen_edgelist_with_prop(current_household_members)
                household_edgelist.update(current_household_edgelist)
            
            ## move to the next individual and household    
            current_node += 1
        
        
        return adult_attribute, household_record, household_edgelist

        
    def _gen_child_attribute(self, child_N, household_num, last_parent_num):
        """
        generate children info given number of children, household number household number
        input:
            child_N (int): number of children in the household
            household_num (int): numeric allocator for the household
            last_parent_number (int): the number of the last parent
        """
        return [False] * child_N, [household_num] * child_N, \
            list(range(last_parent_num+1, last_parent_num+ child_N + 1))
    
    def _gen_edgelist_with_prop(self, foci_nodelist, discount = 1.0):
        """
        generate an edgelist with a nodelist from a foci
        input:
            foci_nodelist (list(int)): a sorted/unsorted nodelist
        output:
            foci_edgelist: a dictionary that maps (from, to) to probability, from < to
        """

        foci_edgelist = dict()

        ### sort in order

        foci_nodelist.sort()
        tie_propensity = 1/len(foci_nodelist) * discount

        for i in range(len(foci_nodelist)-1):
            current_from = foci_nodelist[i]
            for j in range(i+1, len(foci_nodelist)):
                current_to = foci_nodelist[j]
                foci_edgelist[(current_from, current_to)] = tie_propensity
        
        return foci_edgelist
    
    
    def _gen_workplaces(self, N_company, decay_factor, employment_rate):
        """
        Assign adults in the dataset to the company:
        Input:
        N_company (int): Number of companies generated
        decay_factor (float): The k in the distribution of x^{-k}
        unemployment_rate (float): Unemployment rate of adult
        output:
        workplace_assignment (list(int)): assignment of people to workplace, None means not available (or unemployed)
        workplace_edgelist (dict): edgelist of workplace (from_node, to_node):contact_weight
        """

        node_list = self.node_attr["node"]
        adult_status = self.node_attr["adult"]

        ## Generate employment Status for everyone, will be filtered by adult status
        employment_status = np.random.choice([True, False], size = len(node_list), replace = True, p = [employment_rate, 1 - employment_rate])
        employment_status = np.logical_and(adult_status, employment_status)

        ## Generate workplace assignment
        company_probability = utils.power_law_prob(N_company, decay_factor)
        workplace_assignment = np.random.choice(np.arange(0, N_company, dtype = "float32"), replace = True,\
                                                size = len(node_list), p = company_probability)
        
        ## Assign nan to unemployed individuals
        workplace_assignment[np.logical_not(employment_status)] = np.nan

        ## Generate assignment edgelist
        workplace_edgelist = self._gen_foci_edgelist(node_list, workplace_assignment)
    

        return workplace_assignment, workplace_edgelist
    
    def _gen_grocery_stores(self, N_grocery, grocery_discount = 1.0):
        """
        Assign adults in the dataset to the grocery store:
        Input:
        N_grocery (int): Number of grocery stores generated
        output:
        grocery_assignment (list(int)): assignment of people to grocery stores, None means not available
        grocery_edgelist (dict): edgelist of workplace (from_node, to_node): contact_weight
        """

        node_list = self.node_attr["node"]
        adult_status = self.node_attr["adult"]
        household_list = self.node_attr["household"]

        grocery_status = [False] * len(node_list)

        for household in set(household_list):
            adult_index = np.where((adult_status) & (np.array(household_list) == household))[0]
            if len(adult_index) == 1:
                selected_adult = int(adult_index)
            else:
                selected_adult = int(np.random.choice(adult_index, size = 1))
                
            grocery_status[selected_adult] = True

        ## Generate workplace assignment
        grocery_assignment = np.random.choice(np.arange(0, N_grocery, dtype = "float32"), replace = True,\
                                                size = len(node_list))
        
        ## Assign nan to unemployed individuals
        grocery_assignment[np.logical_not(grocery_status)] = np.nan

        ## Generate assignment edgelist
        grocery_edgelist = self._gen_foci_edgelist(node_list, grocery_assignment, grocery_discount)
    
        return grocery_assignment, grocery_edgelist
    

    def _gen_schools(self, N_school, enrollment_rate):
        """
        Assign children in the dataset to the schools:
        Input:
        N_schools (int): Number of grocery schools generated
        non_enrollment_rate (float): 
        output:
        grocery_assignment (list(int)): assignment of people to grocery stores, None means not available
        grocery_edgelist (dict): edgelist of workplace (from_node, to_node): contact_weight
        """

        node_list = self.node_attr["node"]
        adult_status = self.node_attr["adult"]

        school_status = np.random.choice([True, False], \
                                         p = [enrollment_rate, 1-enrollment_rate], \
                                            size = len(node_list), replace = True)

        ## Generate workplace assignment
        school_assignment = np.random.choice(np.arange(0, N_school, dtype = "float32"), replace = True,\
                                                size = len(node_list))
        
        ## Assign nan to unenrolled individuals or adults
        school_assignment[np.logical_not(school_status) | np.array(adult_status)] = np.nan

        ## Generate assignment edgelist
        school_edgelist = self._gen_foci_edgelist(node_list, school_assignment)
    
        return school_assignment, school_edgelist
    
    def get_degree_from_dict_edgelist(self, weighted = False):
        """
        Get the degree of the nodes from an edgelist
        Output:
        degree_count: a node degree dictionary

        """
        if not weighted:
            from_nodes = [key[0] for key in self.edgelist.keys()]
            from_counts = Counter(from_nodes)
            del(from_nodes)
            
            to_nodes = [key[1] for key in self.edgelist.keys()]
            to_counts = Counter(to_nodes)
            del(to_nodes)
            ### Merge two dictionary
            degree_count = self._outer_join_sum_two_edgelists(from_counts, to_counts)
            del(from_counts, to_counts)
            degree_count = [int(degree_count.get(node, 0)) for node in self.nodelist]
        else:
            Graph = self.to_graph()
            
            degree_dict = Graph.degree(weight='weight')

            ### complete weighted degree here

            degree_count = [degree_dict[node] for node in self.nodelist]

        return degree_count
    
    
    
    def preferential_attachment(self, target_increase = 0.2, alpha = 4, beta = 1, gamma = 1, 
                                foci_discount = 1.0):
        """
        Update an edgelist until the total required contact is reached:
        target_increase: the level of increase over the top of the current total contact
        alpha: the skewness of the PA algorithm
        """

        

        ## calculate target total contact
        
        target_contact = np.sum(list(self.edgelist.values())) * (1 + target_increase)
        

        ## Update distribution if it is discounted
        if foci_discount != 1:
            for key in self.edgelist.keys():
                self.edgelist[key] = self.edgelist[key] * foci_discount

        ### generate new degree info
        degree_dist = np.array(self.get_degree_from_dict_edgelist(weighted = True))

        ### Edge-weight Distribution: Will be updated
        weight_dist = dict(Counter(self.edgelist.values()))

        current_contact = np.sum(list(self.edgelist.values()))
        print("Before Total Contact:", current_contact)


        ## weight distribution
        weight_probs = np.array(list(weight_dist.values()))
        weight_probs = weight_probs/np.sum(weight_probs)
        available_weights = list(weight_dist.keys())
        del(weight_dist)

        while current_contact < target_contact:
            current_index = np.random.choice(np.arange(len(self.nodelist)))
            ### Generate node choice probs
            probs = np.power(degree_dist + gamma , alpha) + beta
            probs = probs/np.sum(probs)

            selected_node_index = np.random.choice(np.arange(len(self.nodelist)), \
                                                   p = probs)
            ## In case a self loop was formed
            while selected_node_index == current_index:
                current_index = np.random.choice(np.arange(len(self.nodelist)))
                selected_node_index = np.random.choice(np.arange(len(self.nodelist)),  \
                                                   p = probs)
            
            selected_weight = np.random.choice(available_weights, p =  weight_probs)
            
            ### update total contact
            current_contact += selected_weight

            if self.nodelist[current_index] < self.nodelist[selected_node_index]: 
                small_node = self.nodelist[current_index]
                large_node = self.nodelist[selected_node_index]
            else:
                small_node = self.nodelist[selected_node_index]
                large_node = self.nodelist[current_index]
    

            ### update edgelist and degree distribution
            previous_weight = self.edgelist.get((small_node, large_node), 0)

            ### update node degree distribution
            degree_dist[current_index] += selected_weight
            degree_dist[selected_node_index] += selected_weight

            if previous_weight == 0:
                self.edgelist[(small_node, large_node)] = selected_weight
            else:
                ### if the current link already exists
                self.edgelist[(small_node, large_node)] += selected_weight
                
        print("After Total Contact:", current_contact)
    
        return None


    def _outer_join_sum_two_edgelists(self, edgelist_1, edgelist_2):
        """
        outer join two dictionaries and sum the overlapping elements
        Input:
        edgelist_1, edgelist_2: dictionary to be merged
        Output:
        a merged edgelist
        """
        
        return {key: edgelist_1.get(key, 0) + edgelist_2.get(key, 0) for key in set(edgelist_1) | set(edgelist_2)}



    def _gen_foci_edgelist(self, node_list, foci_assignment, discount = 1.0):
        """
        Generate an edgelist with node_list and their respective foci assignment
        Input:
            node_list (list): The list of nodes
            foci_assignment (list): The assignment of foci
        Output:
            foci_edgelist (dict): The edgelist of the foci
        """
        foci_set = set(foci_assignment)
        foci_edgelist = dict()
        ### Clear up nans
        foci_set = {foci for foci in foci_set if foci == foci}

        ### iterate over foci
        for foci in foci_set:
            nodes_in_foci = node_list[foci_assignment == foci]

            if len(nodes_in_foci) > 1:
                current_foci_edgelist = self._gen_edgelist_with_prop(nodes_in_foci)
                foci_edgelist.update(current_foci_edgelist)
        return foci_edgelist
    
    def to_graph(self):
        """
        Generate an edgelist dataframe given the current edgelist dictionary
        Output:
            Graph: networkx graph object
        """

        edgelist_df = {"from_nodes": [key[0] for key in self.edgelist.keys()],
                        "to_nodes": [key[1] for key in self.edgelist.keys()],
                        "weight": [weight for weight in self.edgelist.values()]}
        Graph = nx.from_pandas_edgelist(edgelist_df, source='from_nodes', target='to_nodes',
                                        edge_attr='weight')
        return Graph
    
    def to_df(self):
        """
        Generate an edgelist dataframe given the current edgelist dictionary
        Output:
            Graph: networkx graph object
        """

        edgelist_df = {"from_nodes": [key[0] for key in self.edgelist.keys()],
                        "to_nodes": [key[1] for key in self.edgelist.keys()],
                        "weight": [weight for weight in self.edgelist.values()]}
        edgelist_df = pd.DataFrame.from_dict(edgelist_df)
        return edgelist_df
    
def random_sub_graph(Graph, number = 1000, max_retain =10):
    """
    Generate a random subgraph from a given graph, retain the top x nodes with the highest node degree
    Input:
        graph (nx graph object): The graph being extracted from
        number: number of nodes in the subgraph
        max_retain: the number of nodes with highest degree
    Return:
        subgraph
    """
    degree_dict = Graph.degree(weight='weight')
    nodelist = np.array(list(Graph.nodes))
    degree_count = [degree_dict[node] for node in nodelist]
    if max_retain != 0:
        top_degrees = np.sort(degree_count)[-max_retain:]
        return_index = np.array([degree_dict[node] in top_degrees for node in nodelist])
        remaining_index = np.logical_not(return_index)
        
        return_nodes = nodelist[return_index]
        remaining_nodes = nodelist[remaining_index]
        selected_nodes = np.random.choice(remaining_nodes, size = number - max_retain, replace = False)
        selected_nodes = np.concatenate((return_nodes,selected_nodes ))
    else:
        selected_nodes = np.random.choice(nodelist, size = number, replace = False)
        
    subgraph = Graph.subgraph(selected_nodes)

    return subgraph
