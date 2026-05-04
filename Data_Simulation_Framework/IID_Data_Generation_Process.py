import pandas as pd
import networkx as nx
import numpy as np
import igraph as ig
import pickle
from math import ceil

def lin_func(x):
        return x
#define linear causal relationship

def relu_func(x):
    return np.maximum(0,x)
class Data_Generation_Process():
#define non-linear causal relationship
    
    def __init__(self,
                beta_lower_limit,
                betta_upper_limit_values,
                cont_noise,
                nr_nodes_values,
                edge_desnity_values,
                data_scale_values,
                num_samples,
                nonlinearities):
        
        self.beta_lower_limit=beta_lower_limit
        self.betta_upper_limit_values=betta_upper_limit_values
        self.cont_noise=cont_noise
        self.nr_nodes_values=nr_nodes_values
        self.edge_desnity_values=edge_desnity_values
        self.data_scale_values=data_scale_values
        self.num_samples=num_samples
        self.nonlinearities=nonlinearities

        super(Data_Generation_Process, self).__init__()
    
    #generate ER graph
    def generate_dag(self,num_nodes,edge_density,seed=None):
        #edge_density is the probability of having a edge between any two nodes
        # Generate graph using networkx package
        G = nx.gnp_random_graph(n=num_nodes, p=edge_density, seed=seed, directed=False)
        # Convert generated graph to DAG
        #Eg: num_nodes = 10, edge_density = 0.2, means
        #among 10 nodes, there’s a roughly 20% chance that any two nodes will be connected to each other.
        dag = nx.DiGraph()
        dag.add_nodes_from(G)
        dag.add_edges_from([(u, v, {}) for (u, v) in G.edges() if u < v])
        assert nx.is_directed_acyclic_graph(dag)
        #make sure we generate a directed_acyclic_graph(DAG)
        return dag
    
    def sample_beta(self,beta_lower_limit,beta_upper_limit):
        if np.random.randint(0,2) == 0:
            return np.random.uniform(-beta_upper_limit,-beta_lower_limit,size=1)[0]
        else:
            return np.random.uniform(beta_lower_limit, beta_upper_limit,size=1)[0]

    
    def apply_transformation(self,dot_product,transformation):
        transformation_func_index=np.random.choice(a=[func_index for func_index in range(0,len(transformation))],
                       p=[func[0] for func in transformation])
        return transformation[transformation_func_index][1](dot_product)
    
    def _simulate_single_equation(self,X, w, scale,causal_transformation,n):
            """X: [n, num of parents], w: [num of parents], x: [n]"""
            if len(w)>0:
                z = np.random.normal(scale=scale, size=n)
                x =self.apply_transformation(dot_product=X @ w,transformation=causal_transformation)+ z
            else:
                z = np.random.normal(scale=scale, size=n)
                x=z
            return x
    
    def simulate_sem(self,
                 G,#graph object: either networkx or igraph
                 W,#weighted ajdacency matrix
                 n,#number of samples:
                 causal_transformation,#linear or nonlinear transformation of parents into children nodes:
                 graph_type,
                 noise_scale=None):
        """Simulate samples from linear SEM with specified type of noise.
        For uniform, noise z ~ uniform(-a, a), where a = noise_scale.
        Args:
            W (np.ndarray): [d, d] weighted adj matrix of DAG
            n (int): num of samples, n=inf mimics population risk
            sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
            noise_scale (np.ndarray): scale parameter of additive noise, default all ones
        Returns:
            X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
        """

        d = W.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale

        if np.isinf(n):  # population risk for linear gauss SEM
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X


        if graph_type=='ER':
            ordered_vertices = list(nx.topological_sort(G))
        else:
            ordered_vertices=G.topological_sorting()

        assert len(ordered_vertices) == d
        X = np.zeros([n, d])

        for j in ordered_vertices:
            parents = list(G.predecessors(j))
            #print('Child index: ',j,' Parent Indices: ',parents)
            X[:, j] = self._simulate_single_equation(X[:, parents], 
                                                W[parents, j],#rows in weighted adjacency matrix=causes, column=effect 
                                                #(the current j, menaing child node)
                                                scale_vec[j],
                                                causal_transformation=causal_transformation,
                                                n=n)
        return X
    
    def get_avg_number_edges_ER_graph(self,frames_descriptions,
                                      save_path_edge_mapping):
    
        avg_number_edges={}

        avg_number_edges['Nodes_10']={0.2:[],
                                    0.3:[],
                                    0.4:[]}

        avg_number_edges['Nodes_20']={0.2:[],
                                    0.3:[],
                                    0.4:[]}

        avg_number_edges['Nodes_50']={0.2:[],
                                    0.3:[],
                                    0.4:[]}

        avg_number_edges['Nodes_100']={0.2:[],
                                    0.3:[],
                                    0.4:[]}

        for idx in range(0,frames_descriptions.shape[0]):
            if frames_descriptions.iloc[idx,1]==10:
                avg_number_edges['Nodes_10'][frames_descriptions.iloc[idx,2]].append(frames_descriptions.iloc[idx,3])
            elif frames_descriptions.iloc[idx,1]==20:
                avg_number_edges['Nodes_20'][frames_descriptions.iloc[idx,2]].append(frames_descriptions.iloc[idx,3])
            elif frames_descriptions.iloc[idx,1]==50:
                avg_number_edges['Nodes_50'][frames_descriptions.iloc[idx,2]].append(frames_descriptions.iloc[idx,3])
            else:
                avg_number_edges['Nodes_100'][frames_descriptions.iloc[idx,2]].append(frames_descriptions.iloc[idx,3])

        for node_key in avg_number_edges.keys():
            avg_number_edges[node_key][0.2]={'e':int(ceil(np.mean(avg_number_edges[node_key][0.2]))),
                                            'd':int(node_key.split('_')[1]),
                                            'k':int(ceil(np.mean(avg_number_edges[node_key][0.2])/int(node_key.split('_')[1])))}

            avg_number_edges[node_key][0.3]={'e':int(ceil(np.mean(avg_number_edges[node_key][0.3]))),
                                            'd':int(node_key.split('_')[1]),
                                            'k':int(ceil(np.mean(avg_number_edges[node_key][0.3])/int(node_key.split('_')[1])))}

            avg_number_edges[node_key][0.4]={'e':int(ceil(np.mean(avg_number_edges[node_key][0.4]))),
                                            'd':int(node_key.split('_')[1]),
                                            'k':int(ceil(np.mean(avg_number_edges[node_key][0.4])/int(node_key.split('_')[1])))}

        avg_number_edges['Nodes_10'][0.4]['k']=avg_number_edges['Nodes_10'][0.4]['k']+1

        
        with open(save_path_edge_mapping, 'wb') as f:
                pickle.dump(avg_number_edges,f)

        return avg_number_edges
    
    def large_scale_simulation(self,
                          graph_type,
                          avg_number_edges=None):
        
    
        seed_runs=[]
        nr_nodes_array=[]
        connectivity_array=[]
        function_transformation_array=[]
        data_scale_array=[]
        beta_upper_array=[]

        #To compute from already sampled graphs:
        number_edges_array=[]
    
        #Arrays for saving all data eventually:
        frames=[]
        true_causal_matrices=[]
        #true_causal_DAGs=[]
        true_weighted_causal_matrices=[]


        
        #Simulate each dataset & graph 10 times:
        for seed_run in range(0,10):
            #Define the number of nodes to use:
            for nr_nodes in self.nr_nodes_values:
                #Define the connectivity: edge density for ER graphs using networkx, k based on ER graphs using igraph
                if graph_type=='ER':
                    connectivity_list=self.edge_desnity_values
                else:
                    connectivity_list=[]
                    for ed_dns in [0.2,0.3,0.4]:
                        if ed_dns in avg_number_edges['Nodes_'+str(nr_nodes)].keys():
                            connectivity_list.append(avg_number_edges['Nodes_'+str(nr_nodes)][ed_dns]['k'])
                        
                    #connectivity_list=[avg_number_edges['Nodes_'+str(nr_nodes)][0.2]['k'],
                    #                   avg_number_edges['Nodes_'+str(nr_nodes)][0.3]['k'],
                    #                   avg_number_edges['Nodes_'+str(nr_nodes)][0.4]['k']]
                    
                for connectivity in connectivity_list:
                    #Define non-linearities:
                    for function_transformation in self.nonlinearities:
                        #Define beta upper limits:
                        for beta_upper_limit in self.betta_upper_limit_values:
                            #Define data scale:
                            for data_scale in self.data_scale_values:
                                #Save seeds run before the actual simulation:
                                seed_runs.append(seed_run)

                                #CAUSAL DAG STRUCTURE; NODES & CONNECTIVITY: ########################################
                                if graph_type=='ER':
                                    current_graph=self.generate_dag(num_nodes=nr_nodes,
                                                              edge_density=connectivity)
                                    edge_list=list(current_graph.edges)
                                else: 
                                    current_graph=ig.Graph.Barabasi(n=nr_nodes,m=connectivity, directed=True)
                                    edge_list=current_graph.get_edgelist()
                                #Save sampled DAGs:
                                #true_causal_DAGs.append(current_graph)
                                #Save number of nodes, connectivity & number of edges:
                                nr_nodes_array.append(nr_nodes)
                                connectivity_array.append(connectivity)
                                number_edges_array.append(len(edge_list))
                                ######################################################################################


                                #TRUE BINARY ADJACENCY MATRIX: #######################################################
                                current_adjacency_matrix=np.zeros(shape=(nr_nodes,nr_nodes))
                                for edge in edge_list:
                                    current_adjacency_matrix[edge[0]][edge[1]]=1
                                #Save true binary adjacency matrix:
                                true_causal_matrices.append(current_adjacency_matrix)
                                ######################################################################################


                                #TRUE WEIGHTED ADJACENCY MATRIX & BETAS: #############################################
                                betas=np.array([self.sample_beta(beta_lower_limit=self.beta_lower_limit,beta_upper_limit=beta_upper_limit) for bt in range(0,nr_nodes*nr_nodes)])
                                weighted_adjacency=np.reshape(betas,newshape=(nr_nodes,nr_nodes))*current_adjacency_matrix
                                weighted_adjacency=np.where(weighted_adjacency==0.0,0.0,weighted_adjacency)
                                #Save weighted adjacency matrix & betta upper limit:
                                true_weighted_causal_matrices.append(weighted_adjacency)
                                beta_upper_array.append(beta_upper_limit)
                                ######################################################################################


                                #DATASET SIMULATED BASED ON GRAPH STRUCTURE: #########################################
                                current_dataframe=self.simulate_sem(G=current_graph,
                                                               W=weighted_adjacency, 
                                                               n=self.num_samples,#sem_type: always gaussian
                                                               causal_transformation=function_transformation,
                                                               graph_type=graph_type,
                                                               noise_scale=self.cont_noise)
                                current_dataframe=pd.DataFrame(current_dataframe,columns=[col_index for col_index in range(0,current_dataframe.shape[1])])
                                #Standardize if necessary:
                                if data_scale=='standardized':
                                    current_dataframe=(current_dataframe-current_dataframe.mean(axis=0))/current_dataframe.std(axis=0)
                                    #scaler=StandardScaler()
                                    #scaler.fit(current_dataframe)
                                    #current_dataframe=pd.DataFrame(scaler.transform(current_dataframe),columns=current_dataframe.columns)
                                frames.append(current_dataframe)
                                #Save sampled data scale:
                                data_scale_array.append(data_scale)
                                
                                #Save linear-nonlinear patterns as strings:
                                if function_transformation==[(1.0,lin_func)]:
                                    function_transformation_array.append('Linear_100%')
                                #ReLU string conditions:
                                elif function_transformation==[(0.5,lin_func),(0.5,relu_func)]:
                                    function_transformation_array.append('Linear_ReLU_50%')
                                elif function_transformation==[(0.3,lin_func),(0.7,relu_func)]:
                                    function_transformation_array.append('Linear_30%_ReLU_70%')
                                else:# function_transformation==[(0.1,lin_func),(0.9,relu_func)]:
                                    function_transformation_array.append('Linear_10%_ReLU_90%')
                                #######################################################################################


        all_datasets_frame=pd.DataFrame({'Seed_Run':np.array(seed_runs),
                                    'Number_Nodes':np.array(nr_nodes_array),
                                    ('Edge_Density' if graph_type=='ER' else 'K'):np.array(connectivity_array),   
                                    'Number_Edges':np.array(number_edges_array),
                                    'Transformation_Function':np.array(function_transformation_array),
                                    'Beta_Upper_Limit':np.array(beta_upper_array),
                                    'Data_Scale':np.array(data_scale_array),
                                    'Graph_Type':np.array([graph_type]*len(data_scale_array))}) 
        return [all_datasets_frame,true_causal_matrices,
               true_weighted_causal_matrices,frames]
    
    
    def save_data(self,
                    frames_descriptions,
                    true_causal_matrices,
                    true_weighted_causal_matrices,
                    frames,
                    nonlinear_pattern,
                    graph_type,
                    sample_size,
                    save_path):
        
        #Save sampled simulations: save path depends on large vs small scale, nonlinear pattern, graph type
        data_10_nodes=[]
        data_20_nodes=[]
        data_50_nodes=[]
        data_100_nodes=[]


        for frame_index in range(0,frames_descriptions.shape[0]):
            frame_description=frames_descriptions.loc[[frame_index]].values.tolist()[0]

            current_adjacency_matrix=true_causal_matrices[frame_index]

            current_weighted_adjacency=true_weighted_causal_matrices[frame_index]

            current_dataframe=frames[frame_index]
            
            if sample_size=='Small_Sample_Size':
                small_sample_size=int(current_dataframe.shape[0]/10)
                current_dataframe=current_dataframe.sample(small_sample_size)

            
            if frame_description[1]==10:#graph with 10 nodes
                data_10_nodes.append([frame_description,
                                        current_adjacency_matrix,
                                        current_weighted_adjacency,
                                        current_dataframe])

            elif frame_description[1]==20:#graph with 20 nodes
                data_20_nodes.append([frame_description,
                                        current_adjacency_matrix,
                                        current_weighted_adjacency,
                                        current_dataframe])

            elif frame_description[1]==50:#graph with 50 nodes
                data_50_nodes.append([frame_description,
                                        current_adjacency_matrix,
                                        current_weighted_adjacency,
                                        current_dataframe])

            else:#graph with 100 nodes
                data_100_nodes.append([frame_description,
                                        current_adjacency_matrix,
                                        current_weighted_adjacency,
                                        current_dataframe])

        if len(data_10_nodes)!=0:
            with open(save_path+graph_type+'_'+sample_size+'_Datasets_'+nonlinear_pattern+'_10_nodes.pkl', 'wb') as f:
                pickle.dump(data_10_nodes,f)

        if len(data_20_nodes)!=0:
            with open(save_path+graph_type+'_'+sample_size+'_Datasets_'+nonlinear_pattern+'_20_nodes.pkl', 'wb') as f:
                pickle.dump(data_20_nodes,f)

        if len(data_50_nodes)!=0:
            with open(save_path+graph_type+'_'+sample_size+'_Datasets_'+nonlinear_pattern+'_50_nodes.pkl', 'wb') as f:
                pickle.dump(data_50_nodes,f)

        if len(data_100_nodes)!=0:
            with open(save_path+graph_type+'_'+sample_size+'_Datasets_'+nonlinear_pattern+'_100_nodes.pkl', 'wb') as f:
                pickle.dump(data_100_nodes,f)