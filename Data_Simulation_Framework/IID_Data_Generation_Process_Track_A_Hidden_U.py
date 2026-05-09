import pandas as pd
import networkx as nx
import numpy as np
import pickle
from math import ceil


def lin_func(x):
        return x


def relu_func(x):
    return np.maximum(0,x)


class Data_Generation_Process():
    
    def __init__(self,
                beta_lower_limit,
                betta_upper_limit_values,
                cont_noise,
                nr_nodes_values,
                edge_desnity_values,
                data_scale_values,
                num_samples,
                nonlinearities,
                track_a_hidden_unfair_nodes=1,
                track_a_proxy_nodes=None):
        
        self.beta_lower_limit=beta_lower_limit
        self.betta_upper_limit_values=betta_upper_limit_values
        self.cont_noise=cont_noise
        self.nr_nodes_values=nr_nodes_values
        self.edge_desnity_values=edge_desnity_values
        self.data_scale_values=data_scale_values
        self.num_samples=num_samples
        self.nonlinearities=nonlinearities
        self.track_a_hidden_unfair_nodes=track_a_hidden_unfair_nodes
        self.track_a_proxy_nodes=track_a_proxy_nodes

        super(Data_Generation_Process, self).__init__()
    
    # Generate ER graph and convert it to a DAG by orienting low id -> high id.
    def generate_dag(self,num_nodes,edge_density,seed=None):
        G = nx.gnp_random_graph(n=num_nodes, p=edge_density, seed=seed, directed=False)
        dag = nx.DiGraph()
        dag.add_nodes_from(G)
        dag.add_edges_from([(u, v, {}) for (u, v) in G.edges() if u < v])
        assert nx.is_directed_acyclic_graph(dag)
        return dag

    # Generate SF/Barabasi-Albert graph and convert it to a DAG by orienting low id -> high id.
    def generate_scale_free_dag(self,num_nodes,edges_per_new_node,seed=None):
        edges_per_new_node=int(edges_per_new_node)
        if edges_per_new_node < 1 or edges_per_new_node >= num_nodes:
            raise ValueError('edges_per_new_node must satisfy 1 <= m < num_nodes')

        G = nx.barabasi_albert_graph(n=num_nodes, m=edges_per_new_node, seed=seed)
        dag = nx.DiGraph()
        dag.add_nodes_from(G.nodes())
        dag.add_edges_from([(u, v) if u < v else (v, u) for (u, v) in G.edges()])
        assert nx.is_directed_acyclic_graph(dag)
        return dag

    def _scale_free_connectivity_from_edge_density(self,nr_nodes,edge_density):
        if edge_density >= 1:
            return min(int(edge_density),nr_nodes-1)
        return max(1,min(nr_nodes-1,int(ceil(edge_density*(nr_nodes-1)/2))))

    def select_hidden_unfairness_roles(self,num_nodes,num_hidden_unfair_nodes=None,num_proxy_nodes=None):
        if num_hidden_unfair_nodes is None:
            num_hidden_unfair_nodes=self.track_a_hidden_unfair_nodes

        if num_nodes < num_hidden_unfair_nodes + 3:
            raise ValueError('Hidden-U Track A requires at least hidden nodes + protected node + target node + one proxy node')

        hidden_unfair_nodes=[node for node in range(0,num_hidden_unfair_nodes)]
        protected_node=num_hidden_unfair_nodes
        target_node=num_nodes-1

        proxy_candidates=[node for node in range(num_nodes)
                          if node not in hidden_unfair_nodes
                          and node != protected_node
                          and node != target_node]
        if num_proxy_nodes is None:
            proxy_nodes=proxy_candidates
        else:
            if num_proxy_nodes < 1:
                raise ValueError('num_proxy_nodes must be positive')
            proxy_nodes=proxy_candidates[:min(num_proxy_nodes,len(proxy_candidates))]

        if len(proxy_nodes) == 0:
            raise ValueError('Hidden-U Track A requires at least one observed proxy/feature node')

        return {'Protected_Node':protected_node,
                'Target_Node':target_node,
                'Hidden_Unfair_Nodes':hidden_unfair_nodes,
                'Proxy_Nodes':proxy_nodes}

    def inject_hidden_unfairness_edges(self,G,track_a_roles):
        protected_node=track_a_roles['Protected_Node']
        target_node=track_a_roles['Target_Node']
        hidden_unfair_nodes=track_a_roles['Hidden_Unfair_Nodes']
        proxy_nodes=track_a_roles['Proxy_Nodes']

        full_graph=G.copy()

        # Hidden unfair variable path:
        # U -> A, U -> X, U -> Y
        for hidden_node in hidden_unfair_nodes:
            full_graph.add_edge(hidden_node,protected_node)
            full_graph.add_edge(hidden_node,target_node)
            for proxy_node in proxy_nodes:
                full_graph.add_edge(hidden_node,proxy_node)

        # Observed causal path:
        # A -> X, A -> Y, X -> Y
        full_graph.add_edge(protected_node,target_node)
        for proxy_node in proxy_nodes:
            full_graph.add_edge(protected_node,proxy_node)
            full_graph.add_edge(proxy_node,target_node)

        assert nx.is_directed_acyclic_graph(full_graph)
        return full_graph

    def observed_nodes_from_roles(self,num_nodes,track_a_roles):
        hidden_nodes=set(track_a_roles['Hidden_Unfair_Nodes'])
        return [node for node in range(num_nodes) if node not in hidden_nodes]

    def remove_hidden_nodes_from_matrix(self,matrix,observed_nodes):
        return matrix[np.ix_(observed_nodes,observed_nodes)]

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
                 G,
                 W,
                 n,
                 causal_transformation,
                 graph_type,
                 noise_scale=None):

        d = W.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale

        if np.isinf(n):
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X

        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        X = np.zeros([n, d])

        for j in ordered_vertices:
            parents = list(G.predecessors(j))
            X[:, j] = self._simulate_single_equation(X[:, parents],
                                                W[parents, j],
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

    def _connectivity_list_for_graph(self,graph_type_key,nr_nodes,avg_number_edges):
        if graph_type_key=='ER':
            return self.edge_desnity_values

        if graph_type_key=='SF':
            if avg_number_edges is not None:
                connectivity_list=[]
                for ed_dns in [0.2,0.3,0.4]:
                    if ed_dns in avg_number_edges['Nodes_'+str(nr_nodes)].keys():
                        connectivity_list.append(avg_number_edges['Nodes_'+str(nr_nodes)][ed_dns]['k'])
                return connectivity_list

            return [self._scale_free_connectivity_from_edge_density(nr_nodes=nr_nodes,
                                                                    edge_density=edge_density)
                    for edge_density in self.edge_desnity_values]

        raise ValueError("graph_type must be either 'ER' or 'SF'")

    def _generate_base_graph(self,graph_type_key,nr_nodes,connectivity):
        if graph_type_key=='ER':
            return self.generate_dag(num_nodes=nr_nodes,edge_density=connectivity)

        return self.generate_scale_free_dag(num_nodes=nr_nodes,edges_per_new_node=connectivity)
    
    def large_scale_simulation(self,
                          graph_type,
                          avg_number_edges=None,
                          num_hidden_unfair_nodes=None,
                          num_proxy_nodes=None):

        graph_type_key=str(graph_type).upper()
        if num_hidden_unfair_nodes is None:
            num_hidden_unfair_nodes=self.track_a_hidden_unfair_nodes
        if num_proxy_nodes is None:
            num_proxy_nodes=self.track_a_proxy_nodes
    
        seed_runs=[]
        nr_nodes_array=[]
        nr_nodes_full_array=[]
        connectivity_array=[]
        function_transformation_array=[]
        data_scale_array=[]
        beta_upper_array=[]
        number_edges_array=[]
        number_edges_full_array=[]

        protected_node_array=[]
        target_node_array=[]
        hidden_unfair_nodes_array=[]
        proxy_nodes_array=[]
        observed_nodes_array=[]

        frames=[]
        true_causal_matrices=[]
        true_weighted_causal_matrices=[]

        full_frames=[]
        full_true_causal_matrices=[]
        full_true_weighted_causal_matrices=[]

        for seed_run in range(0,10):
            for nr_nodes in self.nr_nodes_values:
                connectivity_list=self._connectivity_list_for_graph(graph_type_key=graph_type_key,
                                                                    nr_nodes=nr_nodes,
                                                                    avg_number_edges=avg_number_edges)

                for connectivity in connectivity_list:
                    for function_transformation in self.nonlinearities:
                        for beta_upper_limit in self.betta_upper_limit_values:
                            for data_scale in self.data_scale_values:
                                track_a_roles=self.select_hidden_unfairness_roles(num_nodes=nr_nodes,
                                                                                  num_hidden_unfair_nodes=num_hidden_unfair_nodes,
                                                                                  num_proxy_nodes=num_proxy_nodes)
                                observed_nodes=self.observed_nodes_from_roles(num_nodes=nr_nodes,
                                                                              track_a_roles=track_a_roles)

                                base_graph=self._generate_base_graph(graph_type_key=graph_type_key,
                                                                     nr_nodes=nr_nodes,
                                                                     connectivity=connectivity)
                                full_graph=self.inject_hidden_unfairness_edges(G=base_graph,
                                                                               track_a_roles=track_a_roles)
                                full_edge_list=list(full_graph.edges)

                                full_adjacency_matrix=np.zeros(shape=(nr_nodes,nr_nodes))
                                for edge in full_edge_list:
                                    full_adjacency_matrix[edge[0]][edge[1]]=1
                                full_true_causal_matrices.append(full_adjacency_matrix)

                                betas=np.array([self.sample_beta(beta_lower_limit=self.beta_lower_limit,
                                                                 beta_upper_limit=beta_upper_limit)
                                                for bt in range(0,nr_nodes*nr_nodes)])
                                full_weighted_adjacency=np.reshape(betas,newshape=(nr_nodes,nr_nodes))*full_adjacency_matrix
                                full_weighted_adjacency=np.where(full_weighted_adjacency==0.0,0.0,full_weighted_adjacency)
                                full_true_weighted_causal_matrices.append(full_weighted_adjacency)

                                full_dataframe=self.simulate_sem(G=full_graph,
                                                            W=full_weighted_adjacency, 
                                                            n=self.num_samples,
                                                            causal_transformation=function_transformation,
                                                            graph_type=graph_type,
                                                            noise_scale=self.cont_noise)
                                full_dataframe=pd.DataFrame(full_dataframe,
                                                            columns=[col_index for col_index in range(0,full_dataframe.shape[1])])

                                if data_scale=='standardized':
                                    full_dataframe=(full_dataframe-full_dataframe.mean(axis=0))/full_dataframe.std(axis=0)

                                # This is the key hidden-variable step: U exists in the full SCM,
                                # but U columns are removed from the observed table seen by models.
                                observed_dataframe=full_dataframe.loc[:,observed_nodes].copy()
                                observed_dataframe.columns=[col_index for col_index in range(0,observed_dataframe.shape[1])]

                                observed_adjacency_matrix=self.remove_hidden_nodes_from_matrix(matrix=full_adjacency_matrix,
                                                                                               observed_nodes=observed_nodes)
                                observed_weighted_adjacency=self.remove_hidden_nodes_from_matrix(matrix=full_weighted_adjacency,
                                                                                                 observed_nodes=observed_nodes)

                                frames.append(observed_dataframe)
                                true_causal_matrices.append(observed_adjacency_matrix)
                                true_weighted_causal_matrices.append(observed_weighted_adjacency)
                                full_frames.append(full_dataframe)

                                seed_runs.append(seed_run)
                                nr_nodes_array.append(len(observed_nodes))
                                nr_nodes_full_array.append(nr_nodes)
                                connectivity_array.append(connectivity)
                                number_edges_array.append(int(np.sum(observed_adjacency_matrix)))
                                number_edges_full_array.append(len(full_edge_list))
                                beta_upper_array.append(beta_upper_limit)
                                data_scale_array.append(data_scale)
                                protected_node_array.append(track_a_roles['Protected_Node'])
                                target_node_array.append(track_a_roles['Target_Node'])
                                hidden_unfair_nodes_array.append(track_a_roles['Hidden_Unfair_Nodes'])
                                proxy_nodes_array.append(track_a_roles['Proxy_Nodes'])
                                observed_nodes_array.append(observed_nodes)
                                
                                if function_transformation==[(1.0,lin_func)]:
                                    function_transformation_array.append('Linear_100%')
                                elif function_transformation==[(0.5,lin_func),(0.5,relu_func)]:
                                    function_transformation_array.append('Linear_ReLU_50%')
                                elif function_transformation==[(0.3,lin_func),(0.7,relu_func)]:
                                    function_transformation_array.append('Linear_30%_ReLU_70%')
                                else:
                                    function_transformation_array.append('Linear_10%_ReLU_90%')

        all_datasets_frame=pd.DataFrame({'Seed_Run':np.array(seed_runs),
                                    'Number_Nodes':np.array(nr_nodes_array),
                                    'Number_Nodes_Full':np.array(nr_nodes_full_array),
                                    ('Edge_Density' if graph_type_key=='ER' else 'K'):np.array(connectivity_array),
                                    'Number_Edges':np.array(number_edges_array),
                                    'Number_Edges_Full':np.array(number_edges_full_array),
                                    'Transformation_Function':np.array(function_transformation_array),
                                    'Beta_Upper_Limit':np.array(beta_upper_array),
                                    'Data_Scale':np.array(data_scale_array),
                                    'Graph_Type':np.array([graph_type]*len(data_scale_array)),
                                    'Protected_Node_Full':np.array(protected_node_array),
                                    'Target_Node_Full':np.array(target_node_array),
                                    'Hidden_Unfair_Nodes_Full':hidden_unfair_nodes_array,
                                    'Proxy_Nodes_Full':proxy_nodes_array,
                                    'Observed_Nodes_Full':observed_nodes_array})

        return [all_datasets_frame,
               true_causal_matrices,
               true_weighted_causal_matrices,
               frames,
               full_true_causal_matrices,
               full_true_weighted_causal_matrices,
               full_frames]
    
    def save_data(self,
                    frames_descriptions,
                    true_causal_matrices,
                    true_weighted_causal_matrices,
                    frames,
                    nonlinear_pattern,
                    graph_type,
                    sample_size,
                    save_path):

        data_by_nodes={}

        for frame_index in range(0,frames_descriptions.shape[0]):
            frame_description=frames_descriptions.loc[[frame_index]].values.tolist()[0]

            current_adjacency_matrix=true_causal_matrices[frame_index]
            current_weighted_adjacency=true_weighted_causal_matrices[frame_index]
            current_dataframe=frames[frame_index]
            
            if sample_size=='Small_Sample_Size':
                small_sample_size=int(current_dataframe.shape[0]/10)
                current_dataframe=current_dataframe.sample(small_sample_size)

            node_count=int(current_dataframe.shape[1])
            if node_count not in data_by_nodes:
                data_by_nodes[node_count]=[]

            data_by_nodes[node_count].append([frame_description,
                                              current_adjacency_matrix,
                                              current_weighted_adjacency,
                                              current_dataframe])

        for node_count,node_data in data_by_nodes.items():
            if len(node_data)!=0:
                with open(save_path+graph_type+'_'+sample_size+'_Datasets_'+nonlinear_pattern+'_'+str(node_count)+'_nodes.pkl', 'wb') as f:
                    pickle.dump(node_data,f)
