import pandas as pd
import networkx as nx
import numpy as np
import igraph as ig
import pickle
from math import ceil

# 中文：这个文件是 IID 数据模拟流程的带注解版本，代码主体复制自
#       IID_Data_Generation_Process.py，重点补充每个步骤在因果数据生成中的作用。
# English: This annotated version keeps the original implementation intact and
#          explains the main steps used to generate IID causal simulation data.

def lin_func(x):
        return x
# 中文：线性变换函数，父节点的加权和会原样传给子节点。
# English: Linear transformation; the weighted parent signal is passed through unchanged.

def relu_func(x):
    return np.maximum(0,x)
class Data_Generation_Process():
# 中文：非线性变换函数 ReLU，把负值截断为 0，用来模拟非线性因果关系。
# English: Nonlinear ReLU transformation; negative values are clipped to 0 to model nonlinear effects.
    
    def __init__(self,
                beta_lower_limit,
                betta_upper_limit_values,
                cont_noise,
                nr_nodes_values,
                edge_desnity_values,
                data_scale_values,
                num_samples,
                nonlinearities):
        
        # 中文：保存全局模拟配置，后续 large_scale_simulation 会遍历这些候选值。
        # English: Store global simulation settings; large_scale_simulation iterates over these values.
        self.beta_lower_limit=beta_lower_limit
        self.betta_upper_limit_values=betta_upper_limit_values
        self.cont_noise=cont_noise
        self.nr_nodes_values=nr_nodes_values
        self.edge_desnity_values=edge_desnity_values
        self.data_scale_values=data_scale_values
        self.num_samples=num_samples
        self.nonlinearities=nonlinearities

        super(Data_Generation_Process, self).__init__()
    
    # 中文：生成 Erdos-Renyi 随机图，并把它转换成 DAG。
    # English: Generate an Erdos-Renyi random graph and convert it into a DAG.
    def generate_dag(self,num_nodes,edge_density,seed=None):
        # 中文：edge_density 表示任意两个节点之间出现边的概率。
        # English: edge_density is the probability that any pair of nodes is connected.
        # 中文：先用 networkx 生成无向随机图。
        # English: First generate an undirected random graph with networkx.
        G = nx.gnp_random_graph(n=num_nodes, p=edge_density, seed=seed, directed=False)
        # 中文：再把无向图转换成有向无环图。这里规定只保留 u < v 的方向，所以不会产生环。
        # English: Convert the undirected graph into a DAG by orienting edges from lower to higher node ids.
        # 中文：例子：num_nodes=10 且 edge_density=0.2，表示任意两点约有 20% 概率相连。
        # English: Example: num_nodes=10 and edge_density=0.2 means each node pair has about a 20% chance to connect.
        #Eg: num_nodes = 10, edge_density = 0.2, means
        #among 10 nodes, there’s a roughly 20% chance that any two nodes will be connected to each other.
        dag = nx.DiGraph()
        dag.add_nodes_from(G)
        dag.add_edges_from([(u, v, {}) for (u, v) in G.edges() if u < v])
        assert nx.is_directed_acyclic_graph(dag)
        # 中文：断言用于确认最终结果确实是 DAG。
        # English: The assertion verifies that the final graph is acyclic and directed.
        return dag
    
    def sample_beta(self,beta_lower_limit,beta_upper_limit):
        # 中文：随机抽样一条因果边的权重 beta，符号正负各有 50% 概率。
        # English: Randomly sample one causal edge weight beta, with equal chance of positive or negative sign.
        if np.random.randint(0,2) == 0:
            # 中文：抽取负向因果效应，范围为 [-upper, -lower]。
            # English: Sample a negative causal effect from [-upper, -lower].
            return np.random.uniform(-beta_upper_limit,-beta_lower_limit,size=1)[0]
        else:
            # 中文：抽取正向因果效应，范围为 [lower, upper]。
            # English: Sample a positive causal effect from [lower, upper].
            return np.random.uniform(beta_lower_limit, beta_upper_limit,size=1)[0]

    
    def apply_transformation(self,dot_product,transformation):
        # 中文：根据给定概率选择一个变换函数，例如 100% 线性，或线性/ReLU 混合。
        # English: Choose a transformation function by its configured probability, e.g. all linear or mixed linear/ReLU.
        transformation_func_index=np.random.choice(a=[func_index for func_index in range(0,len(transformation))],
                       p=[func[0] for func in transformation])
        # 中文：把父节点加权和输入到被选中的变换函数中。
        # English: Apply the selected transformation to the weighted parent signal.
        return transformation[transformation_func_index][1](dot_product)
    
    def _simulate_single_equation(self,X, w, scale,causal_transformation,n):
            """X: [n, num of parents], w: [num of parents], x: [n]"""
            # 中文：单个节点的结构方程：子节点 = f(父节点矩阵 @ 权重) + 高斯噪声。
            # English: Single-node structural equation: child = f(parent matrix @ weights) + Gaussian noise.
            if len(w)>0:
                # 中文：如果当前节点有父节点，先生成该节点的独立噪声项。
                # English: If the node has parents, generate its independent noise term.
                z = np.random.normal(scale=scale, size=n)
                # 中文：X @ w 是所有样本上的父节点加权和，再经过线性或非线性变换。
                # English: X @ w is the weighted parent signal for all samples, then transformed linearly or nonlinearly.
                x =self.apply_transformation(dot_product=X @ w,transformation=causal_transformation)+ z
            else:
                # 中文：如果没有父节点，该节点就是外生变量，只由噪声生成。
                # English: If there are no parents, the node is exogenous and generated only from noise.
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
        # 中文：d 是变量/节点个数，也就是加权邻接矩阵的维度。
        # English: d is the number of variables/nodes, taken from the weighted adjacency matrix size.
        if noise_scale is None:
            # 中文：如果没有指定噪声尺度，每个节点默认使用标准差 1。
            # English: If no noise scale is given, every node uses standard deviation 1.
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            # 中文：如果输入是单个数值，则所有节点共享同一个噪声尺度。
            # English: If a scalar is given, all nodes share the same noise scale.
            scale_vec = noise_scale * np.ones(d)
        else:
            # 中文：如果输入是向量，则它必须为每个节点提供一个噪声尺度。
            # English: If a vector is given, it must provide one noise scale per node.
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale

        if np.isinf(n):  # population risk for linear gauss SEM
            # 中文：n 为无穷时，返回理论协方差相关的总体风险表示，而不是有限样本数据。
            # English: When n is infinite, return a population-risk representation instead of finite samples.
            # 中文：构造方式使得 1/d X'X 等于真实协方差。
            # English: The construction makes 1/d X'X equal to the true covariance.
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X


        if graph_type=='ER':
            # 中文：ER 图使用 networkx，因此用 networkx 的拓扑排序。
            # English: ER graphs are networkx graphs, so use networkx topological sorting.
            ordered_vertices = list(nx.topological_sort(G))
        else:
            # 中文：非 ER 图这里使用 igraph，因此调用 igraph 的拓扑排序接口。
            # English: Non-ER graphs use igraph here, so call igraph's topological sorting method.
            ordered_vertices=G.topological_sorting()

        assert len(ordered_vertices) == d
        # 中文：按样本数 n 和变量数 d 初始化数据矩阵，每列对应一个节点变量。
        # English: Initialize the data matrix with n samples and d variables; each column is one node.
        X = np.zeros([n, d])

        for j in ordered_vertices:
            # 中文：拓扑顺序保证父节点已经先被模拟出来，子节点可以依赖它们。
            # English: Topological order ensures parents are simulated before each child node.
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
    
        # 中文：这个函数根据已经模拟出来的 ER 图统计平均边数，并转换成 igraph Barabasi 图需要的 k。
        # English: This function summarizes average ER edge counts and converts them into k values for igraph Barabasi graphs.
        avg_number_edges={}

        # 中文：为不同节点规模和 ER 边密度建立容器，先收集每次模拟中的边数。
        # English: Create containers for each node size and ER density, collecting edge counts from previous simulations.
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
            # 中文：frames_descriptions 的第 2 列是节点数，第 3 列是边密度，第 4 列是边数。
            # English: In frames_descriptions, column 2 stores node count, column 3 edge density, and column 4 edge count.
            if frames_descriptions.iloc[idx,1]==10:
                avg_number_edges['Nodes_10'][frames_descriptions.iloc[idx,2]].append(frames_descriptions.iloc[idx,3])
            elif frames_descriptions.iloc[idx,1]==20:
                avg_number_edges['Nodes_20'][frames_descriptions.iloc[idx,2]].append(frames_descriptions.iloc[idx,3])
            elif frames_descriptions.iloc[idx,1]==50:
                avg_number_edges['Nodes_50'][frames_descriptions.iloc[idx,2]].append(frames_descriptions.iloc[idx,3])
            else:
                avg_number_edges['Nodes_100'][frames_descriptions.iloc[idx,2]].append(frames_descriptions.iloc[idx,3])

        for node_key in avg_number_edges.keys():
            # 中文：对每个节点规模和边密度取平均边数 e，并计算 k=ceil(e/d)。
            # English: For each node size and density, compute mean edge count e and k=ceil(e/d).
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
        # 中文：这里对 10 节点、0.4 密度的 k 做额外加 1，通常是为了让 BA 图边数更接近 ER 平均边数。
        # English: This extra +1 for 10 nodes at density 0.4 usually helps BA graph edge counts better match ER averages.

        
        with open(save_path_edge_mapping, 'wb') as f:
                # 中文：保存映射结果，之后生成非 ER 图时可以复用。
                # English: Save the mapping so non-ER graph generation can reuse it later.
                pickle.dump(avg_number_edges,f)

        return avg_number_edges
    
    def large_scale_simulation(self,
                          graph_type,
                          avg_number_edges=None):
        
    
        # 中文：下面这些数组用于记录每个数据集对应的元信息，最后会合成描述表。
        # English: The arrays below record metadata for each generated dataset and later become a description table.
        seed_runs=[]
        nr_nodes_array=[]
        connectivity_array=[]
        function_transformation_array=[]
        data_scale_array=[]
        beta_upper_array=[]

        #To compute from already sampled graphs:
        # 中文：记录每张图实际采样到的边数。
        # English: Store the actual number of edges sampled for each graph.
        number_edges_array=[]
    
        #Arrays for saving all data eventually:
        # 中文：frames 保存模拟数据，true_causal_matrices 保存二值 DAG，true_weighted_causal_matrices 保存带权 DAG。
        # English: frames stores simulated data; true_causal_matrices stores binary DAGs; true_weighted_causal_matrices stores weighted DAGs.
        frames=[]
        true_causal_matrices=[]
        #true_causal_DAGs=[]
        true_weighted_causal_matrices=[]


        
        #Simulate each dataset & graph 10 times:
        # 中文：外层循环重复 10 次，用不同随机采样生成多个独立数据集。
        # English: The outer loop repeats 10 times to create multiple independently sampled datasets.
        for seed_run in range(0,10):
            #Define the number of nodes to use:
            # 中文：遍历不同节点规模，例如 10、20、50、100。
            # English: Iterate through different graph sizes, such as 10, 20, 50, and 100 nodes.
            for nr_nodes in self.nr_nodes_values:
                #Define the connectivity: edge density for ER graphs using networkx, k based on ER graphs using igraph
                if graph_type=='ER':
                    # 中文：ER 图直接使用预设 edge density。
                    # English: ER graphs directly use the configured edge densities.
                    connectivity_list=self.edge_desnity_values
                else:
                    # 中文：非 ER 图使用 Barabasi 的 m/k 参数，因此需要从 ER 平均边数映射过来。
                    # English: Non-ER graphs use the Barabasi m/k parameter, mapped from ER average edge counts.
                    connectivity_list=[]
                    for ed_dns in [0.2,0.3,0.4]:
                        if ed_dns in avg_number_edges['Nodes_'+str(nr_nodes)].keys():
                            connectivity_list.append(avg_number_edges['Nodes_'+str(nr_nodes)][ed_dns]['k'])
                        
                    #connectivity_list=[avg_number_edges['Nodes_'+str(nr_nodes)][0.2]['k'],
                    #                   avg_number_edges['Nodes_'+str(nr_nodes)][0.3]['k'],
                    #                   avg_number_edges['Nodes_'+str(nr_nodes)][0.4]['k']]
                    
                for connectivity in connectivity_list:
                    #Define non-linearities:
                    # 中文：遍历不同函数组合，比如全线性或线性/ReLU 混合。
                    # English: Iterate through transformation mixes, such as all-linear or linear/ReLU mixtures.
                    for function_transformation in self.nonlinearities:
                        #Define beta upper limits:
                        # 中文：遍历因果权重 beta 的上界，控制边权强度。
                        # English: Iterate through beta upper limits, controlling causal edge strength.
                        for beta_upper_limit in self.betta_upper_limit_values:
                            #Define data scale:
                            # 中文：遍历数据尺度设置，例如原始尺度或标准化尺度。
                            # English: Iterate through data scale settings, such as raw or standardized.
                            for data_scale in self.data_scale_values:
                                #Save seeds run before the actual simulation:
                                # 中文：先保存本轮 seed_run，保证元信息顺序和数据顺序一致。
                                # English: Save seed_run first so metadata order matches generated data order.
                                seed_runs.append(seed_run)

                                #CAUSAL DAG STRUCTURE; NODES & CONNECTIVITY: ########################################
                                # 中文：第一步生成真实因果图结构；不同 graph_type 调用不同图生成器。
                                # English: Step 1 generates the true causal graph structure; different graph_type values use different graph generators.
                                if graph_type=='ER':
                                    current_graph=self.generate_dag(num_nodes=nr_nodes,
                                                              edge_density=connectivity)
                                    edge_list=list(current_graph.edges)
                                else: 
                                    # 中文：Barabasi 图由 igraph 生成，m=connectivity 控制每个新节点连接的边数。
                                    # English: Barabasi graphs are generated by igraph; m=connectivity controls how many edges each new node adds.
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
                                # 中文：第二步把边列表转换成二值邻接矩阵，1 表示存在因果边。
                                # English: Step 2 converts the edge list into a binary adjacency matrix, where 1 means a causal edge exists.
                                current_adjacency_matrix=np.zeros(shape=(nr_nodes,nr_nodes))
                                for edge in edge_list:
                                    current_adjacency_matrix[edge[0]][edge[1]]=1
                                #Save true binary adjacency matrix:
                                true_causal_matrices.append(current_adjacency_matrix)
                                ######################################################################################


                                #TRUE WEIGHTED ADJACENCY MATRIX & BETAS: #############################################
                                # 中文：第三步为每个可能的位置抽样 beta，然后只保留真实边上的权重。
                                # English: Step 3 samples beta values for all possible positions, then keeps weights only on true edges.
                                betas=np.array([self.sample_beta(beta_lower_limit=self.beta_lower_limit,beta_upper_limit=beta_upper_limit) for bt in range(0,nr_nodes*nr_nodes)])
                                weighted_adjacency=np.reshape(betas,newshape=(nr_nodes,nr_nodes))*current_adjacency_matrix
                                # 中文：把 -0.0 或浮点显示上的零统一成 0.0，方便后续保存和检查。
                                # English: Normalize -0.0 or floating zero representations to 0.0 for cleaner storage and inspection.
                                weighted_adjacency=np.where(weighted_adjacency==0.0,0.0,weighted_adjacency)
                                #Save weighted adjacency matrix & betta upper limit:
                                true_weighted_causal_matrices.append(weighted_adjacency)
                                beta_upper_array.append(beta_upper_limit)
                                ######################################################################################


                                #DATASET SIMULATED BASED ON GRAPH STRUCTURE: #########################################
                                # 中文：第四步根据真实带权 DAG 运行 SEM，生成 IID 样本矩阵。
                                # English: Step 4 runs the SEM on the true weighted DAG to generate IID samples.
                                current_dataframe=self.simulate_sem(G=current_graph,
                                                               W=weighted_adjacency, 
                                                               n=self.num_samples,#sem_type: always gaussian
                                                               causal_transformation=function_transformation,
                                                               graph_type=graph_type,
                                                               noise_scale=self.cont_noise)
                                # 中文：把 numpy 矩阵转成 DataFrame，列名使用节点编号。
                                # English: Convert the numpy matrix into a DataFrame with node-index column names.
                                current_dataframe=pd.DataFrame(current_dataframe,columns=[col_index for col_index in range(0,current_dataframe.shape[1])])
                                #Standardize if necessary:
                                if data_scale=='standardized':
                                    # 中文：如果设置为 standardized，则对每个变量按列做 z-score 标准化。
                                    # English: If data_scale is standardized, apply column-wise z-score normalization.
                                    current_dataframe=(current_dataframe-current_dataframe.mean(axis=0))/current_dataframe.std(axis=0)
                                    #scaler=StandardScaler()
                                    #scaler.fit(current_dataframe)
                                    #current_dataframe=pd.DataFrame(scaler.transform(current_dataframe),columns=current_dataframe.columns)
                                frames.append(current_dataframe)
                                #Save sampled data scale:
                                data_scale_array.append(data_scale)
                                
                                #Save linear-nonlinear patterns as strings:
                                # 中文：把函数组合转换成可读字符串，便于之后筛选和保存文件命名。
                                # English: Convert each transformation mix into a readable label for filtering and file naming.
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

        # 中文：all_datasets_frame 是每个模拟数据集的索引表，不直接存数据矩阵，而是存配置和图统计信息。
        # English: all_datasets_frame is the index table for generated datasets; it stores settings and graph stats, not the data matrices.
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
        # 中文：这个函数把模拟结果按节点规模分组，并分别保存为 pkl 文件。
        # English: This function groups generated results by node size and saves each group as a pkl file.
        data_10_nodes=[]
        data_20_nodes=[]
        data_50_nodes=[]
        data_100_nodes=[]


        for frame_index in range(0,frames_descriptions.shape[0]):
            # 中文：逐个读取数据集描述、真实邻接矩阵、真实带权邻接矩阵和模拟数据表。
            # English: For each dataset, load its metadata, true adjacency matrix, weighted adjacency matrix, and simulated DataFrame.
            frame_description=frames_descriptions.loc[[frame_index]].values.tolist()[0]

            current_adjacency_matrix=true_causal_matrices[frame_index]

            current_weighted_adjacency=true_weighted_causal_matrices[frame_index]

            current_dataframe=frames[frame_index]
            
            if sample_size=='Small_Sample_Size':
                # 中文：小样本设置取原始样本量的 1/10，用于构造更困难的数据场景。
                # English: Small-sample mode keeps 1/10 of the original samples to create a harder data scenario.
                small_sample_size=int(current_dataframe.shape[0]/10)
                current_dataframe=current_dataframe.sample(small_sample_size)

            
            if frame_description[1]==10:#graph with 10 nodes
                # 中文：按节点数分桶保存，便于下游实验按规模读取。
                # English: Bucket outputs by node count so downstream experiments can load one graph size at a time.
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
            # 中文：只有对应节点规模存在数据时才写文件，避免生成空 pkl。
            # English: Write a file only when the corresponding node-size bucket has data, avoiding empty pkl files.
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
