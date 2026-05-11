import pandas as pd
import networkx as nx
import numpy as np
import pickle
from math import ceil


def lin_func(x):
    """Linear causal transformation."""
    return x


def relu_func(x):
    """ReLU causal transformation."""
    return np.maximum(0, x)


class Data_Generation_Process():
    """
    IID data generator with one hidden mechanism node.

    Observed node convention:
    - S is the protected attribute and is fixed at node 0.
    - Y is the outcome and is fixed at node n - 1.
    - X nodes are the observed intermediate features 1, ..., n - 2.

    Background DAG convention:
    - The observed background DAG is generated first on nodes 0, ..., n - 1.
    - Every background edge is oriented from the smaller node id to the larger
      node id, so the observed background graph is acyclic by construction.

    Hidden mechanism convention:
    - U is appended as a latent node with full-graph id n.
    - U is not part of the observed ER/SF graph-generation logic.
    - The core hidden path is S -> U -> selected X -> Y.
    - For SF graphs, selected X nodes are chosen from high-degree, hub-like
      intermediate nodes, following the Word note about injecting U into the
      propagation structure instead of adding only an isolated S -> U -> Y path.
    """

    def __init__(self,
                beta_lower_limit,
                betta_upper_limit_values,
                cont_noise,
                nr_nodes_values,
                edge_desnity_values,
                data_scale_values,
                num_samples,
                nonlinearities,
                num_u_children=1,
                u_child_selection='auto',
                inject_direct_u_to_y=False,
                keep_background_edges=True):

        self.beta_lower_limit = beta_lower_limit
        self.betta_upper_limit_values = betta_upper_limit_values
        self.cont_noise = cont_noise
        self.nr_nodes_values = nr_nodes_values
        self.edge_desnity_values = edge_desnity_values
        self.data_scale_values = data_scale_values
        self.num_samples = num_samples
        self.nonlinearities = nonlinearities
        self.num_u_children = num_u_children
        self.u_child_selection = u_child_selection
        self.inject_direct_u_to_y = inject_direct_u_to_y
        self.keep_background_edges = keep_background_edges

        super(Data_Generation_Process, self).__init__()

    def generate_dag(self, num_nodes, edge_density, seed=None):
        """Generate an ER background DAG and orient every edge low id -> high id."""
        graph = nx.gnp_random_graph(n=num_nodes, p=edge_density, seed=seed, directed=False)
        dag = nx.DiGraph()
        dag.add_nodes_from(graph.nodes())
        dag.add_edges_from([(u, v) if u < v else (v, u) for (u, v) in graph.edges()])
        assert nx.is_directed_acyclic_graph(dag)
        return dag

    def generate_scale_free_dag(self, num_nodes, edges_per_new_node, seed=None):
        """
        Generate an SF/Barabasi-Albert background DAG.

        The BA model first creates an undirected scale-free graph. We then orient
        each edge from the smaller id to the larger id so that S, early X nodes,
        late X nodes, and Y follow the required observed ordering.
        """
        edges_per_new_node = int(edges_per_new_node)
        if edges_per_new_node < 1 or edges_per_new_node >= num_nodes:
            raise ValueError('edges_per_new_node must satisfy 1 <= m < num_nodes')

        graph = nx.barabasi_albert_graph(n=num_nodes, m=edges_per_new_node, seed=seed)
        dag = nx.DiGraph()
        dag.add_nodes_from(graph.nodes())
        dag.add_edges_from([(u, v) if u < v else (v, u) for (u, v) in graph.edges()])
        assert nx.is_directed_acyclic_graph(dag)
        return dag

    def _scale_free_connectivity_from_edge_density(self, nr_nodes, edge_density):
        """
        Convert an ER-style edge density value into a BA m value.

        For a BA graph, roughly m new edges are added with each new node. The
        mapping below keeps the old edge_density API usable while giving SF
        graphs a reasonable connectivity level.
        """
        if edge_density >= 1:
            return min(int(edge_density), nr_nodes - 1)
        return max(1, min(nr_nodes - 1, int(ceil(edge_density * (nr_nodes - 1) / 2))))

    def assign_s_x_y_u_roles(self, num_observed_nodes):
        """Assign fixed roles while keeping the observed node labels unchanged."""
        if num_observed_nodes < 3:
            raise ValueError('HiddenMechanism requires at least S, one X node, and Y')

        protected_node = 0
        target_node = num_observed_nodes - 1
        x_nodes = [node for node in range(1, num_observed_nodes - 1)]
        hidden_u_node = num_observed_nodes

        return {'Protected_Node': protected_node,
                'Target_Node': target_node,
                'X_Nodes': x_nodes,
                'Hidden_U_Node': hidden_u_node}

    def select_u_children(self,
                          base_graph,
                          roles,
                          graph_type_key,
                          num_u_children=None,
                          u_child_selection=None):
        """
        Select one or more X nodes that receive hidden influence from U.

        For ER graphs, the default is ordered selection from X nodes by id.
        For SF graphs, the default is hub selection by total degree, then
        out-degree, then node id. This attaches U to the SF propagation backbone.
        """
        if num_u_children is None:
            num_u_children = self.num_u_children
        if u_child_selection is None:
            u_child_selection = self.u_child_selection

        if num_u_children < 1:
            raise ValueError('num_u_children must be positive')

        candidate_x_nodes = list(roles['X_Nodes'])
        if len(candidate_x_nodes) == 0:
            raise ValueError('At least one X node is required')

        selection_key = str(u_child_selection).lower()
        if selection_key == 'auto':
            selection_key = 'hub' if graph_type_key in ['SF', 'SCALE_FREE'] else 'ordered'

        if selection_key == 'hub':
            candidate_x_nodes = sorted(
                candidate_x_nodes,
                key=lambda node: (-base_graph.degree(node), -base_graph.out_degree(node), node)
            )
        elif selection_key == 'ordered':
            candidate_x_nodes = sorted(candidate_x_nodes)
        elif selection_key == 'reverse_ordered':
            candidate_x_nodes = sorted(candidate_x_nodes, reverse=True)
        else:
            raise ValueError("u_child_selection must be 'auto', 'hub', 'ordered', or 'reverse_ordered'")

        return candidate_x_nodes[:min(num_u_children, len(candidate_x_nodes))]

    def inject_hidden_mechanism(self,
                                base_graph,
                                roles,
                                selected_x_children):
        """
        Add U after the observed background graph has been generated.

        U is appended as a latent node outside the observed ER/SF graph logic.
        Its numeric id is larger than all observed nodes so that observed labels
        remain exactly 0, ..., n - 1. The U -> X edge is therefore a deliberate
        exception to the observed low-id -> high-id background rule, but the full
        graph remains a valid DAG because the structural order is S, U, X, Y.
        """
        if self.keep_background_edges:
            full_graph = base_graph.copy()
        else:
            full_graph = nx.DiGraph()
            full_graph.add_nodes_from(base_graph.nodes())

        protected_node = roles['Protected_Node']
        target_node = roles['Target_Node']
        hidden_u_node = roles['Hidden_U_Node']

        full_graph.add_node(hidden_u_node)
        full_graph.add_edge(protected_node, hidden_u_node)

        for x_child in selected_x_children:
            full_graph.add_edge(hidden_u_node, x_child)
            full_graph.add_edge(x_child, target_node)

        if self.inject_direct_u_to_y:
            full_graph.add_edge(hidden_u_node, target_node)

        assert nx.is_directed_acyclic_graph(full_graph)
        return full_graph

    def observed_nodes_from_roles(self, num_observed_nodes):
        """Return observed nodes after hiding U."""
        return [node for node in range(num_observed_nodes)]

    def remove_hidden_nodes_from_matrix(self, matrix, observed_nodes):
        """Drop hidden rows and columns from a full adjacency matrix."""
        return matrix[np.ix_(observed_nodes, observed_nodes)]

    def sample_beta(self, beta_lower_limit, beta_upper_limit):
        """Sample one nonzero structural coefficient with random sign."""
        if np.random.randint(0, 2) == 0:
            return np.random.uniform(-beta_upper_limit, -beta_lower_limit, size=1)[0]
        return np.random.uniform(beta_lower_limit, beta_upper_limit, size=1)[0]

    def apply_transformation(self, dot_product, transformation):
        """Apply one causal transformation chosen from the configured mixture."""
        transformation_func_index = np.random.choice(
            a=[func_index for func_index in range(0, len(transformation))],
            p=[func[0] for func in transformation]
        )
        return transformation[transformation_func_index][1](dot_product)

    def _simulate_single_equation(self, X, w, scale, causal_transformation, n):
        """Simulate one structural equation from its parent samples."""
        if len(w) > 0:
            z = np.random.normal(scale=scale, size=n)
            x = self.apply_transformation(dot_product=X @ w,
                                          transformation=causal_transformation) + z
        else:
            z = np.random.normal(scale=scale, size=n)
            x = z
        return x

    def simulate_sem(self,
                 G,
                 W,
                 n,
                 causal_transformation,
                 graph_type,
                 noise_scale=None):
        """
        Simulate iid samples from the additive-noise SEM defined by G and W.
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

    def get_avg_number_edges_ER_graph(self, frames_descriptions,
                                      save_path_edge_mapping):
        """
        Keep the original helper that maps ER edge counts to SF m values.
        """
        avg_number_edges = {}

        avg_number_edges['Nodes_10'] = {0.2: [],
                                    0.3: [],
                                    0.4: []}

        avg_number_edges['Nodes_20'] = {0.2: [],
                                    0.3: [],
                                    0.4: []}

        avg_number_edges['Nodes_50'] = {0.2: [],
                                    0.3: [],
                                    0.4: []}

        avg_number_edges['Nodes_100'] = {0.2: [],
                                    0.3: [],
                                    0.4: []}

        for idx in range(0, frames_descriptions.shape[0]):
            if frames_descriptions.iloc[idx, 1] == 10:
                avg_number_edges['Nodes_10'][frames_descriptions.iloc[idx, 2]].append(frames_descriptions.iloc[idx, 3])
            elif frames_descriptions.iloc[idx, 1] == 20:
                avg_number_edges['Nodes_20'][frames_descriptions.iloc[idx, 2]].append(frames_descriptions.iloc[idx, 3])
            elif frames_descriptions.iloc[idx, 1] == 50:
                avg_number_edges['Nodes_50'][frames_descriptions.iloc[idx, 2]].append(frames_descriptions.iloc[idx, 3])
            else:
                avg_number_edges['Nodes_100'][frames_descriptions.iloc[idx, 2]].append(frames_descriptions.iloc[idx, 3])

        for node_key in avg_number_edges.keys():
            avg_number_edges[node_key][0.2] = {'e': int(ceil(np.mean(avg_number_edges[node_key][0.2]))),
                                            'd': int(node_key.split('_')[1]),
                                            'k': int(ceil(np.mean(avg_number_edges[node_key][0.2]) / int(node_key.split('_')[1])))}

            avg_number_edges[node_key][0.3] = {'e': int(ceil(np.mean(avg_number_edges[node_key][0.3]))),
                                            'd': int(node_key.split('_')[1]),
                                            'k': int(ceil(np.mean(avg_number_edges[node_key][0.3]) / int(node_key.split('_')[1])))}

            avg_number_edges[node_key][0.4] = {'e': int(ceil(np.mean(avg_number_edges[node_key][0.4]))),
                                            'd': int(node_key.split('_')[1]),
                                            'k': int(ceil(np.mean(avg_number_edges[node_key][0.4]) / int(node_key.split('_')[1])))}

        avg_number_edges['Nodes_10'][0.4]['k'] = avg_number_edges['Nodes_10'][0.4]['k'] + 1

        with open(save_path_edge_mapping, 'wb') as f:
            pickle.dump(avg_number_edges, f)

        return avg_number_edges

    def _connectivity_list_for_graph(self, graph_type_key, nr_nodes, avg_number_edges):
        """Choose edge-density values for ER or BA m values for SF."""
        if graph_type_key == 'ER':
            return self.edge_desnity_values

        if graph_type_key in ['SF', 'SCALE_FREE']:
            if avg_number_edges is not None:
                connectivity_list = []
                for ed_dns in [0.2, 0.3, 0.4]:
                    if ed_dns in avg_number_edges['Nodes_' + str(nr_nodes)].keys():
                        connectivity_list.append(avg_number_edges['Nodes_' + str(nr_nodes)][ed_dns]['k'])
                return connectivity_list

            return [self._scale_free_connectivity_from_edge_density(nr_nodes=nr_nodes,
                                                                    edge_density=edge_density)
                    for edge_density in self.edge_desnity_values]

        raise ValueError("graph_type must be 'ER', 'SF', or 'SCALE_FREE'")

    def _generate_base_graph(self, graph_type_key, nr_nodes, connectivity, seed=None):
        """Generate the observed background DAG before U is injected."""
        if graph_type_key == 'ER':
            return self.generate_dag(num_nodes=nr_nodes,
                                     edge_density=connectivity,
                                     seed=seed)

        return self.generate_scale_free_dag(num_nodes=nr_nodes,
                                            edges_per_new_node=connectivity,
                                            seed=seed)

    def _matrix_from_graph(self, graph, num_nodes):
        """Convert a NetworkX DAG into a binary adjacency matrix."""
        adjacency_matrix = np.zeros(shape=(num_nodes, num_nodes))
        for edge in graph.edges:
            adjacency_matrix[edge[0]][edge[1]] = 1
        return adjacency_matrix

    def _transformation_name(self, function_transformation):
        """Return the original string label for the transformation mixture."""
        if function_transformation == [(1.0, lin_func)]:
            return 'Linear_100%'
        if function_transformation == [(0.5, lin_func), (0.5, relu_func)]:
            return 'Linear_ReLU_50%'
        if function_transformation == [(0.3, lin_func), (0.7, relu_func)]:
            return 'Linear_30%_ReLU_70%'
        return 'Linear_10%_ReLU_90%'

    def large_scale_simulation(self,
                          graph_type,
                          avg_number_edges=None,
                          num_u_children=None,
                          u_child_selection=None):
        """
        Generate observed data with U hidden and retain full-SCM artifacts.

        Returns:
        [descriptions,
         observed_true_causal_matrices,
         observed_true_weighted_causal_matrices,
         observed_frames,
         full_true_causal_matrices,
         full_true_weighted_causal_matrices,
         full_frames]
        """
        graph_type_key = str(graph_type).upper()
        if num_u_children is None:
            num_u_children = self.num_u_children
        if u_child_selection is None:
            u_child_selection = self.u_child_selection

        seed_runs = []
        nr_nodes_array = []
        nr_nodes_full_array = []
        connectivity_array = []
        function_transformation_array = []
        data_scale_array = []
        beta_upper_array = []
        number_edges_array = []
        number_edges_full_array = []

        protected_node_array = []
        target_node_array = []
        hidden_u_node_array = []
        x_nodes_array = []
        selected_x_children_array = []
        observed_nodes_array = []
        hidden_selection_array = []

        frames = []
        true_causal_matrices = []
        true_weighted_causal_matrices = []

        full_frames = []
        full_true_causal_matrices = []
        full_true_weighted_causal_matrices = []

        for seed_run in range(0, 10):
            for nr_nodes in self.nr_nodes_values:
                connectivity_list = self._connectivity_list_for_graph(graph_type_key=graph_type_key,
                                                                     nr_nodes=nr_nodes,
                                                                     avg_number_edges=avg_number_edges)

                for connectivity in connectivity_list:
                    for function_transformation in self.nonlinearities:
                        for beta_upper_limit in self.betta_upper_limit_values:
                            for data_scale in self.data_scale_values:
                                roles = self.assign_s_x_y_u_roles(num_observed_nodes=nr_nodes)

                                base_graph = self._generate_base_graph(graph_type_key=graph_type_key,
                                                                     nr_nodes=nr_nodes,
                                                                     connectivity=connectivity,
                                                                     seed=seed_run)

                                selected_x_children = self.select_u_children(base_graph=base_graph,
                                                                            roles=roles,
                                                                            graph_type_key=graph_type_key,
                                                                            num_u_children=num_u_children,
                                                                            u_child_selection=u_child_selection)

                                full_graph = self.inject_hidden_mechanism(base_graph=base_graph,
                                                                        roles=roles,
                                                                        selected_x_children=selected_x_children)

                                full_num_nodes = nr_nodes + 1
                                full_adjacency_matrix = self._matrix_from_graph(graph=full_graph,
                                                                              num_nodes=full_num_nodes)
                                full_true_causal_matrices.append(full_adjacency_matrix)

                                betas = np.array([self.sample_beta(beta_lower_limit=self.beta_lower_limit,
                                                                 beta_upper_limit=beta_upper_limit)
                                                for bt in range(0, full_num_nodes * full_num_nodes)])
                                full_weighted_adjacency = np.reshape(betas,
                                                                    newshape=(full_num_nodes, full_num_nodes)) * full_adjacency_matrix
                                full_weighted_adjacency = np.where(full_weighted_adjacency == 0.0,
                                                                   0.0,
                                                                   full_weighted_adjacency)
                                full_true_weighted_causal_matrices.append(full_weighted_adjacency)

                                full_dataframe = self.simulate_sem(G=full_graph,
                                                                 W=full_weighted_adjacency,
                                                                 n=self.num_samples,
                                                                 causal_transformation=function_transformation,
                                                                 graph_type=graph_type,
                                                                 noise_scale=self.cont_noise)
                                full_dataframe = pd.DataFrame(full_dataframe,
                                                            columns=[col_index for col_index in range(0, full_dataframe.shape[1])])

                                if data_scale == 'standardized':
                                    full_dataframe = (full_dataframe - full_dataframe.mean(axis=0)) / full_dataframe.std(axis=0)

                                observed_nodes = self.observed_nodes_from_roles(num_observed_nodes=nr_nodes)

                                # U is simulated in the full SCM, then removed from the observed table.
                                observed_dataframe = full_dataframe.loc[:, observed_nodes].copy()
                                observed_dataframe.columns = [col_index for col_index in range(0, observed_dataframe.shape[1])]

                                observed_adjacency_matrix = self.remove_hidden_nodes_from_matrix(matrix=full_adjacency_matrix,
                                                                                              observed_nodes=observed_nodes)
                                observed_weighted_adjacency = self.remove_hidden_nodes_from_matrix(matrix=full_weighted_adjacency,
                                                                                                observed_nodes=observed_nodes)

                                frames.append(observed_dataframe)
                                true_causal_matrices.append(observed_adjacency_matrix)
                                true_weighted_causal_matrices.append(observed_weighted_adjacency)
                                full_frames.append(full_dataframe)

                                seed_runs.append(seed_run)
                                nr_nodes_array.append(nr_nodes)
                                nr_nodes_full_array.append(full_num_nodes)
                                connectivity_array.append(connectivity)
                                number_edges_array.append(int(np.sum(observed_adjacency_matrix)))
                                number_edges_full_array.append(int(np.sum(full_adjacency_matrix)))
                                beta_upper_array.append(beta_upper_limit)
                                data_scale_array.append(data_scale)
                                protected_node_array.append(roles['Protected_Node'])
                                target_node_array.append(roles['Target_Node'])
                                hidden_u_node_array.append(roles['Hidden_U_Node'])
                                x_nodes_array.append(roles['X_Nodes'])
                                selected_x_children_array.append(selected_x_children)
                                observed_nodes_array.append(observed_nodes)

                                selection_label = str(u_child_selection).lower()
                                if selection_label == 'auto':
                                    selection_label = 'hub' if graph_type_key in ['SF', 'SCALE_FREE'] else 'ordered'
                                hidden_selection_array.append(selection_label)
                                function_transformation_array.append(self._transformation_name(function_transformation))

        connectivity_column = 'Edge_Density' if graph_type_key == 'ER' else 'K'
        all_datasets_frame = pd.DataFrame({'Seed_Run': np.array(seed_runs),
                                    'Number_Nodes': np.array(nr_nodes_array),
                                    'Number_Nodes_Full': np.array(nr_nodes_full_array),
                                    connectivity_column: np.array(connectivity_array),
                                    'Number_Edges': np.array(number_edges_array),
                                    'Number_Edges_Full': np.array(number_edges_full_array),
                                    'Transformation_Function': np.array(function_transformation_array),
                                    'Beta_Upper_Limit': np.array(beta_upper_array),
                                    'Data_Scale': np.array(data_scale_array),
                                    'Graph_Type': np.array([graph_type_key] * len(data_scale_array)),
                                    'Protected_Node': np.array(protected_node_array),
                                    'Target_Node': np.array(target_node_array),
                                    'Hidden_U_Node_Full': np.array(hidden_u_node_array),
                                    'X_Nodes': x_nodes_array,
                                    'Selected_U_Children_X': selected_x_children_array,
                                    'Observed_Nodes_Full': observed_nodes_array,
                                    'Hidden_Mechanism_Selection': np.array(hidden_selection_array),
                                    'Core_Causal_Path': np.array(['S->U->X->Y'] * len(data_scale_array))})

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
        """Save generated datasets, grouped by observed node count."""
        data_by_nodes = {}

        for frame_index in range(0, frames_descriptions.shape[0]):
            frame_description = frames_descriptions.loc[[frame_index]].values.tolist()[0]

            current_adjacency_matrix = true_causal_matrices[frame_index]
            current_weighted_adjacency = true_weighted_causal_matrices[frame_index]
            current_dataframe = frames[frame_index]

            if sample_size == 'Small_Sample_Size':
                small_sample_size = int(current_dataframe.shape[0] / 10)
                current_dataframe = current_dataframe.sample(small_sample_size)

            node_count = int(current_dataframe.shape[1])
            if node_count not in data_by_nodes:
                data_by_nodes[node_count] = []

            data_by_nodes[node_count].append([frame_description,
                                              current_adjacency_matrix,
                                              current_weighted_adjacency,
                                              current_dataframe])

        for node_count, node_data in data_by_nodes.items():
            if len(node_data) != 0:
                with open(save_path + graph_type + '_' + sample_size + '_Datasets_' +
                          nonlinear_pattern + '_' + str(node_count) + '_nodes.pkl', 'wb') as f:
                    pickle.dump(node_data, f)
