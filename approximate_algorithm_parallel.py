import multiprocessing
import time
import logging
import networkx
import torch
import os
import copy
import pandas as pd
import networkx as nx
import numpy as np
import hydra
import scipy.sparse as sp
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.utils import add_remaining_self_loops, to_scipy_sparse_matrix, \
    k_hop_subgraph, to_networkx, subgraph
from gnnNets import get_gnnNets
from visualization import PlotUtils
from dataset import get_dataset, get_dataloader
from gspan.gspan import gSpan
from warnings import simplefilter
from tqdm import tqdm
from utils import get_logger
from networkx.algorithms import isomorphism

class ParallelAlgorithm():
    def __init__(self, graph, bounds, model, influence_state,\
                diversity_state, graph_id_begin, gamma, budget, tot_node_number) -> None:
        self.graph = graph
        self.bounds = bounds
        self.model = model
        self.influence_state = influence_state
        self.diversity_state = diversity_state
        self.graph_id_begin = graph_id_begin
        self.gamma = gamma
        self.budget = budget
        self.tot_node_number = tot_node_number
        self.solution = []
        self.elements = []
        self.current_union = set()
        self.current_div_union = set()
        self.element_from_colors = [0 for _ in range(len(self.bounds))]
        self.verify_state = [False for _ in range(self.graph.num_nodes)]
        self.counter = 0
    
    def initialize(self):
        ex_labels = self.model(self.graph)
        result = F.softmax(ex_labels, -1).argmax(-1)
        for i in range(self.graph.num_nodes):
            self.elements.append((i, result[self.graph.batch[i].item()].item()))
        # self.verify_process()

    def select(self):
        best = (-1, (-1, -1))
        self.solution.append((-1, -1))
        for element in self.elements:
            self.solution[len(self.solution) - 1] = element
            if not self.feasible(self.solution, element):
                value = self.cal(element)
                if best[0] < value:
                    best = (value, element)
        if best[0] == -1:
            self.solution.pop()    
            return (-1, (-1, -1))
        self.solution.pop()
        return (best[0], best[1])

    def verify_process(self):
        for verify_element in self.elements:
            subset, _, _, _ = k_hop_subgraph(verify_element[0], 3, self.graph.edge_index)
            edge_index = subgraph(subset, self.graph.edge_index, relabel_nodes=True)[0]
            G = Data(self.graph.x[subset], edge_index)

            origin_graph_id = self.graph.batch[verify_element[0]].item()
            origin_graph = self.graph.__getitem__(origin_graph_id)
            x_cf = copy.deepcopy(origin_graph.x)
            for node in subset:
                node_id = node - self.graph_id_begin[origin_graph_id]
                x_cf[node_id] = 0
            G_cf = Data(x_cf, origin_graph.edge_index)
            if self.verify(G, G_cf, verify_element[1]):
                self.verify_state[verify_element[0]] = True 
    
    def verify(self, subgraph, counter_subgraph, label):
        return F.softmax(self.model(subgraph).to('cpu'), -1).argmax(-1) == label and \
               F.softmax(self.model(counter_subgraph).to('cpu'), -1).argmax(-1) != label
    
    def cal(self, element):
        counter = self.influence_state[element[0]] - self.current_union
        div_counter = self.diversity_state[element[0]] - self.current_div_union
        return float(len(counter)) / self.tot_node_number + self.gamma * float(len(div_counter)) / self.tot_node_number      

    def update(self, element):
        self.current_union = self.current_union.union(self.influence_state[element[0]]) 
        self.current_div_union = self.current_div_union.union(self.diversity_state[element[0]])

    def feasible(self, elements, verify_element):
        if self.verify_state[verify_element[0]] == False:
            return False
        self.element_from_colors[verify_element[1]] += 1
        for i in range(len(self.element_from_colors)):
            if self.bounds[i][1] < self.element_from_colors[i]:
                self.element_from_colors[verify_element[1]] -= 1
                return False
        self.element_from_colors[verify_element[1]] -= 1
        return True
    
    def update_bounds(self, node_label):
        self.element_from_colors[node_label] += 1
    
    def update_select(self, element):
        self.solution.append(element)
        self.update(element)
        self.elements.remove(element)


def update_process(algorithm, node_label):
    algorithm.update_bounds(node_label)
    return algorithm

def greedy_node_selection(algorithm):
    node_pair = algorithm.select()
    return algorithm, node_pair

def initialize_process(algorithm):
    algorithm.initialize()
    return algorithm

def aug_normalized_adj(adj_matrix):
    """
    Args:
        adj_matrix: input adj_matrix
    Returns:
        a normalized_adj which follows influence spread idea
    """
    row_sum = np.array(adj_matrix.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    d_matrix_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_matrix_inv_sqrt.dot(adj_matrix).dot(d_matrix_inv_sqrt).tocoo()


def many_graphs_to_one_graph(graphs, threshold, radium, device):
    influenc_state = []
    diversity_state = []
    tot_node = 0
    graph_id_begin = []
    for graph in graphs:    
        adj_matrix = add_remaining_self_loops(graph.edge_index, num_nodes=graph.num_nodes)[0]
        coo_adj_matrix = to_scipy_sparse_matrix(adj_matrix)
        aug_normalized_adj_matrix = aug_normalized_adj(coo_adj_matrix)
        influence_matrix = torch.FloatTensor(aug_normalized_adj_matrix.todense()).to(device)
        influence_matrix2 = torch.mm(influence_matrix, influence_matrix)
        influence_matrix3 = torch.mm(influence_matrix, influence_matrix2).cpu()
        for i in range(influence_matrix3.shape[0]):
            act_set = set()
            div_set = set()
            for j in range(influence_matrix3.shape[1]):
                if influence_matrix3[i][j] > threshold:
                    act_set.add(tot_node+j)
                minus = influence_matrix3[i].numpy() - influence_matrix3[j].numpy()
                if np.linalg.norm(minus) < radium:
                    div_set.add(tot_node+j)
            influenc_state.append(act_set)
            diversity_state.append(div_set)
        graph_id_begin.append(tot_node)
        tot_node += graph.num_nodes
    return Batch.from_data_list(graphs).to('cpu'), influenc_state, diversity_state, graph_id_begin

@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config):
    config.models.params = config.models.params[config.datasets.dataset_name]
    dataset = get_dataset(dataset_root='datasets', dataset_name=config.datasets.dataset_name)
    dataset_params = {
        'batch_size': config.models.params.batch_size,
        'data_split_ratio': config.datasets.data_split_ratio,
        'seed': config.datasets.seed
    }
    if dataset.data.x is not None:
        dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    log_file = (
        f"approximate_{config.datasets.dataset_name}_{config.models.gnn_name}.log"
    )

    logger = get_logger(config.log_path, log_file, config.console_log, config.log_level)
    logger.info(OmegaConf.to_yaml(config))

    n_cpu = multiprocessing.cpu_count()
    logging.info(f'Number of CPUs: {n_cpu}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device: {device}')

    loader = get_dataloader(dataset, **dataset_params)
    test_indices = loader['test'].dataset.indices
    tot_node_number = 0
    partition_cpu = 4
    partition_num = len(test_indices) // partition_cpu
    pool = multiprocessing.Pool(processes=partition_cpu)
    
    node_id_align = [0 for _ in range(partition_cpu)]
    result = []
    t = time.perf_counter()
    for i in range(partition_cpu):
        result.append(pool.apply_async(func=many_graphs_to_one_graph, args=(dataset[test_indices[i*partition_num:(i+1)*partition_num]], 
                                                              float(config.datasets.threshold),
                                                              float(config.datasets.radium),
                                                              device)))
    
    model = get_gnnNets(input_dim=dataset.num_node_features,
                        output_dim=dataset.num_classes,
                        model_config=config.models)
    
    state_dict = torch.load(os.path.join(config.models.gnn_savedir,
                                         config.datasets.dataset_name,
                                         f'{config.models.gnn_name}_'
                                         f'{len(config.models.params.gnn_latent_dim)}l_best.pth'))['net']
    
    model.load_state_dict(state_dict)
    model.eval()

    bounds_list = config.datasets.bounds
    bounds = []
    bounds_list = OmegaConf.to_container(bounds_list)
    for i in range(config.datasets.num_classes):
        bounds.append((bounds_list[2*i], bounds_list[2*i+1]))
    num_classes = config.datasets.num_classes
    for graph in dataset[test_indices]:
        tot_node_number += graph.num_nodes
    parallel_algorithm = []
    for i in range(partition_cpu):
        result[i] = result[i].get()
        if i > 0:
            node_id_align[i] = node_id_align[i-1] + result[i-1][0].num_nodes
        parallel_algorithm.append(ParallelAlgorithm(graph=result[i][0],model=model,bounds=bounds,
                                                    influence_state=result[i][1],
                                                    diversity_state=result[i][2],
                                                    graph_id_begin=result[i][3],
                                                    gamma=float(config.datasets.gamma),
                                                    budget=int(config.datasets.budget),
                                                    tot_node_number=tot_node_number))
    
    result = []
    for i in range(partition_cpu):
        result.append(pool.apply_async(initialize_process, args=(parallel_algorithm[i],)))
    for i in range(partition_cpu):
        parallel_algorithm[i] = result[i].get()
    
    cardinality_k=int(config.datasets.budget)
    select_nodes = []
    logger.info("Start greedy node selection process")
    for _ in tqdm(range(cardinality_k), desc='Process', leave=True, ncols=100, unit='B', unit_scale=True):
        result = []
        for i in range(partition_cpu):
            result.append(pool.apply_async(greedy_node_selection, args=(parallel_algorithm[i],)))
        max_node = (-1, -1)
        max_value = -100 
        partition_id = -1
        for i in range(partition_cpu):
            parallel_algorithm[i], node_pair = result[i].get()
            if node_pair[0] > max_value:
                max_value = node_pair[0]
                node_id = node_pair[1][0] + node_id_align[i]
                max_node = (node_id, node_pair[1][1])
                partition_id = i
        select_nodes.append(max_node)
        parallel_algorithm[partition_id].update_select((max_node[0]-node_id_align[partition_id], max_node[1]))
        update_result = []
        for i in range(partition_cpu):
            update_result.append(pool.apply_async(update_process, args=(parallel_algorithm[i], max_node[1])))
        for i in range(partition_cpu):
            parallel_algorithm[i] = update_result[i].get()

    pool.close()
    pool.join()
    cost_time = time.perf_counter() - t
    print(f'Cost:{cost_time:.4f}s')
    logger.info(f'Cost:{cost_time:.4f}s')
    logger.info("Finish greedy node selection process")
    


if __name__ == '__main__':
    import sys

    simplefilter(action="ignore", category=FutureWarning)
    sys.argv.append(f"models.gnn_savedir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()
