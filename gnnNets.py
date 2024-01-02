import torch
import torch.nn as nn
from functools import partial
from typing import Union, List
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool
from torch import Tensor
from collections import OrderedDict


def get_gnnNets(input_dim, output_dim, model_config):
    if model_config.gnn_name.lower() == 'gcn':
        gcn_model_param_names = GCNNet.__init__.__code__.co_varnames
        gcn_model_params = {param_name: getattr(model_config.params, param_name)
                            for param_name in gcn_model_param_names
                            if param_name in model_config.params.keys()}
        return GCNNet(input_dim=input_dim,
                      output_dim=output_dim,
                      ** gcn_model_params)
    elif model_config.gnn_name.lower() == 'gin':
        gin_model_param_names = GINNet.__init__.__code__.co_varnames
        gin_model_params = {param_name: getattr(model_config.params, param_name)
                            for param_name in gin_model_param_names
                            if param_name in model_config.params.keys()}
        return GINNet(dim_node=input_dim,
                      num_classes=output_dim,
                      ** gin_model_params)
    else:
        raise ValueError(f"GNN name should be gcn "
                         f"and {model_config.gnn_name} is not defined.")


def identity(x: torch.Tensor, batch: torch.Tensor):
    return x


# cat the max value and sum value (can't understand)
def cat_max_sum(x, batch):
    node_dim = x.shape[-1]
    num_node = 25
    x = x.reshape(-1, num_node, node_dim)
    return torch.cat([x.max(dim=1)[0], x.sum(dim=1)], dim=-1)


class GlobalMeanPool:
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_mean_pool(x, batch)

def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool,
        'identity': identity,
        "cat_max_sum": cat_max_sum,
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    # return the readout_layers dict
    return readout_func_dict[readout.lower()]


# get no_linear layer
def get_nonlinear(nonlinear):
    nonlinear_func_dict = {
        "relu": F.relu,
        "leakyrelu": partial(F.leaky_relu, negative_slope=0.2),
        "sigmoid": F.sigmoid,
        "elu": F.elu
    }
    return nonlinear_func_dict[nonlinear]


class GNNBase(nn.Module):
    def __init__(self):
        super(GNNBase, self).__init__()

    def _argsparse(self, *args, **kwargs):
        r""" Parse the possible input types.
        If the x and edge_index are in args, follow the args.
        In other case, find them in kwargs.
        """
        if args:
            if len(args) == 1:
                data = args[0]
                x = data.x
                edge_index = data.edge_index
                if hasattr(data, 'batch'):
                    batch = data.batch
                else:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)

            elif len(args) == 2:
                x, edge_index = args[0], args[1]
                batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)

            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]

            else:
                raise ValueError(f"forward's args should take 1, 2 or 3 arguments but got {len(args)}")
        else:
            data: Batch = kwargs.get('data')
            if not data:
                x = kwargs.get('x')
                edge_index = kwargs.get('edge_index')
                assert x is not None, "forward's args is empty and required node features x is not in kwargs"
                assert edge_index is not None, "forward's args is empty and required edge_index is not in kwargs"
                batch = kwargs.get('batch')
                if not batch:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
            else:
                x = data.x
                edge_index = data.edge_index
                if hasattr(data, 'batch'):
                    batch = data.batch
                else:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        return x, edge_index, batch

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        new_state_dict = OrderedDict()
        for key in state_dict.keys():
            if key in self.state_dict().keys():
                new_state_dict[key] = state_dict[key]

        super(GNNBase, self).load_state_dict(new_state_dict)


class GCNNet(GNNBase):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 gnn_latent_dim,
                 gnn_dropout: float = 0.0,
                 gnn_emb_normalization: bool = False,
                 gcn_adj_normalization: bool = True,
                 add_self_loop: bool = True,
                 gnn_nonlinear: str = 'relu',
                 readout: str = 'mean',
                 concate: bool = False,
                 fc_latent_dim: Union[List[int], None] = [],
                 fc_dropout: float = 0.0,
                 fc_nonlinear: str = 'relu',
                 ):
        super(GCNNet, self).__init__()
        # first and last layer - dim_features and classes
        self.input_dim = input_dim
        self.output_dim = output_dim
        # GNN part
        self.gnn_latent_dim = gnn_latent_dim
        self.gnn_dropout = gnn_dropout
        self.num_gnn_layers = len(self.gnn_latent_dim)
        self.add_self_loop = add_self_loop
        self.gnn_emb_normalization = gnn_emb_normalization
        self.gcn_adj_normalization = gcn_adj_normalization
        self.gnn_nonlinear = get_nonlinear(gnn_nonlinear)
        self.concate = concate
        # readout
        self.readout_layer = get_readout_layers(readout)
        # FC part
        self.fc_latent_dim = fc_latent_dim
        self.fc_dropout = fc_dropout
        self.num_mlp_layers = len(self.fc_latent_dim) + 1
        self.fc_nonlinear = get_nonlinear(fc_nonlinear)

        if self.concate:
            self.emb_dim = sum(self.gnn_latent_dim)
        else:
            self.emb_dim = self.gnn_latent_dim[-1]

        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, self.gnn_latent_dim[0],
                                  add_self_loops=self.add_self_loop,
                                  normalize=self.gcn_adj_normalization))
        for i in range(1, self.num_gnn_layers):
            self.convs.append(GCNConv(self.gnn_latent_dim[i - 1], self.gnn_latent_dim[i],
                                      add_self_loops=self.add_self_loop,
                                      normalize=self.gcn_adj_normalization))
        # FC layers
        self.mlps = nn.ModuleList()
        if self.num_mlp_layers > 1:
            self.mlps.append(nn.Linear(self.emb_dim, self.fc_latent_dim[0]))

            for i in range(1, self.num_mlp_layers-1):
                self.mlps.append(nn.Linear(self.fc_latent_dim[i-1], self.fc_latent_dim[1]))
            self.mlps.append(nn.Linear(self.fc_latent_dim[-1], self.output_dim))
        else:
            self.mlps.append(nn.Linear(self.emb_dim, self.output_dim))

    def device(self):
        return self.convs[0].weight.device

    def get_emb(self, *args, **kwargs):
        #  node embedding for GNN
        x, edge_index, _ = self._argsparse(*args, **kwargs)
        xs = []
        for i in range(self.num_gnn_layers):
            x = self.convs[i](x, edge_index)
            if self.gnn_emb_normalization:
                x = F.normalize(x, p=2, dim=-1)
            x = self.gnn_nonlinear(x)
            x = F.dropout(x, self.gnn_dropout)
            xs.append(x)

        if self.concate:
            return torch.cat(xs, dim=1)
        else:
            return x

    def forward(self, *args, **kwargs):
        _, _, batch = self._argsparse(*args, **kwargs)
        # node embedding for GNN
        emb = self.get_emb(*args, **kwargs)
        # pooling process
        x = self.readout_layer(emb, batch)

        for i in range(self.num_mlp_layers - 1):
            x = self.mlps[i](x)
            x = self.fc_nonlinear(x)
            x = F.dropout(x, p=self.fc_dropout)

        logits = self.mlps[-1](x)

        return F.softmax(logits, -1)


class GINNet(GNNBase):

    def __init__(self, dim_node, num_classes, dim_hidden=128):
        super().__init__()
        num_layer = 3

        self.conv1 = GINConv(nn.Sequential(nn.Linear(dim_node, dim_hidden), nn.ReLU(),
                                           nn.Linear(dim_hidden, dim_hidden), nn.ReLU()))#,
                                           # nn.BatchNorm1d(dim_hidden)))
        self.convs = nn.ModuleList(
            [
                GINConv(nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
                                      nn.Linear(dim_hidden, dim_hidden), nn.ReLU()))#,
                                      # nn.BatchNorm1d(dim_hidden)))
                for _ in range(num_layer - 1)
             ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(dim_hidden, dim_hidden)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(dim_hidden, num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self._argsparse(*args, **kwargs)

        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)

        out_readout = global_mean_pool(post_conv, batch)

        out = self.ffn(out_readout)
        return F.softmax(out, -1)

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)
        return post_conv