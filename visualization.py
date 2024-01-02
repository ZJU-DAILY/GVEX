import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap
from typing import List, Dict, Union


class PlotUtils(object):
    def __init__(self, dataset_name, is_show=True):
        self.dataset_name = dataset_name
        self.is_show = is_show

    def plot(self, graph, nodelist, figname, title_sentence=None, **kwargs):
        """plot function for different dataset"""
        if self.dataset_name.lower() in ["ba_2motifs"]:
            self.plot_ba2motifs(
                graph, nodelist, title_sentence=title_sentence, figname=figname
            )
        elif self.dataset_name.lower() in ["mutag", "bbbp", "bace", "mutagenicity"]:
            x = kwargs.get("x")
            self.plot_molecule(
                graph, nodelist, x, title_sentence=title_sentence, figname=figname
            )
        elif self.dataset_name.lower() in ["graph_sst2", "twitter"]:
            words = kwargs.get("words")
            self.plot_sentence(
                graph,
                nodelist,
                words=words,
                title_sentence=title_sentence,
                figname=figname,
            )
        else:
            raise NotImplementedError

    def plot_subgraph(
        self,
        graph,
        nodelist,
        colors: Union[None, str, List[str]] = "#FFA500",
        labels=None,
        edge_color="gray",
        edgelist=None,
        subgraph_edge_color="black",
        title_sentence=None,
        figname=None,
    ):

        if edgelist is None:
            edgelist = [
                (n_frm, n_to)
                for (n_frm, n_to) in graph.edges()
                if n_frm in nodelist and n_to in nodelist
            ]

        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(
            graph,
            pos_nodelist,
            nodelist=nodelist,
            node_color="black",
            node_shape="o",
            node_size=400,
        )
        nx.draw_networkx_nodes(
            graph, pos, nodelist=list(graph.nodes()), node_color=colors, node_size=200
        )
        nx.draw_networkx_edges(graph, pos, width=2, edge_color=edge_color, arrows=False)
        nx.draw_networkx_edges(
            graph,
            pos=pos_nodelist,
            edgelist=edgelist,
            width=6,
            edge_color="black",
            arrows=False,
        )

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis("off")
        if title_sentence is not None:
            plt.title(
                "\n".join(wrap(title_sentence, width=60)), fontdict={"fontsize": 15}
            )
        if figname is not None:
            plt.savefig(figname, format=figname[-3:], dpi=300)

        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_sentence(
        self, graph, nodelist, words, edgelist=None, title_sentence=None, figname=None
    ):
        pos = nx.kamada_kawai_layout(graph)
        words_dict = {i: words[i] for i in graph.nodes}
        if nodelist is not None:
            pos_coalition = {k: v for k, v in pos.items() if k in nodelist}
            nx.draw_networkx_nodes(
                graph,
                pos_coalition,
                nodelist=nodelist,
                node_color="yellow",
                node_shape="o",
                node_size=500,
            )
            if edgelist is None:
                edgelist = [
                    (n_frm, n_to)
                    for (n_frm, n_to) in graph.edges()
                    if n_frm in nodelist and n_to in nodelist
                ]
                nx.draw_networkx_edges(
                    graph,
                    pos=pos_coalition,
                    edgelist=edgelist,
                    width=5,
                    edge_color="yellow",
                )

        nx.draw_networkx_nodes(graph, pos, nodelist=list(graph.nodes()), node_size=300)

        nx.draw_networkx_edges(graph, pos, width=2, edge_color="grey")
        nx.draw_networkx_labels(graph, pos, words_dict)

        plt.axis("off")
        plt.title("\n".join(wrap(" ".join(words), width=50)))
        if title_sentence is not None:
            string = "\n".join(wrap(" ".join(words), width=50)) + "\n"
            string += "\n".join(wrap(title_sentence, width=60))
            plt.title(string)
        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_ba2motifs(
        self, graph, nodelist, edgelist=None, title_sentence=None, figname=None
    ):
        return self.plot_subgraph(
            graph,
            nodelist,
            edgelist=edgelist,
            title_sentence=title_sentence,
            figname=figname,
        )

    def plot_molecule(
        self, graph, nodelist, x, edgelist=None, title_sentence=None, figname=None
    ):
        # collect the text information and node color
        if self.dataset_name == "mutag":
            node_dict = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}
            node_idxs = {
                k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])
            }
            node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
            node_color = [
                "#E49D1C",
                "#4970C6",
                "#FF5357",
                "#29A329",
                "brown",
                "darkslategray",
                "#F0EA00",
            ]
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        elif self.dataset_name == "Mutagenicity":
            node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P',
                         9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
            node_idxs = {
                k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])
            }
            node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
            node_color = ["#E49D1C","#FF5357","brown",'#90BEE0',"#4970C6","#29A329","#F0EA00", '#A9A9A9',
                          '#EDDDC3', "#A97AD8", '#4B74B2', '#BA55D3', '#7B68EE', '#DAA520']
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]            
        else:
            raise NotImplementedError

        self.plot_subgraph(
            graph,
            nodelist,
            colors=colors,
            labels=node_labels,
            edgelist=edgelist,
            edge_color="gray",
            subgraph_edge_color="black",
            title_sentence=title_sentence,
            figname=figname,
        )