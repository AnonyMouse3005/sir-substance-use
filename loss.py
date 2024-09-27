import torch
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel.utils import graph_from_networkx


class MMD(torch.nn.Module):

    # NOTE: as long as the base graph kernel is a function that compares the node labels in 2 graphs or sets of graphs
    def __init__(self, kernel=WeisfeilerLehman(n_iter=5, base_graph_kernel=VertexHistogram, normalize=True)):
        super().__init__()
        self.kernel = kernel

    def forward(self, G_sim, G_real):
        """
        **Arguments**:

        - G_sim, G_real: lists of ego networks for all or a subset of agents.
        """
        G = G_sim + G_real
        # print(len(G), networkx.get_node_attributes(G[0], "state"))
        # print(networkx.get_node_attributes(G[200], "state"))
        K = self.kernel.fit_transform(graph_from_networkx(G, node_labels_tag='state'))
        # print(K)
        X_size = len(G_sim)
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return torch.tensor(XX - 2 * XY + YY)
