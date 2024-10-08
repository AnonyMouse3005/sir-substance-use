import torch
import networkx
import torch_geometric
import torch_geometric.transforms
from torch_geometric.utils import dropout_edge, add_random_edge, degree, sort_edge_index
import torch_sparse
import numpy as np

from blackbirds.utils import soft_minimum


class SIR(torch.nn.Module):
    def __init__(
        self,
        graph: networkx.Graph,
        n_timesteps: int,
        n_agents: int,
        device: str = "cpu",
        delta_t: float = 1.0,
    ):
        """
        Implements a differentiable SIR model on a graph.

        **Arguments:**

        - `graph`: a networkx graph
        - `n_timesteps`: the number of timesteps to run the model for
        - `device` : device to use (eg. "cpu" or "cuda:0")
        """
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_agents = n_agents
        self.delta_t = delta_t
        # convert graph from networkx to pytorch geometric
        self.graph = torch_geometric.utils.convert.from_networkx(graph).to(device)
        self.init_mean_degree = torch.mean(degree(self.graph.edge_index[0]))
        self.mp = SIRMessagePassing(aggr="add", node_dim=-1)
        self.aux = torch.ones(self.n_agents, device=device)
        self.device = device

    def sample_bernoulli_gs(self, probs: torch.Tensor, tau: float = 0.1):
        """
        Samples from a Bernoulli distribution in a diferentiable way using Gumble-Softmax

        **Arguments:**

        - probs: a tensor of shape (n,) containing the probabilities of success for each trial
        - tau: the temperature of the Gumble-Softmax distribution
        """
        logits = torch.vstack((probs, 1 - probs)).T.log()
        gs_samples = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True)
        return gs_samples[:, 0]

    def initialize(self, params):
        """
        Initializes the model setting the adequate number of initial infections.

        **Arguments**:

        - `initial_fraction_infected`: the fraction of infected agents at the beginning of the simulation
        """
        params = soft_minimum(params, torch.tensor(0.0, device=params.device), 2)
        initial_fraction_infected = 10**params[2]
        n_agents = self.graph.num_nodes
        # sample the initial infected nodes
        probs = initial_fraction_infected * torch.ones(n_agents, device=self.device)
        new_infected = self.sample_bernoulli_gs(probs)
        # set the initial state
        infected = new_infected
        susceptible = 1 - new_infected
        recovered = torch.zeros(n_agents, device=self.device)
        x = torch.vstack((infected, susceptible, recovered))
        return x.reshape(1, 3, n_agents)

    def step(self, params, x: torch.Tensor):
        """
        Runs the model forward for one timestep.

        **Arguments**:

        - `gamma`: the recovery probability
        - `rho`: the relapse probability
        - x: a tensor of shape (3, n_agents) containing the infected, susceptible, and recovered counts.
        """
        params = soft_minimum(params, torch.tensor(0.0, device=params.device), 2)
        params = 10**params
        gamma, rho, _, _, _, _, _ = params
        infected, susceptible, recovered = x[-1]  ## store the states for each agent
        # Get number of infected neighbors per node, return 0 if node is infected already.
        n_infected_neighbors = self.mp(self.graph.edge_index, infected, 1 - infected)
        n_neighbors = max(1.0, self.mp(  # NOTE: in case a node has no neighbors
            self.graph.edge_index,
            self.aux,
            self.aux
        ))
        lambda_1 = susceptible
        lambda_2 = rho * recovered
        lambda_ = (lambda_1 + lambda_2) * n_infected_neighbors / n_neighbors * self.delta_t
        # each contact has a chance of infecting a susceptible or recovered node
        prob_infected_or_relapsed = 1.0 - torch.exp(-lambda_)
        prob_infected_or_relapsed = torch.clip(prob_infected_or_relapsed, min=1e-10, max=1.0)
        # sample the infected and relapsed nodes
        new_infected_and_relapsed = self.sample_bernoulli_gs(prob_infected_or_relapsed)

        prob_recovery = gamma * infected
        prob_recovery = torch.clip(prob_recovery, min=1e-10, max=1.0)
        # sample recoverd people
        new_recovered = self.sample_bernoulli_gs(prob_recovery)

        # update the state of the agents
        infected = infected + new_infected_and_relapsed - new_recovered
        susceptible = susceptible - (susceptible * new_infected_and_relapsed)
        recovered = recovered + new_recovered - (recovered * new_infected_and_relapsed)
        x = torch.vstack((infected, susceptible, recovered)).reshape(1, 3, -1)
        return x

    def observe(self, x: torch.Tensor):
        """
        Returns the total number of infected and recovered agents per time-step

        **Arguments**:

        - x: a tensor of shape (3, n_agents) containing the infected, susceptible, and recovered counts.
        """
        transform = torch_geometric.transforms.RootedEgoNets(num_hops=1)
        return [
            x[:, 0, :].sum(1) / self.n_agents,
            x[:, 1, :].sum(1) / self.n_agents,
            x[:, 2, :].sum(1) / self.n_agents,
            torch.argmax(torch.squeeze(x), dim=0).reshape(1, -1),  ## state S, I, R takes value of 1, 0, 2, respectively
            transform(self.graph)
        ]

    def forward(self, params):
        """
        Runs the model for the specified number of timesteps.

        **Arguments**:

        - params: a tensor of shape (2,) containing the gamma and rho parameters
        """
        x = self.initialize(params)
        infected_per_day, susceptible_per_day, recovered_per_day, states_per_day, ego_nets = self.observe(x)
        ego_nets_per_day = [ego_nets]
        # Example instance: RootedSubgraphData(edge_index=[2, 10000], num_nodes=1000, sub_edge_index=[2, 78452], n_id=[11000], e_id=[78452], n_sub_batch=[11000], e_sub_batch=[78452])
        for t in range(self.n_timesteps):
            x = self.step(params, x)
            # get the observations
            infected, susceptible, recovered, states, ego_nets = self.observe(x)
            infected_per_day = torch.cat((infected_per_day, infected))
            susceptible_per_day = torch.cat((susceptible_per_day, susceptible))
            recovered_per_day = torch.cat((recovered_per_day, recovered))
            states_per_day = torch.cat((states_per_day, states))
            ego_nets_per_day.append(ego_nets)
            self.update_graph(params, torch.squeeze(states))
        return susceptible_per_day, infected_per_day, recovered_per_day, states_per_day, ego_nets_per_day

    def update_graph(self, params, states):
        """
        Rules for network dynamics.

        **Arguments**:

        - `alpha1`, `alpha2`: the edge deletion probability for same- and different-state agents, respectively
        - `beta1`, `beta2`: the edge addition probability for same- and different-state agents, respectively
        """
        params = soft_minimum(params, torch.tensor(0.0, device=params.device), 2)
        params = 10**params
        _, _, _, alpha1, alpha2, beta1, beta2 = params

        edges = self.graph.edge_index
        mean_degree = torch.mean(degree(edges[0]))
        mask_same_state = torch.eq(states[edges[0]], states[edges[1]])
        edges_same_state = edges[:, mask_same_state]
        edges_diff_state = edges[:, ~mask_same_state]
        if edges_same_state.size(0) > 0:
            retained_edges_same_state, _ = dropout_edge(edges_same_state, force_undirected=True,
                                                        p=fix_prob(alpha1.item(), self.init_mean_degree, mean_degree, 'rm'))
        if edges_diff_state.size(0) > 0:
            retained_edges_diff_state, _ = dropout_edge(edges_diff_state, force_undirected=True,
                                                        p=fix_prob(alpha2.item(), self.init_mean_degree, mean_degree, 'rm'))

        _, new_edges_same_state = add_random_edge(edges_same_state, force_undirected=True, num_nodes=self.n_agents,
                                                  p=fix_prob(beta1.item(), self.init_mean_degree, mean_degree, 'add'))
        _, new_edges_diff_state = add_random_edge(edges_diff_state, force_undirected=True, num_nodes=self.n_agents,
                                                  p=fix_prob(beta2.item(), self.init_mean_degree, mean_degree, 'add'))

        self.graph.edge_index = sort_edge_index(torch.cat([retained_edges_same_state, retained_edges_diff_state,
                                                           new_edges_same_state, new_edges_diff_state], dim=1))
        self.graph = torch_geometric.transforms.RemoveDuplicatedEdges()(self.graph)  # TODO: why are there duplicated edges?


class SIRMessagePassing(torch_geometric.nn.conv.MessagePassing):
    """
    Class used to pass messages between agents about their infected status.
    """

    def forward(
        self,
        edge_index: torch.Tensor,
        infected: torch.Tensor,
        susceptible: torch.Tensor,
    ):
        """
        Computes the sum of the product between the node's susceptibility and the neighbors' infected status.

        **Arguments**:

        - edge_index: a tensor of shape (2, n_edges) containing the edge indices
        - infected: a tensor of shape (n_nodes,) containing the infected status of each node
        - susceptible: a tensor of shape (n_nodes,) containing the susceptible status of each node
        """
        return self.propagate(edge_index, x=infected, y=susceptible)

    def message(self, x_j, y_i):
        return x_j * y_i


def simulate_and_observe_model(
    model: torch.nn.Module,
    params: torch.Tensor,
    data,
    agent_sample,
    loss_fn,
):
    """Runs the simulator for the given parameters and calls the model's observe method.

    **Arguments:**

    - `model`: A torch.nn.Module implemnting the `initialize`, `forward` and `observe` methods.
    - `params`: The parameters taken by the model's `forward` method.
    - `data`: True data to compare against.
    """
    x = model.initialize(params)
    _, _, _, states, ego_nets = model.observe(x)
    loss_per_day = torch.zeros(model.n_timesteps + 1)
    loss_per_day[0] = loss_fn(get_ego_net(ego_nets, states, agent_sample), get_ego_net(data[4][0], data[3][0], agent_sample))
    for t in range(model.n_timesteps):
        x = model.step(params, x)
        # get the observations
        _, _, _, states, ego_nets = model.observe(x)
        loss_per_day[t+1] = loss_fn(get_ego_net(ego_nets, states, agent_sample), get_ego_net(data[4][t+1], data[3][t+1], agent_sample))
        model.update_graph(params, torch.squeeze(states))

    return loss_per_day.sum()


def map_states(states):
    state_dict = {0: 'I', 1: 'S', 2: 'R'}  ## state S, I, R takes value of 1, 0, 2, respectively
    return np.vectorize(state_dict.get)(states)


def get_ego_net(ego_nets, states, agent_sample):
    G = []
    for i in range(len(agent_sample)):
        node_ids = ego_nets.n_id[ego_nets.n_sub_batch == agent_sample[i]]
        ego_net = ego_nets.subgraph(node_ids)
        g = torch_geometric.utils.to_networkx(ego_net, to_undirected=True)
        networkx.set_node_attributes(g, {j: state for j, state in enumerate(map_states(torch.squeeze(states)[node_ids]))}, "state")
        G.append(g)
    return G


def fix_prob(p, init_mean_degree, current_mean_degree, mode):
    if mode == 'add':
        return np.clip(p/np.emath.logn(init_mean_degree, current_mean_degree), 1e-10, 1.0)
    elif mode == 'rm':
        return np.clip(-p/np.emath.logn(init_mean_degree, current_mean_degree) + 2*p, 1e-10, 1.0)


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from time import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_timesteps", type=int, default=100)
    parser.add_argument("--n_agents", type=int, default=1000)
    parser.add_argument("--initial_fraction_infected", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.1)  ## set to np.nan for running simulations exhaustively
    parser.add_argument("--rho", type=float, default=0.3)  ## set to np.nan for running simulations exhaustively
    parser.add_argument("--delta_t", type=float, default=1.0)

    params = parser.parse_args()

    # create a random graph
    # graph = networkx.erdos_renyi_graph(params.n_agents, 0.1)
    # graph = networkx.complete_graph(params.n_agents)
    graph = networkx.watts_strogatz_graph(params.n_agents, 10, 0.01)

    if np.isnan(params.gamma) or np.isnan(params.rho):
        gamma_list = np.arange(0, 0.51, 0.05)
        rho_list = np.arange(0, 0.76, 0.05)
    else:
        gamma_list, rho_list = [params.gamma], [params.rho]

    for gamma in gamma_list:
        for rho in rho_list:
            # create the model
            model = SIR(
                graph, params.n_timesteps, params.n_agents, params.device, params.delta_t
            )
            t1 = time()
            S, I, R, states, ego_nets = model(
                torch.tensor([gamma, rho, params.initial_fraction_infected])
            )
            t2 = time()
            print(f"Time elapsed: {t2 - t1:.2f} seconds")

            # plot the results
            plt.figure()
            plt.plot(S.cpu(), label="S")
            plt.plot(I.cpu(), label="I")
            plt.plot(R.cpu(), label="R")
            plt.xlabel("Time")
            plt.ylabel("Fraction of agents")
            plt.ylim(0, 1)
            plt.legend()
            plt.title("Agents PyTorch")
            plt.savefig(f"./figures/gorman/alcohol_agents_gamma{int(gamma*100):03d}_rho{int(rho*100):03d}.png", dpi=150)
            # plt.show()

            # plot the ego network of some random agent
            a_i = 1
            g_start = get_ego_net(ego_nets[0], a_i)
            g_end = get_ego_net(ego_nets[-1], a_i)
            states = states.cpu()
