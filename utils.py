from collections import defaultdict, Counter
import seaborn as sns
sns.set_style("darkgrid")


class Graph(object):
    def __init__(self, edge_list):
        """ Builds a graph object from an edge list [[u,v], ...] """
        edge_list = [(e[0], e[1]) for e in edge_list]
        self.edge_list = edge_list

        # saves graph as dictionary {node: [node1, ...]}
        edge_dict = defaultdict(list)

        for e in edge_list:
            edge_dict[e[0]].append(e[1])

        self.edge_dict = edge_dict

        # to make successive calls to is_directed() more efficient
        self.directed_bool = None

    def edge_count(self):
        """ Returns the number of edges in G """
        return len(self.edge_list)

    def node_count(self):
        """ Returns the number of nodes in G """
        return len(self.edge_dict.keys())

    def is_directed(self):
        """ Returns True if G is directed, False otherwise """
        if self.directed_bool is not None:
            return self.directed_bool

        reverse_edge_list = [(e[1], e[0]) for e in self.edge_list]
        intersection = set.intersection(set(self.edge_list), set(reverse_edge_list))
        self.directed_bool = len(intersection) != len(self.edge_list)
        return self.directed_bool

    def neighbour_sizes(self):
        return list(map(len, list(self.edge_dict.values())))

    def avg_num_links(self):
        """ Returns the average number of links per node """
        neighbour_size_list = self.neighbour_sizes()
        return sum(neighbour_size_list) / self.node_count()

    def max_possible_edges(self):
        """ Returns the number of possible edge """
        n = self.node_count()
        return n * (n - 1) / 2

        return total

    def density(self):
        """ Returns the density of G according to https://en.wikipedia.org/wiki/Dense_graph """
        ratio = self.edge_count() / self.max_possible_edges()

        if self.is_directed():
            return 2 * ratio

        return ratio

    def plot_degree_distribution(self):
        neighbour_size_list = self.neighbour_sizes()
        counter = Counter()
        counter.update(neighbour_size_list)
        degrees = list(counter.keys())
        freq = list(counter.values())

        splot = sns.scatterplot(x=degrees, y=freq)
        splot.set(title="Log-log scale Degree Distribution",
                  xscale="log", yscale="log",
                  xlabel="Node Degree", ylabel="Number of Nodes");


