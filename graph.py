from collections import defaultdict, Counter
import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
from contextlib import suppress
from scipy import sparse

sns.set_style("darkgrid")
graph_filename = "wikigraph_reduced.csv"
categories_filename = "reduced_categories.txt"


def intersection(list_a, list_b):
    return list(set(list_a).intersection(list_b))


class Graph(object):
    def __init__(self):
        # the following data structures are used to make the specific tasks (RQs)
        # more time efficient, this of course at the expense of space efficiency.
        self.edge_list = []
        self.node_list = set()
        self.categories = defaultdict(list)
        self.in_edge_dict = defaultdict(list)
        self.out_edge_dict = defaultdict(list)
        self.directed_bool = None

        # the following data structures are used to manage labels (categories) of nodes
        self.category_dict = defaultdict(int)  # {category_str : id}
        self.nodes_by_category = defaultdict(list)  # {category_id : [node1, node2, ...]}
        self.category_by_node = defaultdict(int)  # {node : category_id}

    # ----------- GRAPH GENERATION ----------- #

    def create_from_file(self, filename=graph_filename, sep="\t", cols=(1, 2)):
        """ Creates a graph reading the edgelist from filename

        :param filename: str
        :param sep: str
        :param cols: list of ints
        :return: None
        """
        # the expected format is that each row represents an edge with node labels
        # found in cols separated by sep
        df = pd.read_csv(filename, sep=sep, usecols=cols)
        edge_list = df.values.tolist()
        self.create_from_edgelist(edge_list)

    def create_from_edgelist(self, edge_list):
        """ Populates the graph adding nodes and edges from edge_list

        :param edge_list: list of (int, int)
        :return: None
        """
        edge_list = [(e[0], e[1]) for e in edge_list]
        self.edge_list = edge_list

        # saves graph as dictionary {node: [node1, ...]}
        for e in edge_list:
            self.out_edge_dict[e[0]].append(e[1])
            self.in_edge_dict[e[1]].append(e[0])
            self.node_list.add(e[0])
            self.node_list.add(e[1])

        # to make successive calls to is_directed() more efficient
        self.directed_bool = None

    def load_categories(self, filename=categories_filename):
        """ Loads nodes categories from filename, expected format:
            category: node1 node2 ..."""
        idx = 0

        with open(filename, "r") as file:
            for line in file.readlines():
                category, node_list = line.strip().split(':')
                node_list = list(map(int, node_list.split()))
                self.nodes_by_category[idx] = node_list
                self.category_dict[category] = idx
                for node in node_list:
                    self.category_by_node[node] = idx
                idx += 1

    # ----------- GENERAL PURPOSE ----------- #

    # The following functions have been implemented in the process of solving
    # specific RQs but were ultimately not used in the final solution
    def add_node(self, node=None):
        """ Add the specified node to the graph if possible, or adds a node
            with an increasing label wrt to the max current label

        :param node: int
                     if None the node takes the largest index in the graph + 1
        :return: None
        """
        if node is None:
            node = max(self.node_list) + 1
            self.node_list.add(node)
            return node
        elif node not in self.node_list:
            self.node_list.add(node)
        else:
            print("[Error] Node already in the graph.")

    def add_edge(self, edge):
        """ Adds an edge to the graph.
            The current implementation does not support parallel edges (i.e. multigraph)

        :param edge: (int, int)
        :return: None
        """
        assert(len(edge) == 2)
        if edge in self.edge_list:
            print("[Error] Edge already in the graph.")
        else:
            self.out_edge_dict[edge[0]].append(edge[1])
            self.in_edge_dict[edge[1]].append(edge[0])

    def delete_node(self, node):
        """ Deletes the target node and all its incident edges from the graph

        :param node: int
        :return: None
        """
        if node not in self.node_list:
            print("[Error] Specified node not in the graph.")
            return None

        with suppress(KeyError, ValueError):
            # delete incoming edges
            for u in self.in_edge_dict[node]:
                self.out_edge_dict[u].remove(node)
                self.edge_list.remove((u, node))
            # delete outcoming edges
            for u in self.out_edge_dict[node]:
                self.in_edge_dict[u].remove(node)
                self.edge_list.remove((node, u))
            # delete node
            del self.in_edge_dict[node]
            del self.out_edge_dict[node]
            self.node_list.remove(node)

    def delete_edge(self, edge):
        if edge not in self.edge_list:
            print("Specified edge not in the graph.")
            return None

        with suppress(KeyError, ValueError):
            self.in_edge_dict[edge[1]].remove(edge[0])
            self.out_edge_dict[edge[0]].remove(edge[1])
            self.edge_list.remove(edge)

    def edge_count(self):
        """ Returns the number of edges in G """
        return len(self.edge_list)

    def node_count(self):
        """ Returns the number of nodes in G """
        return len(self.node_list)

    def max_node_id(self):
        """ Returns the maximum id of nodes in the graph """
        return max([max(u, v) for u, v in self.edge_list])

    def is_directed(self):
        """ Returns True if G is directed, False otherwise """
        if self.directed_bool is not None:
            return self.directed_bool

        reverse_edge_list = [(e[1], e[0]) for e in self.edge_list]
        edge_intersection = set.intersection(set(self.edge_list), set(reverse_edge_list))
        self.directed_bool = len(edge_intersection) != len(self.edge_list)
        return self.directed_bool

    def neighbours(self, node):
        return self.out_edge_dict[node]

    def neighbours_by_category(self, node, cat_id):
        """ Returns the list of neighbours of category cat_id """
        node_list = self.out_edge_dict[node]
        cat_nodes = self.get_nodes_from_category_id(cat_id)
        intersection = set.intersection(set(node_list), set(cat_nodes))
        return list(intersection)

    def neighbour_sizes(self):
        return list(map(len, list(self.out_edge_dict.values())))

    def avg_num_links(self):
        """ Returns the average number of links per node """
        neighbour_size_list = self.neighbour_sizes()
        return sum(neighbour_size_list) / self.node_count()

    def density(self):
        """ Returns the density of G according to https://en.wikipedia.org/wiki/Dense_graph """
        n = self.node_count()
        den = n * (n - 1) / 2
        ratio = self.edge_count() / den

        if self.is_directed():
            return ratio / 2

        return ratio

    def in_degree(self, node):
        """ Returns the degree of node """
        return len(self.in_edge_dict[node])

    def out_degree(self, node):
        """ Returns the degree of node """
        return len(self.out_edge_dict[node])

    def central_node(self, category_id):
        """ Returns the largest degree node in category_id """
        nodes_in_category = self.get_nodes_from_category_id(category_id)
        central_node = nodes_in_category[0]

        for node in nodes_in_category[1:]:
            if self.in_degree(node) > self.in_degree(central_node):
                central_node = node

        return central_node

    def categories_list(self, as_string=True):
        return list(self.category_dict.keys())

    def induced_subgraph(self, cat_id_1, cat_id_2):
        node_list_1 = self.get_nodes_from_category_id(cat_id_1)
        node_list_2 = self.get_nodes_from_category_id(cat_id_2)
        nodes = list(set.union(set(node_list_2), set(node_list_1)))

        edge_list = []

        for u in nodes:
            for v in intersection(nodes, self.out_edge_dict[u]):
                edge_list.append([u, v])
            for v in intersection(nodes, self.in_edge_dict[u]):
                edge_list.append([v, u])

        graph = Graph()
        graph.create_from_edgelist(edge_list)
        return graph

    # ----------- TASK SPECIFIC ----------- #

    def get_category_id(self, category):
        """ Returns the id of category """
        try:
            return self.category_dict[category]
        except KeyError:
            return None

    def get_node_category(self, node):
        """ Returns the category id of node """
        try:
            return self.category_by_node[node]
        except KeyError:
            return None

    def get_nodes_from_category_id(self, category_id):
        """ Returns the set of nodes belonging to category_id """
        try:
            return self.nodes_by_category[category_id]
        except KeyError:
            return None

    def category_distance(self, cat_id, targets_ids, max_dist=50):
        """ Return the minimum distance from the central node in
            category to all the nodes in node_list that are in category """
        central_node = self.central_node(cat_id)
        frontier = defaultdict(list)
        frontier[0].append(central_node)
        visited = defaultdict(bool)
        visited[central_node] = True

        for i in range(0, max_dist):
            if len(frontier[i]) == 0:
                break
            for node in frontier[i]:
                for v in self.neighbours(node):
                    if not visited[v]:
                        visited[v] = True
                        frontier[i + 1].append(v)
                try:
                    targets_ids.remove(node)
                except ValueError:
                    pass

            if len(targets_ids) == 0:
                break

        if len(targets_ids) > 0:
            return -1
        else:
            return i

    # ---------------- RQ 2 ---------------- #

    def truncated_bfs(self, root, dist):
        """ Returns the set of nodes at distance at most dist from root """
        visited = defaultdict(bool)  # boolean vector of visits
        frontier = defaultdict(list)
        frontier[0].append(root)

        for i in range(0, dist):
            if len(frontier[i]) == 0:
                break
            for u in frontier[i]:
                for v in self.neighbours(u):
                    if not visited[v]:
                        visited[v] = True
                        frontier[i + 1].append(v)

        return list(visited.keys())

    # ---------------- RQ 3 ---------------- #

    def reachable_targets(self, root, targets, dist):
        reached = self.truncated_bfs(root, dist)
        diff = set(targets).difference(set(reached))
        if len(diff) == 0:
            return True
        else:
            return False

    # ---------------- RQ 4 ---------------- #

    def modified_dfs(self, node, target, visited_edges):
        """

        :param node:
        :param target:
        :param path:
        :param visited_edges:
        :return:
        """
        if node == target:
            return True, visited_edges

        next_nodes = self.neighbours(node)
        for u in next_nodes:
            if visited_edges[(node, u)]:
                continue

            # mark next node as visited
            visited_edges[(node, u)] = True
            # visit next node
            flag, visited_edges = self.modified_dfs(u, target, visited_edges)
            if flag:
                return True, visited_edges

        return False, visited_edges

    def mincut_between_nodes(self, source, target):
        """ Returns the set of nodes at distance at most dist from root """
        visited_edges = defaultdict(bool)  # boolean vector of visits
        count = 0

        for node in self.neighbours(source):
            if visited_edges[(source, node)]:
                continue
            flag, visited_edges = self.modified_dfs(node, target, visited_edges)
            if flag:
                count += 1

        return count

    # ---------------- RQ 5 ---------------- #

    def distance_vector(self, root):
        """ Returns the distance vector of each node from root """
        visited = defaultdict(bool)  # boolean vector of visits
        frontier = defaultdict(list)
        frontier[1] = self.neighbours(root)
        distances_by_cat = defaultdict(list)
        i = 1
        distances = dict()

        while True:
            if len(frontier[i]) == 0:  # no more nodes to visit
                break
            for u in frontier[i]:
                distances[u] = i

                for v in self.neighbours(u):
                    if not visited[v]:
                        visited[v] = True
                        frontier[i + 1].append(v)
            i += 1

        return distances

    def get_closest_categories(self, source_cat_id):
        """ Returns the list of categories sorted by their distance from cat_id """
        distances_by_cat = defaultdict(list)

        for node in tqdm(self.get_nodes_from_category_id(source_cat_id)):
            # vector of distances between each vertex and u
            distances = self.distance_vector(node)

            for target_node, dist in distances.items():
                # save node distance grouped by category
                target_cat_id = self.get_node_category(target_node)
                distances_by_cat[target_cat_id].append(dist)

        medians = dict()

        for target_cat_id, dist_list in distances_by_cat.items():
            medians[target_cat_id] = np.median(dist_list)

        del medians[source_cat_id]  # remove source category

        sorted_cat = sorted(zip(self.categories_list(), medians.values()),
                            key=lambda x: x[1])
        return [x[0] for x in sorted_cat]

    # ---------------- RQ 6 ---------------- #

    def get_category_subgraph(self):
        """ Returns the graph representing the connection between categories

        :return:
        """
        edge_list = set()

        for u, v in self.edge_list:
            edge_list.add((self.get_node_category(u), self.get_node_category(v)))

        subgraph = Graph()
        subgraph.create_from_edgelist(edge_list)
        return subgraph

    def get_adjacency_matrix(self, normalize=False):
        """ Returns a numpy adjacency matrix

        :param normalize: bool
                          if True normalizes each row of the matrix
                          useful when called by pagerank()
        :return:
        """
        n = self.max_node_id() + 1
        try:
            M = np.identity(n)
        except MemoryError:
            print("[Memory Error] Underlying matrix too large.")
            return None

        for u, v in self.edge_list:
            M[u, v] = 1

        # normalize matrix
        if normalize:
            M = M / M.sum(axis=0)

        return M

    def pagerank(self, n_iter=100, d=.85):
        """ Returns the pagerank ranking of the nodes
            The code is an adaptation of the one found on
            https://en.wikipedia.org/wiki/PageRank

        :param n_iter: int
                       maximum number of iterations for convergence
        :param d: float
                  damping factor
        :return: list of floats
        """
        n = self.max_node_id() + 1
        M = self.get_adjacency_matrix(normalize=True)
        v = np.random.rand(n, 1)
        v = v / np.linalg.norm(v, 1)
        M_hat = (d * M + (1 - d) / n)

        for i in range(n_iter):
            v = M_hat @ v

        return v

    def sort_category_by_pagerank(self):
        """ Returns the list of categories sorted according to their pagerank score

        :return: list of strings
        """
        g_cat = self.get_category_subgraph()
        pr = g_cat.pagerank()
        categories = self.categories_list()
        sorted_categories = sorted(zip(categories, pr), key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_categories]

    # ---------------- RQ 1 ---------------- #

    def plot_degree_distribution(self):
        neighbour_size_list = self.neighbour_sizes()
        counter = Counter()
        counter.update(neighbour_size_list)
        degrees = list(counter.keys())
        freq = list(counter.values())

        splot = sns.scatterplot(x=degrees, y=freq)
        _ = splot.set(title="Log-log scale Degree Distribution",
                      xscale="log", yscale="log",
                      xlabel="Node Degree", ylabel="Number of Nodes")

    def random_targets(self, cat_id, size):
        nodes = self.get_nodes_from_category_id(cat_id)
        ids = np.random.randint(0, len(nodes), size)
        return list(np.array(nodes)[ids])


# ---------------- RQ 6 ---------------- #

def sorted_by_pagerank(graph, category_list):
    pr = graph.pagerank()
    sort = sorted(zip(category_list, pr), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sort]
