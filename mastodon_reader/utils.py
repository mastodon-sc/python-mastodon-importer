import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

### Matlabs `ismember()` clone taken  from
### https://github.com/erdogant/ismember/blob/master/ismember/ismember.py
### MIT licence
def ismember(a_vec, b_vec, method=None):
    """

    Description
    -----------
    MATLAB equivalent ismember function
    [LIA,LOCB] = ISMEMBER(A,B) also returns an array LOCB containing the
    lowest absolute index in B for each element in A which is a member of
    B and 0 if there is no such index.
    Parameters
    ----------
    a_vec : list or array
    b_vec : list or array
    method : None or 'rows' (default: None).
        rows can be used for row-wise matrice comparison.
    Returns an array containing logical 1 (true) where the data in A is found
    in B. Elsewhere, the array contains logical 0 (false)
    -------
    Tuple

    Example
    -------
    >>> a_vec = np.array([1,2,3,None])
    >>> b_vec = np.array([4,1,2])
    >>> Iloc,idx = ismember(a_vec,b_vec)
    >>> a_vec[Iloc] == b_vec[idx]

    """
    # Set types
    a_vec, b_vec = _settypes(a_vec, b_vec)

    # Compute
    if method is None:
        Iloc, idx = _compute(a_vec, b_vec)
    elif method == "rows":
        if a_vec.shape[0] != b_vec.shape[0]:
            raise Exception("Error: Input matrices should have same number of columns.")
        # Compute row-wise over the matrices
        out = list(map(lambda x, y: _compute(x, y), a_vec, b_vec))
        # Unzipping
        Iloc, idx = list(zip(*out))
    else:
        Iloc, idx = None, None

    return (Iloc, idx)


def _settypes(a_vec, b_vec):
    if "pandas" in str(type(a_vec)):
        a_vec.values[np.where(a_vec.values == None)] = "NaN"
        a_vec = np.array(a_vec.values)
    if "pandas" in str(type(b_vec)):
        b_vec.values[np.where(b_vec.values == None)] = "NaN"
        b_vec = np.array(b_vec.values)
    if isinstance(a_vec, list):
        a_vec = np.array(a_vec)
        # a_vec[a_vec==None]='NaN'
    if isinstance(b_vec, list):
        b_vec = np.array(b_vec)
        # b_vec[b_vec==None]='NaN'

    return a_vec, b_vec


def _compute(a_vec, b_vec):
    bool_ind = np.isin(a_vec, b_vec)
    common = a_vec[bool_ind]
    [common_unique, common_inv] = np.unique(common, return_inverse=True)
    [b_unique, b_ind] = np.unique(b_vec, return_index=True)
    common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]

    return bool_ind, common_ind[common_inv]


def RGBint2RGB(rgb_int):
    """Convert Java integer color to RGB

    Args:
        rgb_int (int): color value stored as int

    Returns:
        tuple[int]: RGB value in uint8
    """ """"""
    B = rgb_int & 255
    G = (rgb_int >> 8) & 255
    R = (rgb_int >> 16) & 255
    return (R, G, B)


def hierarchy_pos(G, root=None, width=1.0, vert_gap=-1.0, vert_loc=0, xcenter=0.5):

    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    """

    import networkx as nx
    import random

    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(
                iter(nx.topological_sort(G))
            )  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(
        G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None
    ):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


class Lineage:
    """Experimental class for lineage analysis based on NetworkX trees

    :alert work in progress
    
    """
    def __init__(self, nx_tree):
        self.nx_tree = nx_tree
        # remove_wrong_splits(nx_tree, 16)

    def __getitem__(self, node):
        return self.nx_tree.nodes[node]

    @property
    def root(self):
        return next(iter(nx.topological_sort(self.nx_tree)))

    @property
    def split_nodes(self):
        return dict(
            [(v, self.nx_tree.nodes[v]) for v, d in self.nx_tree.out_degree() if d == 2]
        )

    @property
    def terminal_nodes(self):
        return dict(
            [(v, self.nx_tree.nodes[v]) for v, d in self.nx_tree.out_degree() if d == 0]
        )

    @property
    def division_vecs(self):
        division_vecs = {}
        for s in self.split_nodes.keys():

            diff = self.division_vector(s)
            division_vecs[s] = diff

        return division_vecs

    def division_vector(self, s):
        b1, b2 = self.nx_tree.successors(s)
        b1_vec = np.array([self[b1][a] for a in ["x", "y", "z"]])
        b2_vec = np.array([self[b2][a] for a in ["x", "y", "z"]])

        diff = b1_vec - b2_vec
        return diff

    def node_positions(self):
        pos = hierarchy_pos(self.nx_tree)

        return pos

    def draw(self, ax=None):
        if ax is None:
            f, ax = plt.subplots()

        nx.draw_networkx(
            self.nx_tree,
            pos=self.node_positions(),
            arrows=False,
            with_labels=False,
            node_size=1,
            ax=ax,
        )

    @staticmethod
    def clone(self):
        return self.__class__(self.nx_tree)

    def all_split_paths_iter(self):
        split_nodes = list(self.split_nodes.keys())
        for node in self.terminal_nodes:
            path = nx.shortest_path(self.nx_tree, self.root, node)
            yield list(filter(lambda n: n in split_nodes, path))
