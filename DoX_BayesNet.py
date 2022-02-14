from graphs.BayesNet import *
import itertools
from graphviz import Source


class DoX_BayesNet(BayesNet):
    """
    Note: This class uses code in directories: nodes, graphs, potentials,
    which was taken from my app Quantum Fog
    https://github.com/artiste-qb-net/quantum-fog

    This class takes as input a BayesNet object called 'in_bnet' with nodes
    X, Y and an arrow X->Y, and then it modifies in_bnet. The new bnet is
    assigned to self of this class and is called a "doX bnet".

    Let trols_list be a list of control nodes. trol nodes should be selected
    judiciously from the set of all nodes of in_bnet. See Judea Pearl's
    advice on good and bad controls:
    https://ftp.cs.ucla.edu/pub/stat_ser/r493.pdf

    This class modifies the in_bnet structure as follows:

    1. it removes all parents of node X

    This class assigns a TPM (transition probability matrix) to each node of
    the doX bnet as follows:

    The TPM of each node is stored in an object of class Potential. ALL
    nodes belong to the following DISJOINT sets, depending on how their pot
    is assigned:

    unobserved nodes, control nodes, X, other nodes

    unobserved nodes are assigned a random pot. Class IdentifiabilityChecker
    uses a doX bnet (i.e., an object of this class) and switches from one
    "world" to the next one by randomizing all the nodes in the unobserved
    nodes list.

    Each control node has only one active state. The active state has unit
    probablity, all other states of the node have zero probability. The
    active state of the k'th node in trol_nds is given by the k'th int in
    the tuple trol_coords.

    Node X has a single active state given by x_val.

    All other nodes (i.e., not in the sets of unobserved nodes, control
    nodes, or X) have the same pot as they do in in_bnet, and it never
    changes.

    Attributes
    ----------
    nd_X : BayesNode
    nd_Y : BayesNode
    trol_list : list[BayesNode]
        list of control nodes
    unobs_nd_list : list[BayesNode]
        list of  unobserved nodes
    x_val : int

    """
    def __init__(self,
                 in_bnet,
                 unobs_nd_list,
                 trol_list,
                 x_val):
        """
        Constructor.

        Parameters
        ----------
        in_bnet : BayesNet
        unobs_nd_list : list[BayesNode]
        trol_list : list[BayesNode]
        x_val : int
        trol_coords : tuple(int)
        """
        BayesNet.__init__(self, in_bnet.nodes)
        self.nd_X = self.get_node_named("X")
        self.nd_Y = self.get_node_named("Y")
        assert self.nd_X.has_child(self.nd_Y), \
            "node X must have node Y as child"
        # For simplicity, assume size of node Y is 2, node X of any size
        assert self.nd_Y.size == 2, \
            "Node Y must have 2 states only."

        self.unobs_nd_list = unobs_nd_list
        self.trol_list = trol_list
        assert set(trol_list).isdisjoint(set(unobs_nd_list)), \
            "the sets of unobserved nodes and control nodes must be disjoint"

        assert self.nd_X not in self.unobs_nd_list \
                and self.nd_Y not in self.unobs_nd_list, \
                "nodes X and Y cannot be unobserved nodes"

        assert self.nd_X not in self.trol_list \
                and self.nd_Y not in self.trol_list, \
                "nodes X and Y cannot be control nodes"

        self.x_val = x_val
        assert self.x_val in list(range(self.nd_X.size)),\
            "x_val is not an integer between 0 and size-1 of node X"
        self.build_self()

    def build_self(self):
        """
        Modifies self from input in_bnet to doX bnet.

        Returns
        -------
        None

        """
        for nd in list(self.nd_X.parents):
            self.nd_X.remove_parent(nd)
        self.topological_sort()

        # define new pots for X and for the control nodes
        pot_arr = np.zeros(shape=(self.nd_X.size, ))
        pot_arr[self.x_val] = 1.0
        self.nd_X.potential = DiscreteUniPot(False, self.nd_X,
                                             pot_arr=pot_arr)

    def refresh_unobs_nodes(self):
        """
        This method randomizes yet again all pots of the unobserved nodes.

        Returns
        -------
        None

        """
        for nd in self.unobs_nd_list:
            nd.potential.set_to_random()
            nd.potential.normalize_self()

    @staticmethod
    def build_test_doX_bnet(draw=False):
        """
        This method builds a doX bnet for testing purposes from a simple
        example of an input bnet.

        Parameters
        ----------
        draw : bool
            True iff draw both input bnet and imagined bnet

        Returns
        -------
        DoX_BayesNet

        """
        nd_X = BayesNode(0, name="X")
        nd_Y = BayesNode(1, name="Y")
        nd_Z = BayesNode(2, name="Z")
        nd_U = BayesNode(3, name="U")

        nd_Y.add_parents({nd_X, nd_Z})
        nd_X.add_parents({nd_Z, nd_U})
        nd_Z.add_parent(nd_U)

        nodes = {nd_X, nd_Y, nd_Z, nd_U}
        in_bnet = BayesNet(nodes)

        # in general
        # DiscreteCondPot(False, [y1, y2, y3, x]) refers to P(x| y1, y2, y3)
        nd_U.potential = DiscreteUniPot(False, nd_U)  # P(a)
        nd_X.potential = DiscreteCondPot(False, [nd_Z, nd_U, nd_X])  # P(b| a)
        nd_Y.potential = DiscreteCondPot(False, [nd_Z, nd_X, nd_Y])
        nd_Z.potential = DiscreteCondPot(False, [nd_U, nd_Z])
        for nd in nodes:
            nd.potential.set_to_random()
            nd.potential.normalize_self()
        if draw:
            in_bnet.draw(algo_num=1)

        unobs_nd_list = [nd_U]
        trol_list = [nd_Z]
        doX_bnet = DoX_BayesNet(in_bnet,
                                unobs_nd_list,
                                trol_list,
                                x_val=0)
        if draw:
            doX_bnet.draw(algo_num=1)
            path1 = './tempo.dot'
            doX_bnet.write_dot(path1)
            graph = Source(open(path1).read())
            graph.view(filename='./tempo.gv')
        return doX_bnet


if __name__ == "__main__":

    def main():
        doX_bnet = DoX_BayesNet.build_test_doX_bnet(draw=True)
        for nd in doX_bnet.nodes:
            print(nd.name, ", parents=" + str([x.name for x in nd.parents]),
                  ", children=" + str([x.name for x in nd.children]))
            print(nd.potential.pot_arr)
            print()

    main()
