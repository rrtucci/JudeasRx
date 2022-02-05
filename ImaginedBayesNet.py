from graphs.BayesNet import *
import itertools

class ImaginedBayesNet(BayesNet):
    """
    Note: This class uses code in directories: nodes, graphs,
    potentials, which was taken from my app Quantum Fog
    https://github.com/artiste-qb-net/quantum-fog

    This class takes as input a BayesNet object called 'in_bnet' with nodes
    X, Y and an arrow X->Y. Let Pa(Y) -X be parents of Y other than X. We
    assume that in in_bnet,  Pa(Y)-X is a single node Z, or empty. If Pa(
    Y)-X is larger than one node, please combine them into a single node Z
    and add arrow Z->Y.

    As usual, we will assume 2 cases: only observational data (
    self.only_obs=True) and both observational and experimental data (
    self.only_obs=False. In case of only experimental data, use an in_bnet
    with no X parents, and assume only_obs=True.

    This class modifies in_bnet as follows:

    1. it adds two new nodes Y0 and Y1

    1. if Z exists, it removes the arrow Z->Y, and adds arrows Z->Y0->Y,
    Z->Y1->Y. If Z doesn't exist, it adds arrows Y0->Y and Y1->Y.

    3. if experimental and observational data (self.only_obs=False), it adds
    arrows X->Y0, X->Y1. If only observational data (self.only_obs=False),
    it does not add those two arrows.

    The new bnet is assigned to self of this class and is called an
    ``imagined bnet".

    The TPM (transition probability matrix) of each node is stored in an
    object of class Potential. Nodes can have either a known pot,
    or a random one. This class stores a list 'random nodes' of the nodes
    with a random pot. Class Bounder_MC uses an imagined bnet (i.e.,
    an object of this class) and switches from one "world" to the next one
    by randomizing all the nodes in the 'random nodes' list.

    Non-random nodes can be control nodes. Such nodes have only one active
    state at a time; the active state has unit probablity, all other states
    of the node have zero probability. This class stores a dictionary
    'trol_nd_to_state' that maps each control node to its active state.
    Class Bounder_MC uses an imagined bnet (i.e., an object of this class)
    and samples all possible trol_nd_to_state.values() for a fixed
    trol_nd_to_state.keys()

    As mentioned earlier, this class adds to in_bnet two new nodes  Y0,
    Y1. The pots of these nodes are known. They are fully specified by
    oe_data=[ o1b0z, o1b1z, px1z, e1b0z, e1b1z], where o1b0z=O_{1|0, z},
    o1b1=O_{ 1|1, z}, px1bz=P( x=1|z), z=E_{1|0, z}, e1b1z=E_{1|1,
    z}. o1b0z, o1b1z, px1z, e1b0z, e1b1z are 5 arrays of shape = (
    number of states of Z, ). If Z doesn't exist, these 5 arrays become 5
    scalars.

    Attributes
    ----------
    trol_nd_to_state : dict[BayesNode, int]
        maps control nodes to their active state
    nd_X : BayesNode
    nd_XY : BayesNode
    nd_XY0 : BayesNode
    nd_XY1 : BayesNode
    nd_Y : BayesNode
    nd_Y0 : BayesNode
    nd_Y1 : BayesNode
    oe_data : list[float]
        [o1b0z, o1b1z, px1z, e1b0z, e1b1z], oe=observational-experimental,
        dim(o1b0z)= dim(o1b1z)=dim(px1z)=dim(e1b0z)=dim(e1b1z)=size of node Z,
        or 1 if in_bnet has no Z.
    only_obs : bool
        True iff only observational data, no experimental data
    random_nodes : list[BayesNode]
    size_Z : int
        size (i.e., num of states) of node Z. Equal to 1 if no Z node.

    """

    def __init__(self,
                 in_bnet,
                 fixed_nd_list,
                 trol_nd_list,
                 trol_nd_coords_to_oe_data,
                 only_obs=False):
        """

        Parameters
        ----------
        in_bnet : BayesNet
        random_nodes : list[BayesNode]
        trol_nd_to_state : dict[BayesNode, int]
        size_Z : int
        oe_data : list[float]
        only_obs : bool
        """
        BayesNet.__init__(self, in_bnet.nodes)
        self.nd_X= self.get_node_named("X")
        self.nd_Y = self.get_node_named("Y")
        assert self.nd_X.has_child(self.nd_Y),\
            "node X must have node Y as child"

        # these nodes will be created
        self.nd_Y0=None
        self.nd_Y1=None
        self.nd_Y0Y1 = None

        self.fixed_nd_list = fixed_nd_list
        self.trol_nd_list = trol_nd_list
        assert set(trol_nd_list).isdisjoint(set(fixed_nd_list)),\
            "the sets of fixed nodes and control nodes must be disjoint"

        assert self.nd_X not in self.fixed_nd_list \
            and self.nd_Y not in self.fixed_nd_list,\
            "nodes X and Y cannot be fixed nodes"

        assert self.nd_X not in self.trol_nd_list \
            and self.nd_Y not in self.trol_nd_list,\
            "nodes X and Y cannot be control nodes"

        self.random_nd_list = list(in_bnet.nodes - set(fixed_nd_list)\
                               -set(trol_nd_list) - {self.nd_X, self.nd_Y})

        self.only_obs = only_obs
        self.trol_nd_coords_to_oe_data = trol_nd_coords_to_oe_data
        self.build_self()

    def build_self(self):
        """
        Modify self from input bnet to imagined bnet.

        Returns
        -------
        None

        """
        # first define structure
        id_nums = [nd.id_num for nd in self.nodes]
        max_id = max(id_nums)
        self.nd_Y0 = BayesNode(max_id + 1, "Y0")
        self.nd_Y1 = BayesNode(max_id + 2, "Y1")
        self.nd_Y0Y1 = BayesNode(max_id + 3, "Y0Y1")
        self.add_nodes({self.nd_Y0, self.nd_Y1, self.nd_Y0Y1})

        if not self.only_obs:
            self.nd_Y0.add_parent(self.nd_X)
            self.nd_Y1.add_parent(self.nd_X)
        self.nd_Y0.add_parents(self.trol_nd_list)
        self.nd_Y1.add_parents(self.trol_nd_list)
        self.nd_Y0Y1.add_parents([self.nd_Y0, self.nd_Y1])

        for nd in list(self.nd_Y.parents):
            self.nd_Y.remove_parent(nd)
        self.nd_Y.add_parents([self.nd_X, self.nd_Y0, self.nd_Y1])

        for nd in list(self.nd_X.parents):
            self.nd_X.remove_parent(nd)
            
        # next define pot for Y
        self.nd_Y.potential = DiscreteCondPot(False,
            [self.nd_X, self.nd_Y0, self.nd_Y1, self.nd_Y])
        self.nd_Y.potential.pot_arr = np.zeros(shape=(2, 2, 2, 2),
                                               dtype=np.float64)
        for prod in itertools.product([0,1], repeat=4):
            x, y0, y1, y = prod
            if y == y0*(1-x) + y1*x:
                self.nd_Y.potential.pot_arr[x, y0, y1, y] = 1
        # define pot for Y0Y1
        self.nd_Y0Y1.potential = DiscreteCondPot(False,
            [self.nd_Y0, self.nd_Y1, self.nd_Y0Y1] )
        self.nd_Y0Y1.potential.pot_arr = np.zeros(shape=(2,2,4),
                                     dtype=np.float64)
        self.nd_Y0Y1.potential.set_to_random()
        self.nd_Y0Y1.potential.normalize_self()
        self.random_nd_list.append(self.nd_Y0Y1)
        
        # define pots for Y0, Y1, X
        for nd in [self.nd_Y0, self.nd_Y1]:
            nd_list =list(self.trol_nd_list)
            nd_size_list = [nd1.size for nd1 in self.trol_nd_list]
            if not self.only_obs:
                nd_list.append(self.nd_X)
                nd_size_list.append(2)
            nd_list.append(nd)
            nd_size_list.append(2)
            nd.potential = DiscreteCondPot(False, nd_list)
            nd.potential.pot_arr = np.zeros(shape=nd_size_list,
                                      dtype=np.float64)

        self.nd_X.potential = DiscreteUniPot(False, self.nd_X)
        self.nd_X.potential.pot_arr = np.zeros(shape=(2,),
                                        dtype=np.float64)

        for coords, oe_data in self.trol_nd_coords_to_oe_data:
            assert len(coords) == len(self.trol_nd_list), \
                "trol_nd_coords in trol_nd_coords_to_oe_data have wrong length"
            assert len(oe_data) == 5 \
                   and max(oe_data)<=1 \
                   and min(oe_data)>=0,\
                "oe_data in trol_nd_coords_to_oe_data must be 5 probabilities"
            oybx, px, eybx = ImaginedBayesNet.get_oe_arrays(oe_data)
            # print("xxxxccc", oybx, px, eybx)

            for x in [0, 1]:
                coords_plus = tuple(list(coords)+[x])
                self.nd_X.potential.pot_arr[coords_plus]= px[x]

            if not self.only_obs:
                for c, x, y in itertools.product([0,1], repeat=3):
                    # P(Y_x=y|X=c, trol_nd_coords)
                    if x == 0:
                        nd = self.nd_Y0
                    else:
                        nd = self.nd_Y1
                    coords_plus = tuple(list(coords)+[x,y])
                    if c==x:
                        nd.potential[coords_plus] = \
                            oybx[y, x]*px[x]
                    else:
                        nd.potential[coords_plus] = \
                            eybx[y, x] - oybx[y, x]*px[x]
            else:
                for x, y in itertools.product([0, 1], repeat=2):
                    # P(x, y)
                    if x == 0:
                        nd = self.nd_Y0
                    else:
                        nd = self.nd_Y1
                    nd.potential[x, y] = oybx[y, x]*px[x]
                    
    @staticmethod
    def get_oe_arrays(oe_data):
        """
        This method takes the five probabilities in the list oe_data and it
        returns 3 numpy arrays oybx=O_{y|x}, px=P(x), eybx=E_{y|x}.

        Parameters
        ----------
        oe_data : list[float]
            [o1b0, o1b1, px1, e1b0, e1b1]

        Returns
        -------
        np.array, np.array, np.array
            oybx, px, eybx
            shape=(2, 2), shape=(2, ), shape=(2, 2)

        """
        o1b0, o1b1, px1, e1b0, e1b1 = oe_data

        px = np.array([1 - px1, px1])
        oybx = np.array([[1 - o1b0, 1 - o1b1],
                         [o1b0, o1b1]])
        eybx = np.array([[1 - e1b0, 1 - e1b1],
                         [e1b0, e1b1]])
        return oybx, px, eybx

    def refresh_random_nodes(self):
        """
        This method randomizes yet again all random pots.

        Returns
        -------
        None

        """
        for nd in self.random_nd_list:
            nd.potential.set_to_random()
            nd.potential.normalize_self()

if __name__ == "__main__":
    from graphviz import Source
    from IPython.display import display

    def main():
        cl = BayesNode(0, name="Cloudy")
        sp = BayesNode(1, name="Sprinkler")
        nd_X = BayesNode(2, name="X")
        nd_Y = BayesNode(3, name="Y")

        nd_Y.add_parents({cl, sp, nd_X})
        nd_Y.add_parent(nd_X)
        sp.add_parent(cl)
        nd_X.add_parent(cl)

        nodes = {cl, nd_X, sp, nd_Y}

        # in general
        # DiscreteCondPot(False, [y1, y2, y3, x]) refers to P(x| y1, y2, y3)
        cl.potential = DiscreteUniPot(False, cl)  # P(a)
        sp.potential = DiscreteCondPot(False, [cl, sp])  # P(b| a)
        nd_X.potential = DiscreteCondPot(False, [cl, nd_X])
        nd_Y.potential = DiscreteCondPot(False, [sp, cl, nd_X, nd_Y])
        for nd in nodes:
            nd.potential.set_to_random()
            nd.potential.normalize_self()

        in_bnet = BayesNet(nodes)
        in_bnet.draw(algo_num=1)

        random_nodes = [cl]
        trol_node_to_state = {sp: 0}
        oe_data = [.3, .5, .7, .1, .9]
        imagined_bnet = ImaginedBayesNet(in_bnet,
                                         random_nodes,
                                         trol_node_to_state,
                                         oe_data,
                                         only_obs=True)
        imagined_bnet.draw(algo_num=1)
        for nd in imagined_bnet.nodes:
            print(nd.name, "parents=" + str([x.name for x in nd.parents]),
                  "children=" + str([x.name for x in nd.children]))
            print(nd.potential.pot_arr)
            print()
        path1 = 'examples_cbnets/tempo.dot'
        imagined_bnet.write_dot(path1)
        graph =Source(open(path1).read())
        graph.view(filename='examples_cbnets/tempo.gv')

    main()







