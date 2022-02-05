from graphs.BayesNet import *
import itertools

class ImaginedBayesNet(BayesNet):
    """
    Note: This class uses code in directories: nodes, graphs, potentials,
    which was taken from my app Quantum Fog
    https://github.com/artiste-qb-net/quantum-fog

    This class takes as input a BayesNet object called 'in_bnet' with nodes
    X, Y and an arrow X->Y.  This class modifies in_bnet. The new bnet is
    assigned to self of this class and is called an ``imagined bnet".

    Let trols_list be a list of control nodes. trol nodes should be selected
    judiciously from the set of all nodes of in_bnet. See Judea Pearl's
    advice on godd and bad controls:
    https://ftp.cs.ucla.edu/pub/stat_ser/r493.pdf

    As usual, we will assume 2 cases: only observational data (
    self.only_obs=True) and both observational and experimental data (
    self.only_obs=False). In case of only experimental data, use an in_bnet
    with no X parents, and assume only_obs=True.

    This class modifies the in_bnet structure as follows:

    1. it adds two new nodes Y0 and Y1 and a selection bias node Y0Y1. Y0Y1
    is given parents Y0, Y1 and no children. It's purpose is to introduce an
    "explaining away" correlation between nodes Y0 and Y1.

    2. it removes all parents of nodes X and Y and makes trols_list the new
    parents of X and (Y0,Y1,X) the new parents of Y. The children of nodes X
    and Y remain the same.

    3. if experimental and observational data (self.only_obs=False),
    it makes (trol_list, X) the parents of both Y0 and Y1. If only
    observational data (self.only_obs=True), it makes trol_list the parents
    of both Y0 and Y1. Hence, X is a parent of both Y0 and Y1 iff
    self.only_obs=True. Why these 2 cases are treated differently: if there
    is both observational and experimental data, there is twice as much data
    to fit into the TPMs (transition probability matrices) of nodes Y0 and
    Y1, so Y0 and Y1 need an extra parent to enlarge the TPM so that it can
    fit twice as much data. Regardless of the value of self.only_obs,
    this class makes(Y, Y0Y1) the children of both Y0 and Y1.

    This class assigns a TPM (transition probability matrix) to each node of
    the imagined bnet as follows:

    The TPM of each node is stored in an object of class Potential. ALL
    nodes of the following DISJOINT types, depending on how their pot is
    assigned:

    fixed nodes, random nodes, control nodes, Y, X, Y0, Y1

    Fixed nodes are assigned a pot by in_bnet, and it never changes.

    Random nodes are assigned a random pot. Class Bounder_MC uses an
    imagined bnet (i.e., an object of this class) and switches from one
    "world" to the next one by randomizing all the nodes in the random nodes
    list. Note Y0Y1 is treated as a random node.

    Control nodes have only one active state at a time; the active state has
    unit probablity, all other states of the node have zero probability.
    Class Bounder_MC uses an imagined bnet (i.e., an object of this class)
    and samples all possible active states for all control nodes.

    The pot for node Y is assigned so that Y=(1-X)*Y0 + X*Y1, where X, Y,
    Y0 and Y1 \in {0,1}.

    The pots for nodes X, Y0 and Y1 are fully specified by the input
    dictionary trol_coords_to_oe_data which maps trol_coords to oe_data.
    trol_coords is a tuple of states for the control nodes. oe_data is a
    list of five probabilities oe_data=[o1b0, o1b1, px1, e1b0, e1b1],
    where o1b0=O_{1|0}, o1b1=O_{1|1}, px1=P(x=1)=E_{1|0}, e1b1=E_{1|1}.
    This notation for probabities is explained in the chapter on
    "Personalized Treatment Effects" of my book Bayesuvius.

    Attributes
    ----------
    fixed_nd_list : list[BayesNode]
        list of fixed nodes
    nd_X : BayesNode
    nd_Y : BayesNode
    nd_Y0 : BayesNode
    nd_Y1 : BayesNode
    nd_Y0Y1 : BayesNode
    only_obs : bool
        True iff only observational data, no experimental data
    random_nd_list : list[BayesNode]
        list of random nodes
    trol_coords_to_oe_data: dict[tuple[int], list[float]]
        This dictionary maps trol_coords to oe_data. trol_coords is a tuple
        of states of the control nodes, in the order given by trol_list.
        oe=observational-experimental. oe_data is a list of 5 probabilities
        [ o1b0, o1b1, px1, e1b0, e1b1]
    trol_list : list[BayesNode]
        list of control nodes
    use_Y0Y1 : bool
        True iff use selection bias node Y0Y1

    """

    def __init__(self,
                 in_bnet,
                 fixed_nd_list,
                 trol_list,
                 trol_coords_to_oe_data,
                 use_Y0Y1=True,
                 only_obs=False):
        """

        Parameters
        ----------
        in_bnet : BayesNet
        fixed_nd_list : list[BayesNode]
        trol_list : list[BayesNode]
        trol_coords_to_oe_data : dict[tuple[int], list[float]]
        use_Y0Y1 : bool
        only_obs : bool
        """
        BayesNet.__init__(self, in_bnet.nodes)
        self.nd_X= self.get_node_named("X")
        self.nd_Y = self.get_node_named("Y")
        assert self.nd_X.has_child(self.nd_Y),\
            "node X must have node Y as child"

        # these nodes will be created by build_self()
        self.nd_Y0=None
        self.nd_Y1=None
        self.nd_Y0Y1 = None

        self.fixed_nd_list = fixed_nd_list
        self.trol_list = trol_list
        assert set(trol_list).isdisjoint(set(fixed_nd_list)),\
            "the sets of fixed nodes and control nodes must be disjoint"

        assert self.nd_X not in self.fixed_nd_list \
            and self.nd_Y not in self.fixed_nd_list,\
            "nodes X and Y cannot be fixed nodes"

        assert self.nd_X not in self.trol_list \
            and self.nd_Y not in self.trol_list,\
            "nodes X and Y cannot be control nodes"

        self.random_nd_list = list(in_bnet.nodes - set(fixed_nd_list)\
                               -set(trol_list) - {self.nd_X, self.nd_Y})

        self.trol_coords_to_oe_data = trol_coords_to_oe_data
        self.use_Y0Y1 = use_Y0Y1
        self.only_obs = only_obs
        self.build_self()

    def build_self(self):
        """
        Modifies self from input in_bnet to imagined bnet.

        Returns
        -------
        None

        """
        # define new nodes Y0, Y1, Y0Y1 and add them to bnet
        id_nums = [nd.id_num for nd in self.nodes]
        max_id = max(id_nums)
        self.nd_Y0 = BayesNode(max_id + 1, "Y0")
        self.nd_Y1 = BayesNode(max_id + 2, "Y1")
        self.add_nodes({self.nd_Y1, self.nd_Y0Y1})
        if self.use_Y0Y1:
            self.nd_Y0Y1 = BayesNode(max_id + 3, "Y0Y1")
            self.add_nodes({self.nd_Y0Y1})

        # define structure
        if not self.only_obs:
            self.nd_Y0.add_parent(self.nd_X)
            self.nd_Y1.add_parent(self.nd_X)
        self.nd_Y0.add_parents(self.trol_list)
        self.nd_Y1.add_parents(self.trol_list)
        if self.use_Y0Y1:
            self.nd_Y0Y1.add_parents([self.nd_Y0, self.nd_Y1])

        for nd in list(self.nd_Y.parents):
            self.nd_Y.remove_parent(nd)
        self.nd_Y.add_parents([self.nd_X, self.nd_Y0, self.nd_Y1])

        for nd in list(self.nd_X.parents):
            self.nd_X.remove_parent(nd)
        self.nd_X.add_parents(self.trol_list)

        # define pot for Y
        self.nd_Y.potential = DiscreteCondPot(False,
            [self.nd_X, self.nd_Y0, self.nd_Y1, self.nd_Y])
        self.nd_Y.potential.pot_arr = np.zeros(shape=(2, 2, 2, 2),
                                               dtype=np.float64)
        for prod in itertools.product([0,1], repeat=4):
            x, y0, y1, y = prod
            if y == y0*(1-x) + y1*x:
                self.nd_Y.potential.pot_arr[x, y0, y1, y] = 1

        # define pot for Y0Y1 and add Y0Y1 to random node list
        if self.use_Y0Y1:
            self.nd_Y0Y1.potential = DiscreteCondPot(False,
                [self.nd_Y0, self.nd_Y1, self.nd_Y0Y1] )
            self.nd_Y0Y1.potential.pot_arr = np.zeros(shape=(2,2,4),
                                         dtype=np.float64)
            self.nd_Y0Y1.potential.set_to_random()
            self.nd_Y0Y1.potential.normalize_self()
            self.random_nd_list.append(self.nd_Y0Y1)
        
        # define pots without pot_arr for Y0, Y1, X
        for nd in [self.nd_Y0, self.nd_Y1]:
            nd_list = list(self.trol_list)
            nd_size_list = [nd1.size for nd1 in self.trol_list]
            if not self.only_obs:
                nd_list.append(self.nd_X)
                nd_size_list.append(2)
            nd_list.append(nd)
            nd_size_list.append(2)
            nd.potential = DiscreteCondPot(False, nd_list)
            nd.potential.pot_arr = np.zeros(shape=nd_size_list,
                                      dtype=np.float64)
        nd_list = list(self.trol_list)
        nd_size_list = [nd1.size for nd1 in self.trol_list]
        nd_list.append(self.nd_X)
        nd_size_list.append(2)
        self.nd_X.potential = DiscreteCondPot(False, nd_list)
        self.nd_X.potential.pot_arr = np.zeros(shape=nd_size_list,
                                               dtype=np.float64)
        # define pot_arr for nodes Y0, Y1, X
        for coords, oe_data in self.trol_coords_to_oe_data.items():
            assert len(coords) == len(self.trol_list), \
                "trol_coords in trol_coords_to_oe_data have wrong length"
            assert len(oe_data) == 5 \
                   and max(oe_data)<=1 \
                   and min(oe_data)>=0,\
                "oe_data in trol_coords_to_oe_data must be 5 probabilities"
            oybx, px, eybx = ImaginedBayesNet.get_oe_arrays(oe_data)
            # print("xxxxccc", oybx, px, eybx)

            for x in [0, 1]:
                coords_plus = tuple(list(coords)+[x])
                self.nd_X.potential.pot_arr[coords_plus]= px[x]

            if not self.only_obs:
                for c, x, y in itertools.product([0,1], repeat=3):
                    # P(Y_x=y|X=c, trol_coords)
                    if x == 0:
                        nd = self.nd_Y0
                    else:
                        nd = self.nd_Y1
                    coords_plus = tuple(list(coords)+[c,y])
                    if c==x:
                        prob = oybx[y, c]
                    else:
                        prob = (eybx[y, c] - oybx[y, c]*px[c])/px[1-c]
                        assert prob >= 0,\
                            "invalid oe_data"
                    nd.potential[coords_plus] = prob
                    if list(coords)[0]==0 and x==0:
                        print("nnbbh---coords,x,y,y_x=y at c, prob",
                              coords_plus,
                              f"y_{x}={y} at c={c}", prob)
            else:
                for x, y in itertools.product([0, 1], repeat=2):
                    # P(x, y)
                    if x == 0:
                        nd = self.nd_Y0
                    else:
                        nd = self.nd_Y1
                    coords_plus = tuple(list(coords) + [y])
                    nd.potential[coords_plus] = oybx[y, x]
                    
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
        in_bnet = BayesNet(nodes)

        # in general
        # DiscreteCondPot(False, [y1, y2, y3, x]) refers to P(x| y1, y2, y3)
        cl.potential = DiscreteUniPot(False, cl)  # P(a)
        sp.potential = DiscreteCondPot(False, [cl, sp])  # P(b| a)
        nd_X.potential = DiscreteCondPot(False, [cl, nd_X])
        nd_Y.potential = DiscreteCondPot(False, [sp, cl, nd_X, nd_Y])
        for nd in nodes:
            nd.potential.set_to_random()
            nd.potential.normalize_self()

        in_bnet.draw(algo_num=1)

        fixed_nd_list = [cl]
        trol_list = [sp]
        trol_coords_to_oe_data = {
            (0,):[.48,.51,.53,.47,.5],
            (1,):[.5,.5,.5,.5,.5]
        }
        imagined_bnet = ImaginedBayesNet(in_bnet,
                                         fixed_nd_list,
                                         trol_list,
                                         trol_coords_to_oe_data,
                                         only_obs=False)
        imagined_bnet.draw(algo_num=1)
        for nd in imagined_bnet.nodes:
            print(nd.name, ", parents=" + str([x.name for x in nd.parents]),
                  ", children=" + str([x.name for x in nd.children]))
            print(nd.potential.pot_arr)
            print()
        path1 = 'examples_cbnets/tempo.dot'
        imagined_bnet.write_dot(path1)
        graph =Source(open(path1).read())
        graph.view(filename='examples_cbnets/tempo.gv')

    main()







