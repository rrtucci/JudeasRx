from graphs.BayesNet import *
import itertools

class ImaginedBayesNet(BayesNet):
    """
    Note: This class uses code in directories: nodes, graphs,
    potentials, which was taken from my app Quantum Fog
    https://github.com/artiste-qb-net/quantum-fog

    This class takes as input a BayesNet object called 'in_bnet' with nodes
    X, Y and an arrow X->Y. The class modifies in_bnet by

    1. removing all arrows entering Y except X. Call Pa(Y)-X= parents of X
    minus X

    2. adding two new nodes Y0 and Y1 and arrows [Pa(X)-X]->Y0->Y,
    [Pa(X)-X]->Y1->Y

    3. if experimental and observational data (self.only_obs=False), adding
    constraint nodes XY0, XY1 and arrows X->XY0<-Y0, X->XY1<-Y1

    4. if only observational data (self.only_obs=False), adding constraint
    node XY and arrows X->XY<-Y

    The new bnet is assigned to self of this class and is called an
    ``imagined bnet".

    The TPM (transition probability matrix) of each node is stored in an
    object of class Potential. Nodes can have either a known pot,
    or a random one. This class stores a list 'random nodes' of the nodes
    with a random pot. Class Bounder_MC uses an imagined bnet (i.e.,
    an object of this class) and switches from one "world" to the next one
    by randomizing all the nodes in the 'random nodes' list.

    Non-random nodes can be evidence nodes. Such nodes have only one active
    state at a time; the active state has unit probablity, all other states
    of the node have zero probability. This class stores a dictionary
    'evi_nd_to_state' that maps each evidence node to its active state.
    Class Bounder_MC uses an imagined bnet (i.e., an object of this class)
    and samples all possible evi_nd_to_state.values() for a fixed
    evi_nd_to_state.keys()

    As mentioned earlier, this class adds to in_bnet either one (XY) or two
    (XY0, XY1) constraint nodes. The pots of the constraint nodes are known.
    They are fully specified by the 5 independent probablities in the input
    list oe_data=[o1b0, o1b1, px1, e1b0, e1b1], where o1b0=O_{1|0}, o1b1=O_{
    1|1}, px1=P(x=1), e1b0=E_{1|0}, e1b1=E_{1|1}.

    Attributes
    ----------
    evi_nd_to_state : dict[BayesNode, int]
        maps evidence nodes to their active state
    nd_X : BayesNode
    nd_XY : BayesNode
    nd_XY0 : BayesNode
    nd_XY1 : BayesNode
    nd_Y : BayesNode
    nd_Y0 : BayesNode
    nd_Y1 : BayesNode
    oe_data : list[float]
        [o1b0, o1b1, px1, e1b0, e1b1], oe=observational-experimental
    only_obs : bool
        True iff only obsevational data, no experimental data
    random_nodes : list[BayesNode]

    """

    def __init__(self,
                 in_bnet,
                 random_nodes,
                 evi_nd_to_state,
                 oe_data,
                 only_obs=False):
        """

        Parameters
        ----------
        in_bnet : BayesNet
        random_nodes : list[BayesNode]
        evi_nd_to_state : dict[BayesNode, int]
        oe_data : list[float]
        only_obs : bool
        """
        BayesNet.__init__(self, in_bnet.nodes)

        self.random_nodes = random_nodes
        self.evi_nd_to_state= evi_nd_to_state
        assert set(evi_nd_to_state.keys()).isdisjoint(set(random_nodes)),\
            "the sets of evidence nodes and random nodes must be disjoint"
        self.oe_data = oe_data
        assert len(oe_data) == 5 and max(oe_data) <= 1 and min(oe_data) >=0,\
            str(oe_data) + "must contain 5 probabilities"
        self.only_obs = only_obs

        self.nd_X= self.get_node_named("X")
        self.nd_Y = self.get_node_named("Y")
        assert self.nd_X.has_child(self.nd_Y),\
            "node X must have node Y as child"

        # these nodes will be created
        self.nd_Y0=None
        self.nd_Y1=None
        self.nd_XY0 = None
        self.nd_XY1 = None
        self.nd_XY = None
        
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
        self.add_nodes({self.nd_Y0, self.nd_Y1})
        if not self.only_obs:
            self.nd_XY0 = BayesNode(max_id + 3, "XY0")
            self.nd_XY1 = BayesNode(max_id + 4, "XY1")
            self.add_nodes({self.nd_XY0, self.nd_XY1})
        else:
            self.nd_XY = BayesNode(max_id + 3, "XY")
            self.add_nodes({self.nd_XY})

        parents_Y_minus_X = set(self.nd_Y.parents)
        # parents_Y_minus_X contains nd_X at this point
        parents_Y_minus_X.remove(self.nd_X)
        self.nd_Y0.add_parents(parents_Y_minus_X)
        self.nd_Y1.add_parents(parents_Y_minus_X)

        for nd in list(self.nd_Y.parents):
            self.nd_Y.remove_parent(nd)
        self.nd_Y.add_parents([self.nd_X, self.nd_Y0, self.nd_Y1])

        if not self.only_obs:
            self.nd_XY0.add_parents([self.nd_X, self.nd_Y0])
            self.nd_XY1.add_parents([self.nd_X, self.nd_Y1])
        else:
            self.nd_XY.add_parents([self.nd_X, self.nd_Y])

        # next define pots for all new nodes
        ord_pa_Yi = list(parents_Y_minus_X)
        for nd in [self.nd_Y0, self.nd_Y1]:
            nd.potential = DiscreteCondPot(False, ord_pa_Yi + [nd])
            nd.potential.pot_arr = np.zeros(shape=(2,)*(len(ord_pa_Yi)+1),
                                      dtype=np.float64)
            nd.potential.set_to_random()
            nd.potential.normalize_self()
            self.random_nodes.append(nd)

        self.nd_Y.potential = DiscreteCondPot(False,
            [self.nd_X, self.nd_Y0, self.nd_Y1, self.nd_Y])

        pot_arr = np.zeros(shape=(2, 2, 2, 2), dtype=np.float64)
        for prod in itertools.product([0,1], repeat=4):
            x, y0, y1, y = prod
            if y == y0*(1-x) + y1*x:
                pot_arr[x, y0, y1, y] = 1
        self.nd_Y.potential.pot_arr = pot_arr

        if not self.only_obs:
            self.nd_XY0.potential = DiscreteCondPot(False,
                    [self.nd_X, self.nd_Y0, self.nd_XY0])
            self.nd_XY0.potential.pot_arr =\
                np.zeros(shape=(2, 2, 4), dtype=np.float64)

            self.nd_XY1.potential = DiscreteCondPot(False,
                    [self.nd_X, self.nd_Y1, self.nd_XY1])
            self.nd_XY1.potential.pot_arr =\
                np.zeros(shape=(2, 2, 4), dtype=np.float64)
        else:
            self.nd_XY.potential = DiscreteCondPot(False,
                    [self.nd_X, self.nd_Y, self.nd_XY])
            self.nd_XY.potential.pot_arr =\
                np.zeros(shape=(2, 2, 4), dtype=np.float64)

        self.refresh_oe_data(self.oe_data)

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

    def refresh_oe_data(self, oe_data):
        """
        This method replaces the oe_data stored in the class by the input
        oe_data. The oe_data in this class is stored in self.oe_data and in
        the pots of the constraint nodes.

        Parameters
        ----------
        oe_data : list[float]

        Returns
        -------
        None

        """

        self.oe_data = oe_data
        oybx, px, eybx = ImaginedBayesNet.get_oe_arrays(oe_data)
        # print("xxxxccc", oybx, px, eybx)

        if not self.only_obs:
            for c, x, y in itertools.product([0,1], repeat=3):
                # P(Y_x=y|X=c)
                xy = y + 2*x
                if x == 0:
                    nd = self.nd_XY0
                else:
                    nd = self.nd_XY1
                if c==x:
                    nd.potential[x, y, xy] = oybx[y, x]*px[x]
                else:
                    nd.potential[x, y, xy] = eybx[y, x] - oybx[y, x]*px[x]
        else:
            for x, y in itertools.product([0, 1], repeat=2):
                # P(x, y)
                xy = y + 2 * x
                self.nd_XY.potential[x, y, xy] = oybx[y, x]*px[x]

    def refresh_random_nodes(self):
        """
        This method randomizes yet again all random pots.

        Returns
        -------
        None

        """
        for nd in self.random_nodes:
            nd.potential.set_to_random()
            nd.potential.normalize_self()

    def refresh_evidence(self, evi_nd_to_state):
        """
        This method replaces self.evi_nd_to_state by the input evi_nd_to_state

        Parameters
        ----------
        evi_nd_to_state : dict[BayesNode, int]

        Returns
        -------
        None

        """
        self.evi_nd_to_state = evi_nd_to_state
        for nd, val in self.evi_nd_to_state:
            assert 0 <= val <= nd.size,\
                "state of evidence node " + nd.name + " is out of range."
            nd.active_states = [val]

if __name__ == "__main__":
    from examples_cbnets.HuaDar import *

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
            nd.potential.normalize_self

        in_bnet = BayesNet(nodes)
        in_bnet.draw(algo_num=1)

        random_nodes = [cl, sp]
        evi_node_to_state = {sp: 0}
        oe_data = [.3, .5, .7, .1, .9]
        imagined_bnet = ImaginedBayesNet(in_bnet,
                                         random_nodes,
                                         evi_node_to_state,
                                         oe_data,
                                         only_obs=False)
        imagined_bnet.draw(algo_num=1)
        # for nd in imagined_bnet.nodes:
        #     print(nd.name,
        #           [x.name for x in nd.parents],
        #           [x.name for x in nd.children])
        path1 = 'examples_cbnets/tempo.dot'
        imagined_bnet.write_dot(path1)
    main()







