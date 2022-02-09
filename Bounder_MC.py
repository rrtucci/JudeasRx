from ImaginedBayesNet import *

import numpy as np
import itertools
from pprint import pprint

import pymc3 as pm
import theano

np.set_printoptions(precision=3, floatmode="fixed")


# Installing pymc3 correctly proved to be a bit tricky.
# See
# https://github.com/pymc-devs/pymc/wiki/Installation-Guide-%28Linux%29
# I set up a conda virtual environment with Python 3.9. Then used:
# conda install -c conda-forge pymc3 theano-pymc mkl mkl-service
# conda install graphviz
# conda install pydotplus
# conda install python-graphviz
# conda install pydot

class Bounder_MC:
    """
    Note: This class uses code in directories: nodes, graphs, potentials,
    that was borrowed from my app Quantum Fog
    https://github.com/artiste-qb-net/quantum-fog

    This class also benifitted greatly from the following thread in the
    PyMC3 Discourse
    https://discourse.pymc.io/t/bayes-nets-belief-networks-and-pymc/5150

    We call this class Bounder_MC= Bounder Monte Carlo, to distinguish it
    from Bounder_ana= Bounder analytic.

    Theano is complementary to our Quantum Fog classes. They both use DAGs
    and potentials, but QFog uses Bayesian networks whereas Theano uses
    block diagrams. (rv=random variable, TPM=Transition Probability matrix)

    Here is how I translate a block diagram to a bnet
    block diagram -> Bayesian network
    rvs live in arrows, TPMs live in nodes-> both rvs and TPMs live in nodes
    arrow=rv -> arrow just a mapping thing, nothing lives there
    node= theano function or shared array -> TPM
    shared array node -> general TPM
    function node -> determinstic TPM

    This class has an imagined bnet (an object of ImaginedBayesNet) as
    input. The main goal of this class is to calculate bounds for the 3
    quantities PNS3=(PNS, PN, PS), for each strata z. Here a strata is one
    possible assignment of states to the control nodes of the imagined bnet.

    One particular assignment of TPMs to the random nodes of the imagined
    bnet is called a "world". This class loops over num_worlds number of
    worlds.

    For each world, this class loops over a pm model for each element of the
    cartesian product of all the states of each control node. We refer to
    the elements of that cartesian product as the trol coords ( i.e.,
    control coordinates). The control nodes that reprent the axes of a trol
    coord are stored in self.trol_list.

    Attributes
    ----------
    imagined_bnet : ImaginedBayesNet
    num_1world_samples : int
        number of samples taken by pm for a single world.
    num_worlds : int
        number of worlds. Each world has a different random assignmet to the
        TPM of the random nodes.          
    pm_model : pm.Model
        PyMC3 model
    topo_index_to_nd : dict[int, BayesNode]
        dictionary mapping topological index to node
    trol_coords_to_PNS3_bds : dict[list[int], np.array]
        np.array shape=(3, 2)  dictionary mapping control coordinates to
        PNS3 bounds
    trol_list : list[BayesNode]
        list of control nodes
    trol_to_pos : dict[BayesNode, int]
        dictionary mapping control node to its position in the trol_list
    """

    def __init__(self, imagined_bnet, num_1world_samples, num_worlds):
        """
        Constructor.

        Parameters
        ----------
        imagined_bnet : ImaginedBayesNet
        num_1world_samples : int
        num_worlds : int
        """
        self.imagined_bnet = imagined_bnet
        self.num_1world_samples = num_1world_samples
        self.num_worlds = num_worlds
        self.pm_model = None

        self.trol_coords_to_PNS3_bds = {}
        trol_range_list = [range(nd.size) for nd in
                           self.imagined_bnet.trol_list]
        # assume that initially, 0<= PNS <=1, 0 <= PN <= 1, 0 <= PS <= 1
        for trol_coords in itertools.product(*trol_range_list):
            self.trol_coords_to_PNS3_bds[trol_coords] = np.array([[1., 0],
                                                                  [1., 0],
                                                                  [1., 0]])

        self.topo_index_to_nd = {}
        for nd in self.imagined_bnet.nodes:
            self.topo_index_to_nd[nd.topo_index] = nd

        # num_nds = len(self.imagined_bnet.nodes)
        # for ind in range(num_nds):
        #     nd = self.topo_index_to_nd[ind]
        #     print("mmmkk", ind, nd.name,
        #           "parents=", [x.name for x in nd.parents],
        #           "children=", [x.name for x in nd.children], '\n',
        #           nd.potential.pot_arr)
        
        self.trol_list = self.imagined_bnet.trol_list
        self.trol_to_pos = {trol: pos for pos, trol in enumerate(
            self.trol_list)}

    def refresh_pm_model(self, trol_coords):
        """
        Refreshes the pm (PyMC3) model for the control coordinates
        trol_coords.

        Parameters
        ----------
        trol_coords : tuple[int]
            control coordinates is a tuple of giving the state of each
            control node in the order self.trol_list.

        Returns
        -------
        None

        """
        if self.pm_model is not None:
            del self.pm_model
        self.pm_model = pm.Model()

        with self.pm_model:

            # nd_to_rv = node to random variable
            # this dictionary maps each BayesNode object in imagined_bnet to a
            # pymc3 random variable
            nd_to_rv = {}
            num_nds = len(self.imagined_bnet.nodes)
            for topo_index in range(num_nds):
                nd = self.topo_index_to_nd[topo_index]
                # print("llhhhf", topo_index, nd.name)
                nd_pa_list = nd.potential.ord_nodes[:-1]
                num_parents = len(nd_pa_list)

                if nd in self.trol_list:
                    active_state = trol_coords[self.trol_to_pos[nd]]

                    nd_arr = np.zeros(shape=nd.potential.pot_arr.shape)
                    # slicex = tuple([slice(None), slice(None), ...,
                    # slice(None), active_state])
                    slicex = tuple([slice(None)] * num_parents +
                                   [active_state])
                    # print('aaasss', num_parents, active_state, slicex)
                    nd_arr[slicex] = 1.0
                else:
                    nd_arr = nd.potential.pot_arr
                # print("vvvvv", nd.name, nd_arr)

                if num_parents == 0:
                    nd_to_rv[nd] = pm.Categorical(nd.name, nd_arr)
                    # print("dxxxx", nd.name, nd_arr)
                else:
                    lookup_table = theano.shared(np.asarray(nd_arr))

                    def fun(*rv_pa_list):
                        return lookup_table[tuple(rv_pa_list)]
                    # print("lllkk", nd.name, fun(*([0]*num_parents)))

                    rv_pa_list = [nd_to_rv[nd1] for nd1 in nd_pa_list]
                    nd_to_rv[nd] = pm.Categorical(nd.name, fun(*rv_pa_list))

    def estimate_PNS3_for_these_trol_coords(self, trol_coords):
        """
        Estimates PNS3 = (PNS, PN, PS) for a single tuple of control
        coordinates.

        Parameters
        ----------
        trol_coords : tuple[int]

        Returns
        -------
        np.array
            shape=(3, ), [PNS, PN, PS]

        """
        self.refresh_pm_model(trol_coords)
        with self.pm_model:
            trace = pm.sample_prior_predictive(self.num_1world_samples)

            prob_Y0_is_1 =\
                trace['Y0'][(trace['X'] == 1) & (trace['Y'] == 1)].mean()
            PN = 1 - prob_Y0_is_1
            prob_Y1_is_1 = \
                trace['Y1'][(trace['X'] == 0) & (trace['Y'] == 0)].mean()
            PS = prob_Y1_is_1

            prob_Y0_is_0 = 1 - trace['Y0'].mean()
            prob_Y1_is_1_if_Y0_is_0 = trace['Y1'][(trace['Y0'] == 0)].mean()
            PNS = prob_Y1_is_1_if_Y0_is_0 * prob_Y0_is_0

            return np.array([PNS, PN, PS])

    def estimate_PNS3_for_all_trol_coords(self):
        """
        Estimates the PNS3 bounds for each of all the possible control
        coordinates.

        Returns
        -------
        dict[tuple, np.array]
            np.array of shape=(3, )

        """
        trol_range_list = [list(range(nd.size)) for nd in
                           self.imagined_bnet.trol_list]
        trol_coords_to_PNS3 = {}
        for trol_coords in itertools.product(*trol_range_list):
            PNS, PN, PS = self.estimate_PNS3_for_these_trol_coords(trol_coords)
            # print("xxccd, trol_coords, PNS, PN, PS", trol_coords, PNS, PN,
            # PS)
            trol_coords_to_PNS3[trol_coords] = np.array([PNS, PN, PS])

        return trol_coords_to_PNS3

    def get_PNS3_bds(self):
        """
        Gets class attribute trol_coords_to_PNS3_bds, where PNS3=(PNS, PN,
        PS).

        Returns
        -------
        dict[tuple, np.array]
            np.array shape=(3, 2)
            [[PNS low, PNS high],
            [PN low, PN high],
            [PS low, PS high]]

        """
        return self.trol_coords_to_PNS3_bds

    def set_PNS3_bds(self):
        """
        Sets class attribute trol_coords_to_PNS3_bds, where PNS3=(PNS, PN,
        PS).

        Returns
        -------
        None

        """
        for world in range(self.num_worlds):
            print("world", world)
            self.imagined_bnet.refresh_random_nodes()
            trol_coords_to_PNS3 = self.estimate_PNS3_for_all_trol_coords()
            for trol_coords, PNS3 in trol_coords_to_PNS3.items():
                PNS3_bounds = self.trol_coords_to_PNS3_bds[trol_coords]
                for k in range(3):
                    low, high = PNS3_bounds[k]
                    if PNS3[k] <= low:
                        PNS3_bounds[k][0] = PNS3[k]
                    if PNS3[k] > high:
                        PNS3_bounds[k][1] = PNS3[k]
                    # print("kkhhh", f"k={k}, low={low}, high={high}, "
                    #                f"PNS[k]={PNS3[k]}, "
                    #                f"new low={PNS3_bounds[k][0]},"
                    #                f"new high={PNS3_bounds[k][1]}")


if __name__ == "__main__":
    def main1():
        imagined_bnet = ImaginedBayesNet.build_test_imagined_bnet()
        # print("kkkll", imagined_bnet.nodes)
        bder = Bounder_MC(imagined_bnet,
                          num_1world_samples=1000,
                          num_worlds=2)
        bder.set_PNS3_bds()
        print("control nodes:", [nd.name for nd in bder.trol_list])
        pprint(bder.get_PNS3_bds())

    def main2():
        lookup_table = theano.shared(np.asarray([
            [[.99, .01], [.1, .9]],
            [[.9, .1], [.1, .9]]]))
        print(lookup_table.get_value()[0, 0])

        def fun1(*pa_list):
            return lookup_table[tuple(pa_list)]
        print("fun1", fun1(1, 0))

        def fun2(p1, p2):
            return lookup_table[p1, p2]
        print("fun2", fun2(1, 0))

    main1()
