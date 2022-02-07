from potentials.DiscreteUniPot import *
from potentials.DiscreteCondPot import *
from ImaginedBayesNet import *

import numpy as np
import pymc3 as pm
import itertools
import theano
import theano.tensor as tt


# pymc3 refused to be imported until I used an older version
# of arviz than the latest. I had to use arviz=0.6.1


class Bounder_MC:
    """
    Note: This class uses code in directories: nodes, graphs, potentials,
    which was taken from my app Quantum Fog
    https://github.com/artiste-qb-net/quantum-fog

    This class also benifitted from the following thread in the
    PyMC3 Discourse
    https://discourse.pymc.io/t/bayes-nets-belief-networks-and-pymc/5150

    We call this class Bounder_MC= Bounder Monte Carlo, to distinguish it
    from Bounder_ana= Bounder analytic.

    Theano is complementary to our Quantum Fog classes. They both use DAGs
    and potentials, but QFog uses Bayesian networks whereas Theano uses
    block diagrams. (rv=random variable, TPM=Transition Probability matrix)

    Bayesian network: node=rv=TPM=potential, arrows=indices of node

    block diagram: node=theano function or shared array=potential,
    arrows=rvs=indices of node



    Attributes
    ----------
    imagined_bnet : ImaginedBayesNet
    num_1world_samples : int
    num_worlds : int
    pm_model : pm.Model
    topo_index_to_nd : dict[int, BayesNode]
        dictionary mapping topological index to node
    trol_coords_to_PNS3_bds : dict[list[int], np.array]
        np.array shape=(3, 2)
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
            self.trol_coords_to_PNS3_bds[trol_coords] = np.array([[0, 1.],
                                                                  [0, 1.],
                                                                  [0, 1.]])

        self.topo_index_to_nd = {}
        for nd in self.imagined_bnet.nodes:
            self.topo_index_to_nd[nd.topo_index] = nd

        num_nds = len(self.imagined_bnet.nodes)
        for ind in range(num_nds):
            nd = self.topo_index_to_nd[ind]
            print("mmmkk", ind, nd.name,
                  "parents=", [x.name for x in nd.parents],
                  "children=", [x.name for x in nd.children], '\n',
                  nd.potential.pot_arr)
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
            control node in the order imagined_bnet.trol_list.

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
                # fam = family = nd and its parents
                nd_fam_list = nd.potential.ord_nodes
                print("llhhhf", topo_index, nd.name, [x.name for x in
                                                      nd_fam_list])
                nd_arr = None
                if nd in self.trol_list:
                    active_state = trol_coords[self.trol_to_pos[nd]]
                    fam_size = len(nd_fam_list)
                    nd_arr = np.zeros(shape=nd.potential.pot_arr.shape)
                    # slicex = tuple([slice(None), slice(None), ...,
                    # slice(None), active_state])
                    slicex = tuple([slice(None)] * (fam_size - 1) + \
                                   [active_state])
                    # print('aaasss', fam_size, active_state, slicex)
                    nd_arr[slicex] = 1.0
                else:
                    nd_arr = nd.potential.pot_arr
                print("vvvvv", nd.name, nd_arr)

                if len(nd_fam_list) == 1:
                    nd_to_rv[nd] = pm.Categorical(nd.name, nd_arr)
                    print("dddee", nd.name, nd_arr)
                else:
                    rv_fam_tuple = tuple(rv_fam_list)
                    lookup_table = theano.shared(np.asarray(nd_arr))
                    print("dddee", nd.name, lookup_table.get_value())

                    def fun(rv_fam_tuple):
                        return lookup_table[rv_fam_tuple]

                    nd_to_rv[nd] = pm.Categorical(
                        nd.name, fun(rv_fam_tuple))
                rv_fam_list = [nd_to_rv[nd1] for nd1 in nd_fam_list]

    def estimate_PNS3_for_trol_coords(self, trol_coords):
        """
        Estimates PNS3 = (PNS, PN, PS) for a single control coordinates.

        Parameters
        ----------
        trol_coords : tuple[int]

        Returns
        -------
        float, float, float
            PNS, PN, PS

        """
        self.refresh_pm_model(trol_coords)
        trace = pm.sample_prior_predictive(self.num_1world_samples)

        PN = trace['Y0'][trace['X'] == 1 & trace['Y'] == 1].mean()[0]
        PS = trace['Y1'][trace['X'] == 0 & trace['Y'] == 0].mean()[1]

        prob_Y0_is_0 = trace['Y0'].mean()[0]
        prob_Y1_is_1_if_Y0_is_0 = trace['Y1'][trace['Y0'] == 0].mean()[1]
        PNS = prob_Y1_is_1_if_Y0_is_0 * prob_Y0_is_0

        return PNS, PN, PS

    def estimate_PNS3_for_all_trol_coords(self):
        """
        Estimates the PNS3 bounds for each of all the possible control
        coordinates.

        Returns
        -------
        dict[tuple, np.array]
            np.array shape=(3, 2)

        """
        trol_range_list = [list(range(nd.size)) for nd in
                           self.imagined_bnet.trol_list]
        trol_coords_to_PNS3 = {}
        for trol_coords in itertools.product(*trol_range_list):
            PNS, PN, PS = self.estimate_PNS3_for_trol_coords(trol_coords)
            print("xxccd, trol_coords, PNS, PN, PS", trol_coords, PNS, PN, PS)
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
                PNS3_bds = self.trol_coords_to_PNS3_bds[trol_coords]
                for k in range(3):
                    low, high = PNS3_bds[k]
                    if PNS3[k] < low:
                        PNS3_bds[k][0] = PNS3[k]
                    if PNS3[k] > high:
                        PNS3_bds[k][1] = PNS3[k]


if __name__ == "__main__":
    def main():
        imagined_bnet = ImaginedBayesNet.build_test_imagined_bnet()
        # print("kkkll", imagined_bnet.nodes)
        bder = Bounder_MC(imagined_bnet,
                          num_1world_samples=10,
                          num_worlds=100)
        bder.set_PNS3_bds()
        print(bder.get_PNS3_bds())


    main()