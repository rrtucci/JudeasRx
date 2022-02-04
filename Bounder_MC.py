from potentials.DiscreteUniPot import *
from potentials.DiscreteCondPot import *

import numpy as np
import pymc3 as pm
import itertools

import theano
import theano.tensor as tt

class Bounder_MC:
    """
    Bounder_MC= Bounder Monte Carlo, to distinguish it from Bounder_ana=
    Bounder analytic.
    https://discourse.pymc.io/t/bayes-nets-belief-networks-and-pymc/5150

    Attributes
    ----------
    imagined_bnet : ImaginedBayesNet
    num_1world_samples : int
    num_worlds : int
    ord_evi_nodes : list[BayesNode]
    pm_model : pm.Model
    """

    def __init__(self, imagined_bnet, num_1world_samples, num_worlds):
        """

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
        self.ord_evi_nodes = list(imagined_bnet.evi_nd_to_state.keys())
        self.evi_coords_to_PNS3_bds = None

    def refresh_pm_model(self, evi_nd_to_state):
        """

        Parameters
        ----------
        evi_nd_to_state : dict[BayesNode, int]

        Returns
        -------
        None

        """

        if self.pm_model is not None:
            del self.pm_model
        self.pm_model = pm.Model()
        # nd_to_rv = node to random variable
        # this dictionary maps each BayesNode object to a
        # pymc3 random variable
        nd_to_rv = {}
        topo_index_to_nd = {}
        for nd in self.imagined_bnet:
            topo_index_to_nd[nd.topo_index] = nd
        topo_indices = list(topo_index_to_nd.keys()).sort()
        with self.pm_model:
            for topo_index in topo_indices:
                nd = topo_index_to_nd(topo_index)
                nd_pot = nd.potential
                nd_parents = nd_pot.ord_nodes
                rv_parents = tuple(nd_to_rv[pa_nd] for pa_nd in
                                   nd_parents)
                if nd in self.ord_evi_nodes:
                    arr = np.zeros(shape=(nd.size,))
                    arr[evi_nd_to_state[nd]] = 1.0
                    nd_to_rv[nd] = pm.Categorical(nd.name, arr)
                else:
                    if len(rv_parents) == 0:
                        nd_to_rv[nd] = pm.Categorical(nd.name, nd.pot_arr)
                    else:
                        nd_to_rv[nd] = pm.Categorical(
                            nd.name, lambda rv_parents:
                            theano.shared(nd.pot_arr)[rv_parents])

    def estimate_PNS3_for_one_evi_case(self, evi_nd_to_state):
        """

        Parameters
        ----------
        evi_nd_to_state : dict[BayesNode, int]

        Returns
        -------
        None

        """
        self.refresh_pm_model(evi_nd_to_state)
        trace = pm.sample_prior_predictive(self.num_1world_samples)

        PN = trace['Y0'][trace['X']==1 & trace['Y']==1].mean()[0]
        PS = trace['Y1'][trace['X']== 0 & trace['Y'] == 0].mean()[1]

        prob_Y0_is_0 =  trace['Y0'].mean()[0]
        prob_Y1_is_1_if_Y0_is_0 = trace['Y1'][trace['Y0']==0].mean()[1]
        PNS = prob_Y1_is_1_if_Y0_is_0*prob_Y0_is_0
        
        return PNS, PN, PS

    def estimate_PNS3_for_all_evi_cases(self):
        """

        Returns
        -------
        dict[tuple, np.array]
            np.array shape=(3, 2)

        """
        evi_nd_size_list = [nd.size for nd in self.ord_evi_nodes]
        evi_coords_to_PNS3 = {}
        for evi_coords in itertools.product(evi_nd_size_list):
            evi_nd_to_state = dict(zip(self.ord_evi_nodes, evi_coords))
            PNS, PN, PS = self.estimate_PNS3_for_one_evi_case(evi_nd_to_state)
            evi_coords_to_PNS3[evi_coords] = np.array([PNS, PN, PS])

        return evi_coords_to_PNS3

    def get_PNS3_bds(self):
        """

        Returns
        -------
        dict[tuple, np.array]
            np.array shape=(3, 2)

        """
        return self.evi_coords_to_PNS3_bds

    def set_PNS3_bds(self):
        """

        Returns
        -------
        None

        """
        for world in range(self.num_worlds):
            self.imagined_bnet.refresh_random_nodes()
            evi_coords_to_PNS3 = self.estimate_PNS3_for_all_evi_cases()
            for evi_coords, PNS3 in evi_coords_to_PNS3.items():
                PNS3_bds = self.evi_coords_to_PNS3_bds[evi_coords]
                for k in range(3):
                    low, high = PNS3_bds[k]
                    if PNS3[k] < low:
                        PNS3_bds[k][0] = PNS3[k]
                    if PNS3[k] > high:
                        PNS3_bds[k][1] = PNS3[k]