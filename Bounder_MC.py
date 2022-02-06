from potentials.DiscreteUniPot import *
from potentials.DiscreteCondPot import *

import numpy as np
import pymc3 as pm
import itertools

import theano
import theano.tensor as tt

class Bounder_MC:
    """
    Note: This class uses code in directories: nodes, graphs, potentials,
    which was taken from my app Quantum Fog
    https://github.com/artiste-qb-net/quantum-fog

    This class also benifitted greatly from the following thread in the
    PyMC3 Discourse
    https://discourse.pymc.io/t/bayes-nets-belief-networks-and-pymc/5150

    We call this class Bounder_MC= Bounder Monte Carlo, to distinguish it
    from Bounder_ana= Bounder analytic.


    Attributes
    ----------
    imagined_bnet : ImaginedBayesNet
    num_1world_samples : int
    num_worlds : int
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
        self.trol_coords_to_PNS3_bds = None

    def refresh_pm_model(self, trol_coords):
        """

        Parameters
        ----------
        trol_coords : tuple[int]

        Returns
        -------
        None

        """
        if self.pm_model is not None:
            del self.pm_model
        self.pm_model = pm.Model()
        # nd_to_rv = node to random variable
        # this dictionary maps each BayesNode object in imagined_bnet to a
        # pymc3 random variable
        nd_to_rv = {}
        topo_index_to_nd = {}
        for nd in self.imagined_bnet.nodes:
            topo_index_to_nd[nd.topo_index] = nd
        topo_indices = list(topo_index_to_nd.keys()).sort()
        trol_list = self.imagined_bnet.trol_list
        with self.pm_model:
            for topo_index in topo_indices:
                nd = topo_index_to_nd[topo_index]
                nd_arr = None
                nd_pa_list = nd.potential.ord_nodes
                rv_pa_list = [nd_to_rv[nd1] for nd1 in nd_pa_list]
                if nd in trol_list:
                    for k, nd in enumerate(trol_list):
                        active_state = trol_coords[k]
                        num_parents = len(nd_pa_list)
                        nd_arr = np.zeros(shape=nd.potential.pot_arr.shape)
                        arr[active_state] = 1.0
                else:
                    nd_arr = nd.potential.pot_arr

                if len(rv_pa_list) == 0:
                    nd_to_rv[nd] = pm.Categorical(nd.name, nd_arr)
                else:
                    rv_pa_tuple = tuple(rv_pa_list)
                    nd_to_rv[nd] = pm.Categorical(
                        nd.name, lambda rv_pa_tuple:
                        theano.shared(nd_arr)[rv_pa_tuple])

    def estimate_PNS3_for_trol_coords(self, trol_coords):
        """

        Parameters
        ----------
        trol_coords : dict[BayesNode, int]

        Returns
        -------
        None

        """
        self.refresh_pm_model(trol_coords)
        trace = pm.sample_prior_predictive(self.num_1world_samples)

        PN = trace['Y0'][trace['X']==1 & trace['Y']==1].mean()[0]
        PS = trace['Y1'][trace['X']== 0 & trace['Y'] == 0].mean()[1]

        prob_Y0_is_0 =  trace['Y0'].mean()[0]
        prob_Y1_is_1_if_Y0_is_0 = trace['Y1'][trace['Y0']==0].mean()[1]
        PNS = prob_Y1_is_1_if_Y0_is_0*prob_Y0_is_0
        
        return PNS, PN, PS

    def estimate_PNS3_for_all_trol_cases(self):
        """

        Returns
        -------
        dict[tuple, np.array]
            np.array shape=(3, 2)

        """
        trol_nd_size_list = [nd.size for nd in self.ord_trol_nodes]
        trol_coords_to_PNS3 = {}
        for trol_coords in itertools.product(trol_nd_size_list):
            trol_nd_to_state = dict(zip(self.ord_trol_nodes, trol_coords))
            PNS, PN, PS = self.estimate_PNS3_for_trol_coords(trol_nd_to_state)
            trol_coords_to_PNS3[trol_coords] = np.array([PNS, PN, PS])

        return trol_coords_to_PNS3

    def get_PNS3_bds(self):
        """

        Returns
        -------
        dict[tuple, np.array]
            np.array shape=(3, 2)

        """
        return self.trol_coords_to_PNS3_bds

    def set_PNS3_bds(self):
        """

        Returns
        -------
        None

        """
        for world in range(self.num_worlds):
            self.imagined_bnet.refresh_random_nodes()
            trol_coords_to_PNS3 = self.estimate_PNS3_for_all_trol_cases()
            for trol_coords, PNS3 in trol_coords_to_PNS3.items():
                PNS3_bds = self.trol_coords_to_PNS3_bds[trol_coords]
                for k in range(3):
                    low, high = PNS3_bds[k]
                    if PNS3[k] < low:
                        PNS3_bds[k][0] = PNS3[k]
                    if PNS3[k] > high:
                        PNS3_bds[k][1] = PNS3[k]