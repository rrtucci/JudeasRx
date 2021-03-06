import numpy as np
from collections import OrderedDict
from Bounder_ana import Bounder_ana
from Plotter_nz import Plotter_nz
from Reader import *
np.set_printoptions(precision=3, floatmode="fixed")


class MultiBounder_ana:
    """
    This class takes as input the probabilities for o1b0=O_{1|0,z}, o1b1=O_{
    1|1, z}, px1= P(x=1|z), e1b0=E_{1|0,z}, e1b1=E_{1|1,z}, pz=P(z) for each
    stratum z with name 'zname'. It can also take as input a utility
    function alp_y0_y1 = \alpha_{y_0, y_1} and various flags.

    If only_obs=True (i.e., only observational data, no experimental data),
    set e1b0 and e1b1 for each stratum to a dummy value. It will not be used.

    For each stratum, this class constructs a Bounder_ana object and asks it to
    calculate PNS3 and EU bounds. It stores all the bounders it constructs.
    Its method plot_bds() calls class Plotter_nz to plot the bounds stored in
    each bounder.

    This class has two constructors: __init__ takes its input probabilities
    from a dictionary named zname_to_input_probs, whereas create_from_file()
    takes them from a csv file.


    Attributes
    ----------
    alp_y0_y1 : np.array
        [shape=(2, 2)]
        utility function
    exogeneity : bool
    monotonicity : bool
    only_obs : bool
        True iff only observational probabilities, no experimental ones are
        available.
    strong_exo : bool
    zname_to_bounder : OrderedDict[str, Bounder_ana]
    zname_to_pz : OrderedDict[str, float]
    """

    def __init__(self,
                 zname_to_input_probs,
                 alp_y0_y1=None,
                 only_obs=False,
                 exogeneity=False,
                 strong_exo=False,
                 monotonicity=False):
        """

        Parameters
        ----------
        zname_to_input_probs : OrderedDict[str, list[float]]
            dictionary mapping stratum name zname to the stratum's list of
            probabilities [o1b0, o1b1, px1, e1b0, e1b1, pz]
        alp_y0_y1 : np.array
            [shape=(2, 2)]
            utility function
        only_obs : bool
        exogeneity : bool
        strong_exo : bool
        monotonicity : bool
        """
        # always init flags first, in case calculations depend on them
        self.only_obs = only_obs
        self.exogeneity = exogeneity
        self.strong_exo = strong_exo
        self.monotonicity = monotonicity

        self.alp_y0_y1 = np.zeros(shape=(2, 2))
        if alp_y0_y1 is not None:
            self.alp_y0_y1 = alp_y0_y1

        self.zname_to_pz = OrderedDict()
        self.zname_to_bounder = OrderedDict()
        for zname, input_probs in zname_to_input_probs.items():
            o1b0 = input_probs[0]
            o1b1 = input_probs[1]
            px1 = input_probs[2]
            e1b0 = input_probs[3]
            e1b1 = input_probs[4]
            pz = input_probs[5]

            self.zname_to_pz[zname] = pz
            
            o_y_bar_x = np.array([
                [1 - o1b0, 1 - o1b1],
                [o1b0, o1b1]])
            px = np.array([1 - px1, px1])
            # print("ddddddddddd", zname, '\n', o_y_bar_x, "\n", px)
            self.zname_to_bounder[zname] = Bounder_ana(o_y_bar_x, px)
            bder = self.zname_to_bounder[zname]
            bder.only_obs = self.only_obs
            bder.exogeneity = self.exogeneity
            bder.strong_exo = self.strong_exo
            bder.monotonicity = self.monotonicity

            bder.set_exp_probs_bds()  # this depends on bder.monotonicity flag
            if self.only_obs:
                e_y_bar_x = np.array([[.5, .5],
                                      [.5, .5]])
            else:
                e_y_bar_x = np.array([
                    [1 - e1b0, 1 - e1b1],
                    [e1b0, e1b1]])
            bder.set_exp_probs(e_y_bar_x)

            if not self.only_obs:
                print(zname + ":")
                bder.print_exp_probs_bds()
                bder.check_exp_prob_bds_satisfied()
                print("Checked that Exp. Probs. satisfy bounds" +
                      " imposed by Obs. Probs. for stratum " + zname)

            bder.set_PNS3_bds()
            if alp_y0_y1 is not None:
                bder.set_utility_fun(alp_y0_y1)
                bder.set_EU_bds()
        Bounder_ana.check_prob_vec(np.array(list(self.zname_to_pz.values())))
        # for zname, bder in self.zname_to_bounder.items():
        #     print(zname+':')
        #     bder.print_all_probs()
        #     bder.print_PNS3_bds()

    @staticmethod
    def create_from_file(path, **kwargs):
        """
        This static method is a constructor. Whereas the constructor
        __init__ gets its input probabilities from a dictionary,
        this constructor gets them from a file located at 'path'. The file
        is a csv file with a very special structure which is checked. The
        file's structure is described in the docstring for
        Reader.get_obs_exp_probs().

        Parameters
        ----------
        path : str
            path to file with input probabilities.
        kwargs :
            same keyword arguments as the __init__ constructor

        Returns
        -------
        MultiBounder_ana

        """
        return MultiBounder_ana(Reader.get_obs_exp_probs(path), **kwargs)

    def plot_bds(self, horizontal=True):
        """
        This method asks class Plotter_nz to plot all the bounds stored in
        each bounder of the dictionary zname_to_bounder.

        Parameters
        ----------
        horizontal : bool

        Returns
        -------
        None

        """
        zname_to_p3_bds = OrderedDict()
        zname_to_EU_bds = OrderedDict()
        for zname, bder in self.zname_to_bounder.items():
            zname_to_p3_bds[zname] = bder.get_PNS3_bds()
            zname_to_EU_bds[zname] = bder.get_EU_bds()
        # print(zname_to_p3_bds)
        Plotter_nz.plot_all_bds(zname_to_p3_bds,
                                zname_to_EU_bds,
                                horizontal=horizontal)

    def plot_both_ATE(self):
        """
        Plots output of get_both_ATE().

        Returns
        -------
        None

        """
        bdoorATE = self.get_bdoorATE()
        if self.only_obs:
            ATE = None
        else:
            ATE = self.get_ATE()
        Plotter_nz.plot_both_ATE(bdoorATE, ATE=ATE)

    def get_ATE(self):
        """
        Returns a tuple consisting of (1) a dictionary mapping zname to
        ATE_z = E{1|1,z} - E_{1,0,z} and (2) the mean ATE_z
        defined as ATE=\sum_z P(z) ATE_z

        Returns
        -------
        OrderedDict[str, float], float

        """
        exp = 0
        zname_to_ATE = OrderedDict()
        for zname, bder in self.zname_to_bounder.items():
            pz = self.zname_to_pz[zname]
            ate = bder.get_ATE()
            exp += ate*pz
            zname_to_ATE[zname] = ate
        return zname_to_ATE, exp

    def get_bdoorATE(self):
        """
        Returns a tuple consisting of (1) a dictionary mapping zname to
        backdoor ATE_z defined as bdoorATE_z = O{1|1,z} - O_{1,0,z} and (2)
        the mean bdoorATE_z defined as bdoorATE=\sum_z P(z) bdoorATE_z

        Returns
        -------
        OrderedDict[str, float], float

        """
        exp = 0
        zname_to_bATE = OrderedDict()
        for zname, bder in self.zname_to_bounder.items():
            pz = self.zname_to_pz[zname]
            ate = bder.get_bdoorATE()
            exp += ate * pz
            zname_to_bATE[zname] = ate
        return zname_to_bATE, exp

    def print_ATE(self):
        """
        prints output of get_ATE()

        Returns
        -------
        None

        """
        if self.only_obs:
            return
        ate_dict, exp = self.get_ATE()
        print("z name:  probability of z, ATE_z")
        for zname, ate in ate_dict.items():
            print(zname, ":", '%.3f, %.3f' % (self.zname_to_pz[zname], ate))
        print("mean ATE_z=", exp)

    def print_bdoorATE(self):
        """
        prints output of get_bdoorATE()

        Returns
        -------
        None

        """
        bATE_dict, exp = self.get_bdoorATE()
        print("z name:  probability of z, bdoorATE_z")
        for zname, ate in bATE_dict.items():
            print(zname + ":", '%.3f, %.3f'%(self.zname_to_pz[zname], ate))
        print("mean bdoorATE_z=", exp)

    def print_both_ATE(self):
        """
        Prints output of get_ATE() and get_bdoorATE() side by side for easy
        comparison.

        Returns
        -------
        None

        """
        if self.only_obs:
            self.print_bdoorATE()
            return

        ate_dict_o, mean_o = self.get_bdoorATE()
        ate_dict_e, mean_e = self.get_ATE()
        print("z name:  probability of z, ATE_z, bdoorATE_z")
        for zname, ez in ate_dict_e.items():
            oz = ate_dict_o[zname]
            print(zname + ":", '%.3f, %.3f, %.3f'
                  %(self.zname_to_pz[zname], ez, oz))
        print("mean ATE_z=", mean_e)
        print("mean bdoorATE_z=", mean_o)

if __name__ == "__main__":

    def main(only_obs):
        # o1b0 = input_probs[0]
        # o1b1 = input_probs[1]
        # px1 = input_probs[2]
        # e1b0 = input_probs[3]
        # e1b1 = input_probs[4]
        # pz = input_probs[5]
        zname_to_input_probs = OrderedDict()
        zname_to_input_probs['a'] = [.5, .33, .62, .5, .5, .2]
        zname_to_input_probs['b'] = [.37, .62, .71, .5, .5, .3]
        zname_to_input_probs['c'] = [.2, .5, .7, .1, .6, .5]
        alp_y0_y1 = np.array([[.5, -.4], [.2, .1]])
        mba = MultiBounder_ana(zname_to_input_probs,
                       alp_y0_y1=alp_y0_y1,
                       only_obs=only_obs)
        print("------------------------------------")
        mba.print_both_ATE()
        mba.plot_bds()
        mba.plot_both_ATE()
        print("************************")
    main(False)
    main(True)





