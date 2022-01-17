import numpy as np
from collections import OrderedDict
import pandas as pd
from Bounder import Bounder
from Plotter_nz import Plotter_nz


class Comparer:
    """
    This class takes as input the probabilities for o1b0=O_{1|0,z}, o1b1=O_{
    1|1, z}, px1= P(x=1|z), e1b0=E_{1|0,z}, e1b1=E_{1|1,z}, pz=P(z) for each
    stratum z with name 'zname'. It can also take as input a utility
    function alp_y0_y1 = \alpha_{y_0, y_1} and various flags.

    For each stratum, this class constructs a Bounder object and asks it to
    calculate PNS3 and EU bounds. It stores all the bounders it constructs.
    Its method plot_bds() calls class Plotter_nz to plot the bounds stored in
    each bounder.

    This class has two constructors: __init__ takes its input probabilities
    from a dictionary named zname_to_input_probs, whereas create_from_file()
    takes them from a file.


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
    zname_to_bounder : OrderedDict[str, Bounder]
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
        self.only_obs = only_obs
        self.exogeneity = exogeneity
        self.strong_exo = strong_exo
        self.monotonicity = monotonicity

        self.alp_y0_y1 = np.zeros(shape=(2, 2))
        if alp_y0_y1 is not None:
            assert alp_y0_y1.shape == (2, 2)
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
            self.zname_to_bounder[zname] = Bounder(o_y_bar_x, px)
            bder = self.zname_to_bounder[zname]
            bder.exogeneity = self.exogeneity
            bder.strong_exo = self.strong_exo
            bder.monotonicity = self.monotonicity

            bder.set_exp_probs_bds()  # this depends on bder.monotonicity flag
            print(zname + ":")
            bder.print_exp_probs_bds()

            if self.only_obs:
                e_y_bar_x = np.array([[.5, .5],
                                      [.5, .5]])
            else:
                e_y_bar_x = np.array([
                    [1 - e1b0, 1 - e1b1],
                    [e1b0, e1b1]])
            bder.set_exp_probs(e_y_bar_x)
            bder.check_exp_prob_bds_satisfied()
            bder.set_pns3_bds()
            if alp_y0_y1 is not None:
                bder.set_utility_fun(alp_y0_y1)
                bder.set_eu_bds()
        Bounder.check_prob_vec(np.array(list(self.zname_to_pz.values())))
        # for zname, bder in self.zname_to_bounder.items():
        #     print(zname+':')
        #     bder.print_all_probs()
        #     bder.print_pns3_bds()

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
        zname_to_eu_bds = OrderedDict()
        for zname, bder in self.zname_to_bounder.items():
            zname_to_p3_bds[zname] = bder.get_pns3_bds()
            zname_to_eu_bds[zname] = bder.get_eu_bds()
        # print(zname_to_p3_bds)
        Plotter_nz.plot_all_bds(zname_to_p3_bds,
                                zname_to_eu_bds,
                                horizontal)

    @staticmethod
    def create_from_file(path, **kwargs):
        """
        This static method is a constructor. Whereas the constructor
        __init__ gets its input probbilities from a dictionary,
        this constructor gets them from a file located at 'path'.

        Parameters
        ----------
        path : str
            path to file with input probabilities. The file is a csv file
            with very special structure which is checked. The file must
            contain columns called "zname", "o1b0", "o1b1", "px1", "e1b0",
            "e1b1", and "pz". The file may contain other columns, but they
            will be disregarded. The "zname" column contains the names of
            the strata. The other columns all contain probabilities.
        kwargs :
            same keyword arguments as the __init__ constructor

        Returns
        -------
        Comparer

        """
        df = pd.read_csv(path)
        # print(df)
        cols = ['zname', 'o1b0', 'o1b1', 'px1', 'e1b0', 'e1b1', 'pz']
        assert all(x in df.columns for x in cols)
        df = df[cols]
        znames = list(df['zname'])
        # print(df)
        zname_to_input_probs = OrderedDict()
        for k, zname in enumerate(znames):
            row = list(df.iloc[k])[1:]
            # print(k, zname, row)
            zname_to_input_probs[zname] = row
        return Comparer(zname_to_input_probs, **kwargs)


if __name__ == "__main__":

    def main():
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
        cer = Comparer(zname_to_input_probs,
                       alp_y0_y1=alp_y0_y1,
                       only_obs=False)
        cer.plot_bds()
    main()





