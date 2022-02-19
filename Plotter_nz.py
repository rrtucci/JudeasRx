import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import OrderedDict


class Plotter_nz:
    """
    This class has no constructor or attributes. It consists of just static
    methods that plot using matplotlib. This class generalizes Plotter_2z to
    more than 2 z's (i.e., 2 strata).

    """
    @staticmethod
    def text_size(bar_width, horizontal=False):
        """
        Returns the size of the font in the text attached to each bar.

        Parameters
        ----------
        bar_width : float
            the relative width of a bar. Separation between bar families is 1.
        horizontal : bool
            True iff horizontal bars.

        Returns
        -------
        float

        """
        if not horizontal:
            size = 150*bar_width
        else:
            size = 60*bar_width
        if size > 9:
            size = 9
        return size

    @staticmethod
    def plot_p3_bds(ax, zname_to_p3_bds, zname_to_p3_stats=None,
                    horizontal=False):
        """
        This method plots 3 min-max bars for the bounds of PNS3 = (PNS, PN,
        PS) for each stratum with name 'zname'. If zname_to_p3_stats is
        given, it plots error bars within those min-max bars. The error bars
        are one sigma (i.e., standard deviation) long on each side of mu (
        i.e., mean).

        Parameters
        ----------
        ax : Axes
            an axis from matplotlib
        zname_to_p3_bds :  OrderedDict(str, np.array[shape=(3, 2)])
            ordered dictionary mapping stratum named zname to its pns3 bounds.
        zname_to_p3_stats :  OrderedDict(str, np.array[shape=(3, 2)])
            ordered dictionary mapping stratum named zname to its pns3
            statistics. The array has the same shape as in zname_to_p3_bds,
            but it replaces the low bound by the mean mu, and the high bound
            by the standard deviation sigma.          
        horizontal : bool

        Returns
        -------
        None

        """
        nz = len(zname_to_p3_bds)
        bar_width = .7/nz
        tsize = Plotter_nz.text_size(bar_width, horizontal=horizontal)
        if not horizontal:
            x_labels = ("PNS", "PN", "PS")
            plt.sca(ax)
            plt.xticks(range(3), x_labels)
            ax.set_ylim(0, 1)
            y_labels = np.arange(0, 1.1, .1)
            ax.set_yticks(y_labels)
            ax.grid(linestyle='--', axis='y')
            ax.set_ylabel('probability')
        else:
            y_labels = ("PNS", "PN", "PS")
            plt.sca(ax)
            plt.yticks(range(3), y_labels)
            ax.set_xlim(0, 1)
            x_labels = np.arange(0, 1.1, .1)
            ax.set_xticks(x_labels)
            ax.grid(linestyle='--', axis='x')
            ax.set_xlabel('probability')
        
        zindex = 0
        # print('bar width', bar_width)
        for zname, p3_bds in zname_to_p3_bds.items():
            zindex += 1
            # print("cccccccccc", zindex, zname)
            bar_centers = np.arange(3) + (2*zindex - nz-1)*bar_width/2
            texts = ['(%.2f, %.2f) %s' % (p3_bds[k, 0],
                p3_bds[k, 1], zname) for k in range(3)]
            # print('bar centers', bar_centers)
            if not horizontal:
                ax.bar(bar_centers, p3_bds[:, 1]-p3_bds[:, 0],
                        width=bar_width, bottom=p3_bds[:, 0])
                if zname_to_p3_stats is not None:
                    p3_stats = zname_to_p3_stats[zname]
                    ax.errorbar(bar_centers,
                                p3_stats[:, 0],
                                yerr=p3_stats[:, 1],
                                fmt=".k",
                                capsize=3)
                for k in range(3):  # ax.text not vectorized
                    ax.text(bar_centers[k]-tsize/500, p3_bds[k, 1]+.02,
                            texts[k],
                            size=tsize,
                            rotation=90)
            else:
                ax.barh(bar_centers, p3_bds[:, 1] - p3_bds[:, 0],
                       height=bar_width, left=p3_bds[:, 0])
                for k in range(3):  # ax.text not vectorized
                    ax.text(p3_bds[k, 1], bar_centers[k]-tsize/500,
                            texts[k],
                            size=tsize,
                            rotation=0)
                    if zname_to_p3_stats is not None:
                        p3_stats = zname_to_p3_stats[zname]
                        ax.errorbar(p3_stats[:, 0],
                                    bar_centers,
                                    xerr=p3_stats[:, 1],
                                    fmt=".k",
                                    capsize=3)

    @staticmethod
    def plot_EU_bds(ax, zname_to_EU_bds, zname_to_EU_stats=None,
                    horizontal=False, positive=False):
        """
        This method plots a min-max bar for the bounds of EU for each
        stratum with name 'zname'. If zname_to_EU_stats is given, it plots
        error bars within those min-max bars. The error bars are one sigma (
        i.e., standard deviation) long on each side of mu (i.e., mean).

        Parameters
        ----------
        ax : Axes
            an axis from matplotlib
        zname_to_EU_bds :  OrderedDict(str, np.array[shape=(2, )])
            ordered dictionary mapping stratum named zname to its EU bounds.
        zname_to_EU_stats :  OrderedDict(str, np.array[shape=(2, )])
            ordered dictionary mapping stratum named zname to its EU
            statistics. The array has the same shape as in zname_to_EU_bds,
            but it replaces the low bound by the mean mu, and the high bound
            by the standard deviation sigma.
        horizontal : bool
        positive : bool
            True iff EU>=0

        Returns
        -------
        None

        """
        nz = len(zname_to_EU_bds)
        bar_width = .7/nz
        tsize = Plotter_nz.text_size(bar_width, horizontal=horizontal)
        plt.sca(ax)
        if not horizontal:
            if not positive:
                plt.xticks([0], ["EU"])
                ax.set_ylim(-1, 1)
                y_labels = np.arange(-1, 1.1, .2)
                ax.set_ylabel('utility')
            else:
                plt.xticks([0], ["query"])
                ax.set_ylim(0, 1)
                y_labels = np.arange(0, 1.1, .1)
                ax.set_ylabel('probability')
            ax.set_yticks(y_labels)
            ax.grid(linestyle='--', axis='y')
        else:
            if not positive:
                plt.yticks([0], ["EU"])
                ax.set_xlim(-1, 1)
                x_labels = np.arange(-1, 1.1, .2)
                ax.set_xlabel('utility')
            else:
                plt.yticks([0], ["query"])
                ax.set_xlim(0, 1)
                x_labels = np.arange(0, 1.1, .1)
                ax.set_xlabel('probability')
            ax.set_xticks(x_labels)
            ax.grid(linestyle='--', axis='x')

        zindex = 0
        for zname, eu_bds in zname_to_EU_bds.items():
            zindex += 1
            bar_center = (2*zindex - nz-1)*bar_width/2
            texto = '(%.2f, %.2f) %s' % (eu_bds[0], eu_bds[1], zname)
            if not horizontal:
                ax.bar(bar_center, eu_bds[1]-eu_bds[0],
                        width=bar_width, bottom=eu_bds[0])
                ax.text(bar_center-tsize/500, eu_bds[1]+.02, texto,
                        size=tsize,
                        rotation=90)
                if zname_to_EU_stats is not None:
                    eu_stats = zname_to_EU_stats[zname]
                    ax.errorbar(bar_center,
                                eu_stats[0],
                                yerr=eu_stats[1],
                                fmt=".k",
                                capsize=3)
            else:
                ax.barh(bar_center, eu_bds[1]-eu_bds[0],
                        height=bar_width, left=eu_bds[0])
                ax.text(eu_bds[1], bar_center-tsize/500,  texto,
                        size=tsize,
                        rotation=0)
                if zname_to_EU_stats is not None:
                    eu_stats = zname_to_EU_stats[zname]
                    ax.errorbar(eu_stats[0],
                                bar_center,
                                xerr=eu_stats[1],
                                fmt=".k",
                                capsize=3)
            if not positive:
                if not horizontal:
                    ax.axhline(y=0, color='black')
                else:
                    ax.axvline(x=0, color='black')

    @staticmethod
    def plot_all_bds(zname_to_p3_bds,
                     zname_to_EU_bds=None,
                     zname_to_p3_stats=None,
                     zname_to_EU_stats=None,
                     horizontal=False):
        """
        This method calls both plot_p3_bds() and plot_EU_bds().

        Parameters
        ----------
        zname_to_p3_bds : OrderedDict(str, np.array[shape=(3, 2)])
        zname_to_EU_bds : OrderedDict(str, np.array[shape=(2, )])
        zname_to_p3_stats : OrderedDict(str, np.array[shape=(3, 2)])
        zname_to_EU_stats : OrderedDict(str, np.array[shape=(2, )])
        horizontal : bool

        Returns
        -------
        None
        """
        nz = len(zname_to_p3_bds)
        if not horizontal:
            plt.figure(figsize=(12, 5))
            gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
        else:
            plt.figure(figsize=(5, 5))
            gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        Plotter_nz.plot_p3_bds(ax1, zname_to_p3_bds,
                               zname_to_p3_stats=zname_to_p3_stats,
                               horizontal=horizontal)
        Plotter_nz.plot_EU_bds(ax2, zname_to_EU_bds,
                               zname_to_EU_stats=zname_to_EU_stats,
                               horizontal=horizontal)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_both_ATE(bdoorATE, ATE=None):
        """
        This method is called by class Cprob_. It draws a scatter plot of
        (bdoorATE_z, z) for all z and (ATE_z, z) for all z. It also plots
        vertical lies at the mean bdoorATE_z and mean ATE_z.

        Parameters
        ----------
        bdoorATE : (OrderedDict[str, float], float)
        ATE : (OrderedDict[str, float], float), None

        Returns
        -------
        None

        """
        ATE_o, exp_o = list(bdoorATE[0].values()), bdoorATE[1]
        znames = list(ATE[0].keys())

        fig, ax = plt.subplots(1, 1)
        ax.scatter(ATE_o, znames, color='blue')
        ax.axvline(x=exp_o, color='blue')

        if ATE is not None:
            ATE_e, exp_e = list(ATE[0].values()), ATE[1]
            ax.scatter(ATE_e, znames, color='hotpink')
            ax.axvline(x=exp_e, color='hotpink')
            ax.legend(["mean ATE_z", "mean Backdoor ATE_z"])
        else:
            ax.legend(["mean Backdoor ATE_z"])

        ax.set_xlim(-1, 1)
        ax.grid(linestyle='--', axis='y')
        ax.set_ylabel('z name')
        ax.set_xlabel('difference of two probabilities')

        plt.show()

    @staticmethod
    def plot_query_bds(zname_to_query_bds, zname_to_query_stats=None,
                    horizontal=False):
        """
        This method plots a min-max bar for the bounds of a query for each
        stratum with name 'zname'. If zname_to_query_stats is given,
        it plots error bars within those min-max bars. The error bars are
        one sigma ( i.e., standard deviation) long on each side of mu (i.e.,
        mean). We define a query to be a probability, so its value is between
        0 and 1.

        Parameters
        ----------
        zname_to_query_bds :  OrderedDict(str, np.array[shape=(2, )])
            ordered dictionary mapping stratum named zname to its query bounds.
        zname_to_query_stats :  OrderedDict(str, np.array[shape=(2, )])
            ordered dictionary mapping stratum named zname to its query
            statistics. The array has the same shape as in zname_to_query_bds,
            but it replaces the low bound by the mean mu, and the high bound
            by the standard deviation sigma.
        horizontal : bool

        Returns
        -------
        None

        """
        ax = plt.subplot()
        Plotter_nz.plot_EU_bds(ax, zname_to_query_bds,
                               zname_to_EU_stats=zname_to_query_stats,
                               horizontal=horizontal,
                               positive=True)
        plt.show()


if __name__ == "__main__":

    def main1(horizontal):
        zname_to_p3_bds = OrderedDict()
        zname_to_p3_stats = OrderedDict()
        zname_to_p3_bds['m'] = np.array([[.3, .3],
                                        [.2, .45],
                                        [.5, .68]])
        zname_to_p3_stats['m'] = np.array([[.3, 0],
                                        [.3, .05],
                                        [.55, .02]])

        zname_to_p3_bds['f'] = np.array([[.35, .45],
                                        [.25, .55],
                                        [.55, .72]])
        zname_to_p3_stats['f'] = np.array([[.37, .01],
                                        [.40, .1],
                                        [.65, .03]])
        zname_to_EU_bds = OrderedDict()
        zname_to_EU_stats = OrderedDict()
        zname_to_EU_bds['m'] = np.array([-.4, .8])
        zname_to_EU_stats['m'] = np.array([.1, .2])

        zname_to_EU_bds['f'] = np.array([0, .5])
        zname_to_EU_stats['f'] = np.array([.3, .1])

        Plotter_nz.plot_all_bds(zname_to_p3_bds,
                                zname_to_EU_bds=zname_to_EU_bds,
                                zname_to_p3_stats=zname_to_p3_stats,
                                zname_to_EU_stats=zname_to_EU_stats,
                                horizontal=horizontal)

    import random

    def main2(nz, horizontal=False):
        zname_to_p3_bds = OrderedDict()
        zname_to_EU_bds = OrderedDict()

        def pair():
            a = random.uniform(0, 1)
            b = random.uniform(0, 1)
            if a > b:
                a, b = b, a
            return [a, b]

        for zindex in range(nz):
            zname_to_p3_bds[str(zindex+1)] = np.array([pair(), pair(), pair()])
            zname_to_EU_bds[str(zindex+1)] = np.array(pair())
        Plotter_nz.plot_all_bds(zname_to_p3_bds,
                                zname_to_EU_bds=zname_to_EU_bds,
                                horizontal=horizontal)

    def main3():
        ATE_dict = OrderedDict()
        bdoorATE_dict = OrderedDict()
        ATE_dict['a'], bdoorATE_dict['a'] = .4, .3
        ATE_dict['b'], bdoorATE_dict['b'] = .2, .1
        ATE_dict['c'], bdoorATE_dict['c'] = -.1, 0
        ATE = (ATE_dict, .3)
        bdoorATE = (bdoorATE_dict, -.3)

        Plotter_nz.plot_both_ATE(ATE, bdoorATE)

    def main4(horizontal):
        zname_to_query_bds = OrderedDict()
        zname_to_query_stats = OrderedDict()
        zname_to_query_bds['m'] = np.array([0, .8])
        zname_to_query_stats['m'] = np.array([.3, .2])

        zname_to_query_bds['f'] = np.array([0, .5])
        zname_to_query_stats['f'] = np.array([.3, .1])

        Plotter_nz.plot_query_bds(zname_to_query_bds,
                                zname_to_query_stats=zname_to_query_stats,
                                horizontal=horizontal)

    main1(horizontal=True)
    main2(10, horizontal=True)
    main3()
    main4(horizontal=False)

