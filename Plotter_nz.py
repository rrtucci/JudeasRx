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
    def plot_p3_bds(ax, zname_to_p3_bds, horizontal=False):
        """
        This method plots 3 error bars for the bounds of PNS3 = (PNS,
        PN, PS) for each stratum with name 'zname'.

        Parameters
        ----------
        ax : Axes
            an axis from matplotlib
        zname_to_p3_bds :  OrderedDict(str, np.array[shape=(3, 2)])
            ordered dictionary mapping stratum named zname its pns3 bounds.
        horizontal : bool

        Returns
        -------
        None

        """
        nz = len(zname_to_p3_bds)
        bar_width = .7/nz
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
                for k in range(3):  # ax.text not vectorized
                    ax.text(bar_centers[k]-bar_width/5, p3_bds[k, 1]+.02,
                            texts[k],
                            size=Plotter_nz.text_size(bar_width),
                            rotation=90)
            else:
                ax.barh(bar_centers, p3_bds[:, 1] - p3_bds[:, 0],
                       height=bar_width, left=p3_bds[:, 0])
                for k in range(3):  # ax.text not vectorized
                    ax.text(p3_bds[k, 1], bar_centers[k] - bar_width / 4,
                            texts[k],
                            size=Plotter_nz.text_size(bar_width,
                                                      horizontal=True),
                            rotation=0)

    @staticmethod
    def plot_eu_bds(ax, zname_to_eu_bds, horizontal=False):
        """
        This method plots an error bar for the bounds of EU for each stratum
        with name 'zname'.

        Parameters
        ----------
        ax : Axes
            an axis from matplotlib
        zname_to_eu_bds :  OrderedDict(str, np.array[shape=(2, )])
            ordered dictionary mapping stratum named zname to its EU bounds.
        horizontal : bool

        Returns
        -------
        None

        """
        nz = len(zname_to_eu_bds)
        bar_width = .7/nz
        plt.sca(ax)
        if not horizontal:
            plt.xticks([0], ["EU"])
            ax.set_ylim(-1, 1)
            y_labels = np.arange(-1, 1.1, .2)
            ax.set_yticks(y_labels)
            ax.grid(linestyle='--', axis='y')
            ax.set_ylabel('utility')
        else:
            plt.yticks([0], ["EU"])
            ax.set_xlim(-1, 1)
            x_labels = np.arange(-1, 1.1, .2)
            ax.set_xticks(x_labels)
            ax.grid(linestyle='--', axis='x')
            ax.set_xlabel('utility')
        zindex = 0
        for zname, eu_bds in zname_to_eu_bds.items():
            zindex += 1
            bar_center = (2*zindex - nz-1)*bar_width/2
            texto = '(%.2f, %.2f) %s' % (eu_bds[0], eu_bds[1], zname)
            if not horizontal:
                ax.bar(bar_center, eu_bds[1]-eu_bds[0],
                        width=bar_width, bottom=eu_bds[0])
                ax.text(bar_center-bar_width/5, eu_bds[1]+.02, texto,
                        size=Plotter_nz.text_size(bar_width),
                        rotation=90)
            else:
                ax.barh(bar_center, eu_bds[1]-eu_bds[0],
                        height=bar_width, left=eu_bds[0])
                ax.text(eu_bds[1], bar_center - bar_width/4,  texto,
                        size=Plotter_nz.text_size(bar_width,
                                                  horizontal=True),
                        rotation=0)
        if not horizontal:
            ax.axhline(y=0, color='black')

    @staticmethod
    def plot_all_bds(zname_to_p3_bds, zname_to_eu_bds, horizontal=False):
        """
        This method calls both plot_p3_bds() and plot_eu_bds().

        Parameters
        ----------
        zname_to_p3_bds : OrderedDict(str, np.array[shape=(3, 2)])
        zname_to_eu_bds : OrderedDict(str, np.array[shape=(2, )])
        horizontal : bool

        Returns
        -------
        None
        """
        if not horizontal:
            plt.figure(figsize=(12, 5))
            gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
        else:
            plt.figure(figsize=(5, 5))
            gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        Plotter_nz.plot_p3_bds(ax1, zname_to_p3_bds, horizontal=horizontal)
        Plotter_nz.plot_eu_bds(ax2, zname_to_eu_bds, horizontal=horizontal)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    def main1():
        zname_to_p3_bds = OrderedDict()
        zname_to_p3_bds['m'] = np.array([[.3, .3],
                                        [.2, .45],
                                        [.5, .68]])
        zname_to_p3_bds['f'] = np.array([[.35, .45],
                                        [.25, .55],
                                        [.55, .72]])
        zname_to_eu_bds = OrderedDict()
        zname_to_eu_bds['m'] = np.array([-.4, .8])
        zname_to_eu_bds['f'] = np.array([0, .5])

        Plotter_nz.plot_all_bds(zname_to_p3_bds, zname_to_eu_bds)

    import random

    def main2(nz, horizontal=False):
        zname_to_p3_bds = OrderedDict()
        zname_to_eu_bds = OrderedDict()

        def pair():
            a = random.uniform(0, 1)
            b = random.uniform(0, 1)
            if a > b:
                a, b = b, a
            return [a, b]

        for zindex in range(nz):
            zname_to_p3_bds[str(zindex+1)] = np.array([pair(), pair(), pair()])
            zname_to_eu_bds[str(zindex+1)] = np.array(pair())
        Plotter_nz.plot_all_bds(zname_to_p3_bds, zname_to_eu_bds, horizontal)


    main2(10, horizontal=True)
