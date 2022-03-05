from DotTool import *
from graphs.BayesNet import *
from DoX_BayesNet import *
from IdentifiabilityChecker import *
from Plotter_nz import *

"""
The purpose of this script is to provide utility methods that are used in 
the series of jupyter notebooks named "good_bad_trols_*.ipynb" 
"""

# all graph names (i.e., gnames). An example of a gname is "G3"
all_gnames = ["G" + str(z) for z in range(1, 12)] + ["G11u"] \
             + ["G" + str(z) for z in range(12, 19)]


def get_unobs_nd_names(gname):
    """
    For a graph named gname, this method returns a list of the names of its
    unobserved nodes (i.e., either [], or ["U"] or ["U1", "U2"])

    Parameters
    ----------
    gname : str

    Returns
    -------
    list[str]

    """
    if gname in ["G2", "G3", "G5", "G6", "G10", "G11u", "G15", "G16"]:
        li = ["U"]
    elif gname == "G7":
        li = ["U1", "U2"]
    else:
        li = []
    return li

# all graphs have a single control node named "Z"


def get_dot_file_path(gname):
    """
    For a graph named gname, this method returns the path to its dot file in
    the dot_lib directory.

    Parameters
    ----------
    gname : str

    Returns
    -------
    str

    """
    return "dot_lib/good_bad_trols_" + gname + ".dot"


def build_in_bnet(gname, jupyter=True):
    """
    For a graph named gname, this method builds a bnet called in_bnet that
    has the structure specified by gname's dot file. in_bnet is drawn using
    graphviz, and a description of it is printed. jupyter=True iff the graph
    will be drawn in a jupyter notebook. in_bnet is returned by the method.

    Parameters
    ----------
    gname : str
    jupyter : bool

    Returns
    -------
    BayesNet

    """
    path = get_dot_file_path(gname)
    DotTool.draw(path, jupyter)
    nx_graph = DotTool.nx_graph_from_dot_file(path)
    in_bnet = BayesNet.new_from_nx_graph(nx_graph)
    # in general
    # DiscreteCondPot(False, [y1, y2, y3, x]) refers to P(x| y1, y2, y3)
    for nd in in_bnet.nodes:
        if len(nd.parents) == 0:
            nd.potential = DiscreteUniPot(False, nd)  # P(a)
        else:
            nd.potential = DiscreteCondPot(False,
                        list(nd.parents) + [nd])   # P(b| a)
    for nd in in_bnet.nodes:
        nd.potential.set_to_random()
        nd.potential.normalize_self()
    print(in_bnet)
    return in_bnet
        

def build_doX_bnet(gname, in_bnet, jupyter=True):
    """
    For a graph named gname with a BayesNet object in_bnet, this method
    builds a doX bnet. The doX bnet is drawn using graphviz, and a
    description of it is printed. jupyter=True iff the graph will be drawn
    in a jupyter notebook. doX_bnet is returned by the method.

    Parameters
    ----------
    gname : str
    in_bnet : BayesNet
    jupyter : bool

    Returns
    -------
    DoX_bnet

    """
    trol_list = [in_bnet.get_node_named("Z")]
    unobs_nd_list = [in_bnet.get_node_named(name)
                     for name in get_unobs_nd_names(gname)]
    doX_bnet = DoX_BayesNet(in_bnet,
                            trol_list,
                            unobs_nd_list,
                            x_val=0)
    doX_bnet.gv_draw(jupyter)
    print(doX_bnet)
    return doX_bnet


def run(gname_list,
        num_1world_samples=1000,
        num_worlds=100,
        jupyter=True):
    """
    For the graphs in the gname list gname_list, this method runs PyMC3 for
    each graph, and plots the graph's query bounds and statistics. The query
    being considered is P(y|do(X)=x, z) with Y=1, X=0,1 and Z=0,1.

    Parameters
    ----------
    gname_list : list[str]
    num_1world_samples : int
    num_worlds : int
    jupyter : bool

    Returns
    -------
    None

    """
    for gname in gname_list:
        print("*******************  " + gname + "  *************************")
        in_bnet = build_in_bnet(gname, jupyter=jupyter)
        doX_bnet = build_doX_bnet(gname, in_bnet, jupyter=jupyter)
        checker = IdentifiabilityChecker(doX_bnet,
                                    num_1world_samples=num_1world_samples,
                                    num_worlds=num_worlds)
        for x_val in [0, 1]:
            doX_bnet.reset_x_val(x_val)
            checker.set_query_bds_and_stats()
            checker.print_query_bds_and_stats()
            Plotter_nz.plot_query_bds(doX_bnet.x_val,
                checker.get_query_bds(),
                zname_to_query_stats=checker.get_query_stats(),
                horizontal=True)


if __name__ == "__main__":
    import random
    random.seed(871)
    def main():
        print(all_gnames)
        run(all_gnames[0:2],
            num_1world_samples=10,
            num_worlds=3,
            jupyter=False)
    main()
