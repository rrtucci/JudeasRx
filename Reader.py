import pprint as pp
import pandas as pd
from collections import OrderedDict


class Reader:
    """
    This class has no constructor or attributes. It contains static methods
    for reading data from a file.

    """

    @staticmethod
    def get_obs_exp_probs(path):
        """
        This static method reads a file located at 'path'. The file is a csv
        file with a very special structure which is checked. The file must
        contain columns called "zname", "o1b0", "o1b1", "px1", "e1b0",
        "e1b1", and "pz". The order of the columns does not matter. The file
        may contain other columns, but they will be disregarded. The "zname"
        column contains the names of the strata. The other columns all
        contain probabilities.

        If you have no experimental data (i.e., only_obs=True), create dummy
        e1b0 and e1b1 columns anyway (their values will not be used). pz is
        used by MultiBounder_ana but not by MultiBounder_MC. The latter
        usecase generates its own pz values via Monte Carlo from the
        imagined bnet. You must create a dummy pz column for that usecase (
        its values will not be used).

        Parameters
        ----------
        path : str
            path to file with input probabilities.

        Returns
        -------
        OrderedDict[str, list[float]]

        """
        df = pd.read_csv(path)
        # print(df)
        cols = ['zname', 'o1b0', 'o1b1', 'px1', 'e1b0', 'e1b1', 'pz']
        assert all(x in df.columns for x in cols), \
            "file must have a column named all of the following," \
            " even if some columns won't be used:" \
            "['zname', 'o1b0', 'o1b1', 'px1', 'e1b0', 'e1b1', 'pz']"
        df = df[cols]
        znames = list(df['zname'])
        # print(df)
        zname_to_input_probs = OrderedDict()
        for k, zname in enumerate(znames):
            row = list(df.iloc[k])[1:]
            # print(k, zname, row)
            zname_to_input_probs[zname] = row
        return zname_to_input_probs


if __name__ == "__main__":

    def create_csv_file():

        # o1b0 = input_probs[0]
        # o1b1 = input_probs[1]
        # px1 = input_probs[2]
        # e1b0 = input_probs[3]
        # e1b1 = input_probs[4]
        # pz = input_probs[5]

        # zname_to_input_probs['a'] = [ .5, .33, .62, .5, .5, .2]
        # zname_to_input_probs['b'] = [.37, .62, .71, .5, .5, .3]
        # zname_to_input_probs['c'] = [ .2, .5 , .7 , .1, .6, .5]
        # alp_y0_y1 = np.array([[.5, -.4], [.2, .1]])

        o1b0 = [.5, .37, .2]
        o1b1 = [.33, .62, .5]
        px1 = [.62, .71, .7]
        e1b0 = [.5, .5, .1]
        e1b1 = [.5, .5, .6]
        pz = [.2, .3, .5]

        col_name_to_col_list = {
            'o1b0': o1b0,
            'o1b1': o1b1,
            'px1': px1,
            'e1b0': e1b0,
            'e1b1': e1b1,
            'pz': pz
        }

        df = pd.DataFrame(col_name_to_col_list,
                          index=['a', 'b', 'c'])
        # print("aaaaaaaaaa\n", df)
        df.reset_index(inplace=True)
        df = df.rename(columns={'index': 'zname'})
        # print("bbbbbbbbb\n", df)

        # saving the dataframe
        df.to_csv('3z_example.csv', index=False)

    def main():
        create_csv_file()
        zname_to_input_probs = Reader.get_obs_exp_probs('3z_example.csv')
        pp.pprint(zname_to_input_probs)

    main()
