from Comparer import Comparer
import numpy as np
import pandas as pd

df = pd.read_csv('3z_example.csv')
print(df, '\n')

cer = Comparer.create_from_file(
    '3z_example.csv',
only_obs=True,
exogeneity=False,
strong_exo=False,
monotonicity=False)
cer.plot_bds()