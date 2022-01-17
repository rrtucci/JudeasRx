import pandas as pd

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

col_name_to_col = {
    'o1b0': o1b0,
    'o1b1': o1b1,
    'px1': px1,
    'e1b0': e1b0,
    'e1b1': e1b1,
    'pz': pz
}

df = pd.DataFrame(col_name_to_col,
                  index=['a', 'b', 'c'])
# print("aaaaaaaaaa\n", df)
df.reset_index(inplace=True)
df = df.rename(columns={'index': 'zname'})
# print("bbbbbbbbb\n", df)

# saving the dataframe
df.to_csv('3z_example.csv', index=False)
