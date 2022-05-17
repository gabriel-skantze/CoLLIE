import pandas as pd

data = pd.read_csv('result-lad.tsv',sep='\t')

print(pd.pivot_table(data =data, index=["training", "round","method"], columns=[])[["rank"]])