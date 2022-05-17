import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('result-tangrams.tsv',sep='\t')

data = data[data['type'] == 'orig']

sns.set_theme(style="darkgrid")

sns.relplot(data=data, x="round", y="rank", hue="method", kind="line", ci=95)

plt.show()
