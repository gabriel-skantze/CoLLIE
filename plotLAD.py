import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('result-lad.tsv',sep='\t')
data = pd.DataFrame(data)

sns.set_theme(style="darkgrid")

sns.relplot(data=data, x="round", y="rank", hue="method", col="training", kind="line", ci=95)

plt.show()
