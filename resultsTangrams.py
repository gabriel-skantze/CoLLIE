import pandas as pd

data = pd.read_csv('result-tangrams.tsv',sep='\t')

iterations = len(data[(data["type"] == "orig") & (data["method"] == "CoLLIE") & (data["round"] == 30)].index)
print(f"Number of iterations: {iterations}")

print("--- CLIP, ORIGINAL WORDS ---")

print(pd.pivot_table(data[(data["type"] == "orig") & (data["method"] == "CLIP")], index="name", columns=[])[["rank"]])

print("--- CLIP, SYNONYMS ---")

print(pd.pivot_table(data[(data["type"] == "syn") & (data["method"] == "CLIP")], index="name", columns=[])[["rank"]])

print("--- CoLLIE, ORIGINAL WORDS, round 30 ---")

print(pd.pivot_table(data[(data["type"] == "orig") & (data["method"] == "CoLLIE") & (data["round"] == 30)], index="name", columns=[])[["rank"]])

print("--- CoLLIE, SYNONYMS, round 30 ---")

print(pd.pivot_table(data[(data["type"] == "syn") & (data["method"] == "CoLLIE") & (data["round"] == 30)], index="name", columns=[])[["rank"]])

print("--- ORIGINAL WORDS, round 30 ---")

print(pd.pivot_table(data[(data["type"] == "orig") & (data["round"] == 30)], index="method", columns=[])[["rank"]])

print("--- SYNONYMS, round 30 ---")

print(pd.pivot_table(data[(data["type"] == "syn") & (data["round"] == 30)], index="method", columns=[])[["rank"]])