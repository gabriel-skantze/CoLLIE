import glob
import re
import random
from collie.models import *

random.seed(0)

iterations = 3000
rounds = 31

colDict = {"0":"red", "20":"yellow", "40":"green", "60":"blue", "80":"purple"}

synonyms = {"arrow":"pointer",
            "barn":"shed",
            "boat":"ship",
            "bridge":"overpass",
            "chevron":"rank",
            "chicken":"hen",
            "crown":"tiara",
            "giraffe":"camel",
            "goose":"duck",
            "head":"skull",
            "lozenge":"troche",
            "monitor":"screen",
            "mountain":"hill",
            "rock":"stone",
            "spikes":"spears",
            "temple":"church",
            "wedge":"chock"}

names = []
imageFiles = []
imageNames = []
imagesCols = []
dataDir = "tangrams"
for file in glob.glob(dataDir + '/*.png'):
    imageFiles.append(file)
    imageNames.append(re.sub(r'_.*', '', os.path.basename(file)))
    col = re.sub(r'.*?_(\d+).*', r'\1', os.path.basename(file))
    imagesCols.append(colDict[col])

nbest = len(imageFiles)
imgEmbeds = collie.encode_images(imageFiles)

phrases = []
synPhrases = []
for k in range(nbest):
    phrases.append("the " + imagesCols[k] + " " + imageNames[k])
    synPhrases.append("the " + imagesCols[k] + " " + synonyms[imageNames[k]])

out = open("result-tangrams.tsv", "w")
out.write("iter\ttype\tround\tname\ttext\tmethod\trank\n")

for iter in range(iterations):
    print(f"Iteration {iter}")

    baseline = BaselineModel(name="CLIP")
    models = [baseline,
              CollieFullModel(),
              CollieModelWOScaling(),
              FewShotLearner()]

    for round in range(rounds):
        k = random.randint(0, nbest-1)
        name = phrases[k].split(" ")[2]

        for model in models:
            mrr = collie.mrr(model.find_best(phrases[k], imgEmbeds), k)
            if mrr is None:
                mrr = collie.mrr(baseline.find_best(phrases[k], imgEmbeds), k)
            out.write(f"{iter}\torig\t{round}\t{name}\t{phrases[k]}\t{model.name}\t{mrr}\n")
            mrrsyn = collie.mrr(model.find_best(synPhrases[k], imgEmbeds), k)
            if mrrsyn is None:
                mrrsyn = collie.mrr(baseline.find_best(synPhrases[k], imgEmbeds), k)
            out.write(f"{iter}\tsyn\t{round}\t{name}\t{synPhrases[k]}\t{model.name}\t{mrrsyn}\n")

        for model in models:
            model.teach(phrases[k], imgEmbeds, k)
            model.train()

out.close()