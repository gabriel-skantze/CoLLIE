import glob
import random
from collie.models import *

random.seed(0)

iterations = 50
rounds = 6
n_labels_teach = 50
n_labels_test = 50
n_labels_tot = 200

# Set this to your LAD image directory
dataDir = "~/data/LAD/images"

allNames = collie.read_text_file("unusualNames.txt")

out = open("result-lad.tsv", "w")
out.write("iter\tround\ttext\tmethod\tcategory\tlabel\ttraining\trank\n")

for iteration in range(iterations):
    print(f"Iteration {iteration}")

    labels = []
    categories = []
    imageFiles = []
    dirs = []
    for dir in glob.glob(dataDir + '/*'):
        category = os.path.basename(dir)[0:1]
        if category in ["E", "A", "F", "V"]:
            dirs.append(dir)
    if len(dirs) == 0:
        print("Error: No images found in " + dataDir)
        exit(0)
    for dir in random.sample(dirs, n_labels_tot):
        category = os.path.basename(dir)[0:1]
        name = os.path.basename(dir)[2:].replace("_", " ")
        labels.append(name)
        categories.append(category)
        for file in random.sample(glob.glob(dir + '/*.jpg'), rounds):
            imageFiles.append(file)

    names = random.sample(allNames, n_labels_teach)

    models = [BaselineModel(),
              CollieModelWOScaling(),
              CollieFullModel(),
              FewShotLearner()]

    for round in range(rounds):

        print(f"  Round {round}")

        imageFilesRound = []
        for i in range(n_labels_tot):
            imageFilesRound.append(imageFiles[i * rounds + round])
        imgEmbedsRound = collie.encode_images(imageFilesRound)

        for i in range(n_labels_teach):

            # Now we evaluate performance on the teached objects using new names
            phrase = names[i]
            for model in models:
                mrr = collie.mrr(model.find_best(phrase, imgEmbedsRound), i)
                if mrr is not None:
                    out.write(f"{iteration}\t{round}\t{phrase}\t{model.name}\t{categories[i]}\t{labels[i]}\tcl\t{mrr}\n")

            # Now we evaluate performance on the teached objects using original names
            phrase = labels[i]
            for model in models:
                mrr = collie.mrr(model.find_best(phrase, imgEmbedsRound), i)
                if mrr is not None:
                    out.write(f"{iteration}\t{round}\t{phrase}\t{model.name}\t{categories[i]}\t{labels[i]}\tzs-same\t{mrr}\n")

        # Now we evaluate performance on not-teached objects using original names
        for i in range(n_labels_teach, n_labels_teach + n_labels_test):
            phrase = labels[i]
            for model in models:
                mrr = collie.mrr(model.find_best(phrase, imgEmbedsRound), i)
                if mrr is not None:
                    out.write(f"{iteration}\t{round}\t{phrase}\t{model.name}\t{categories[i]}\t{labels[i]}\tzs-other\t{mrr}\n")

        # Now we teach it the new names
        for model in models:
            for i in range(n_labels_teach):
                phrase = names[i]
                model.teach(phrase, imgEmbedsRound, i)
            model.train()

        del imgEmbedsRound

out.close()