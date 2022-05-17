# CoLLIE: Continual Learning of Language Grounding from Language-Image Embeddings

These are instructions on how to run the experiments in the article. 
The code is written in Python. 
The code should be possible to run on both cpu and gpu.

## Installing packages

We suggest you use Conda package manager and install everything within a Conda environment. 
On a CUDA GPU machine, the following will install everything you should need:

```
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm sklearn matplotlib pandas seaborn
$ pip install git+https://github.com/openai/CLIP.git
```

Replace `cudatoolkit=11.0` above with the appropriate CUDA version on your machine or `cpuonly` when installing on a machine without a GPU.

## Downloading the LAD dataset

1. Go to <https://paperswithcode.com/dataset/lad> and follow the links to download the images from the LAD dataset. 
2. Unzip the file to a directory of your choice.
3. Open `runLAD.py` and set `dataDir` to that directory. 

## Running the LAD experiment

1. Run `runLAD.py`. If you want, you can open the file and reduce the number of `iterations` (default is 50). This will save you time, but result in a less smooth performance curve with larger CI. This will generate a file called `result-lad.tsv`. 
2. Run `plotLAD.py` to plot the results.  
3. Run `resultsLAD.py` to print a summary of the results in table format. 
 a summary of the results in table format. 

## Running the Tangrams experiment

1. Run `runTangrams.py`. If you want, you can open the file and reduce the number of `iterations` (default is 3000). This will save you time, but result in a less smooth performance curve with larger CI. This will generate a file called `result-tangrams.tsv`. 
2. Run `plotTangrams.py` to plot the results.  
3. Run `resultsTangrams.py` to print a summary of the results in table format. 
 a summary of the results in table format. 
