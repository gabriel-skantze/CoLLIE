import os
import pickle

import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR
from torch import Tensor

import collie

class Model:

    def __init__(self):
        self.X = torch.empty(0, 512)
        self.Y = torch.empty(0, 512)
        self.model = None

    def save(self, filename):
        pickle.dump(self.model, open(filename + ".model", 'wb'))

    def load(self, filename):
        self.model = pickle.load(open(filename + ".model", 'rb'))

    def model_predict(self, x):
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        return self.model.predict(x)[0]

    # Returns the result sorted (pairs with (score,N))
    def find_best(self, text, imgEmbeds):
        with torch.no_grad():
            textEmbed = collie.encode_texts([text]).flatten()
            if self.model is not None:
                textEmbedCPU = textEmbed.cpu()
                res = Tensor(self.model_predict(textEmbedCPU))
                textEmbedAdj = textEmbedCPU.add(res)
                return collie.find_best(textEmbedAdj, imgEmbeds)
            else:
                return collie.find_best(textEmbed, imgEmbeds)


defaultAlpha = 0.001

class BaselineModel(Model):

    def __init__(self, name="Baseline"):
        self.name = name
        super().__init__()

    def model_predict(self, x):
        return [0.0 for i in range(512)]

    def teach(self, textEmbed, imgEmbeds, k):
        pass

    def train(self):
        pass


class CollieModelWOScaling(Model):

    def __init__(self, name="CoLLIE (without scaling)"):
        self.alpha = defaultAlpha
        self.name = name
        super().__init__()

    def teach(self, text, imgEmbeds, k):
        with torch.no_grad():
            textEmbed = collie.encode_texts([text]).flatten()
            imgEmbed = imgEmbeds[k].flatten()
            diff = imgEmbed.sub(textEmbed)
            self.X = torch.cat((self.X, textEmbed.unsqueeze(0).cpu()))
            self.Y = torch.cat((self.Y, diff.unsqueeze(0).cpu()))

    def train(self):
        self.model = Ridge(alpha=self.alpha).fit(self.X, self.Y)


pickledNouns = pickle.load(open(os.path.join(os.path.dirname(__file__), 'nouns1000.pickle'), 'rb'))

class CollieFullModel(Model):

    def __init__(self, name="CoLLIE"):
        self.alpha = defaultAlpha
        self.name = name
        self.Xm = torch.empty(0, 512)
        self.Ym = torch.empty(0)
        self.modelM = None
        self.Xm = pickledNouns
        self.Ym = torch.zeros(len(pickledNouns))
        super().__init__()

    def save(self, filename):
        pickle.dump(self.model, open(filename + ".adjust.model", 'wb'))
        pickle.dump(self.modelM, open(filename + ".scale.model", 'wb'))

    def load(self, filename):
        self.model = pickle.load(open(filename + ".adjust.model", 'rb'))
        self.modelM = pickle.load(open(filename + ".scale.model", 'rb'))

    def model_predict(self, x):
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        return self.model.predict(x)[0] * self.modelM.predict(x)[0]

    def teach(self, text, imgEmbeds, k):
        with torch.no_grad():
            imgEmbed = imgEmbeds[k].flatten()
            textEmbed = collie.encode_texts([text]).flatten()
            diff = imgEmbed.sub(textEmbed)
            textEmbeds = textEmbed.unsqueeze(0).cpu()
            self.X = torch.cat((self.X, textEmbeds))
            self.Y = torch.cat((self.Y, diff.unsqueeze(0).cpu()))
            self.Xm = torch.cat((self.Xm, textEmbeds))
            self.Ym = torch.cat((self.Ym, torch.ones(1)))

    def train(self):
        self.model = Ridge(alpha=self.alpha).fit(self.X, self.Y)
        self.modelM = SVR().fit(self.Xm, self.Ym)


class FewShotLearner(Model):

    def __init__(self, name="Few-shot learner"):
        self.alpha = defaultAlpha
        self.name = name
        self.dict = []
        self.X = torch.empty(0, 512)
        self.Y = []
        self.model = None

    def teach(self, text, imgEmbeds, k):
        with torch.no_grad():
            if not (text in self.dict):
                self.dict.append(text)
            label = self.dict.index(text)
            imgEmbed = imgEmbeds[k].unsqueeze(0).cpu()
            self.X = torch.cat((self.X, imgEmbed))
            self.Y.append(label)

    def train(self):
        if len(self.dict) > 1:
            self.model = LogisticRegression().fit(self.X, self.Y)

    # Returns the result sorted (pairs with (score,N))
    def find_best(self, text, imgEmbeds):
        with torch.no_grad():
            if text in self.dict and self.model is not None:
                label = self.dict.index(text)
                result = self.model.predict_proba(imgEmbeds.cpu())
                results = []
                for i in range(len(result)):
                    results.append([result[i,label],i])
                results = sorted(results, reverse=True, key=lambda x: x[0])
                return results
            else:
                None