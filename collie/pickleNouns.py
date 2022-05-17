import pickle
import torch
import collie

words = collie.read_text_file('nouns1000.txt')

tot = []
for i in range(0,10):
    print(i)
    emb = collie.encode_texts(words[i*100:(i+1)*100])
    tot.append(emb)
tot = torch.cat(tot)
pickle.dump(tot, open("nouns1000.pickle", 'wb'))