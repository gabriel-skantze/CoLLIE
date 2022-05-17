import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device " + device)
model, preprocess = clip.load("ViT-B/32", device=device)

with torch.no_grad():

    def encode_images(files):
        s = torch.stack([preprocess(Image.open(file)).to(device) for file in files])
        res = model.encode_image(s).to(device)
        res /= res.norm(dim=-1, keepdim=True)
        return res

    def encode_texts(texts):
        res = model.encode_text(clip.tokenize(texts).to(device))
        res /= res.norm(dim=-1, keepdim=True)
        return res

    def normalize(emb):
        emb /= emb.norm(dim=-1, keepdim=True)
        return emb

    # Returns the result sorted (pairs with (score,N))
    def find_best(queryTensor, dataTensor, dataMeta = None):
        if dataMeta is None:
            dataMeta = range(len(dataTensor))
        if len(queryTensor.size()) == 1:
            queryTensor = queryTensor.unsqueeze(0)
        if dataTensor.dtype == torch.float16:
            queryTensor = queryTensor.half().to(device)
        similarity = (100.0 * queryTensor @ dataTensor.T).softmax(dim=-1)
        return sorted(zip(similarity[0], dataMeta), reverse=True, key=lambda x: x[0])

    def find_similarity(queryTensor, dataTensor):
        if len(queryTensor.size()) == 1:
            queryTensor = queryTensor.unsqueeze(0)
        similarity = (100.0 * queryTensor @ dataTensor.T).softmax(dim=-1)
        return similarity[0].tolist()

    def dot_product(queryTensor, dataTensor):
        if len(queryTensor.size()) == 1:
            queryTensor = queryTensor.unsqueeze(0)
        similarity = (100.0 * queryTensor @ dataTensor.T)
        return similarity[0].tolist()

    def read_text_file(path):
        with open(path, 'r') as file:
            data = [line.strip() for line in file.readlines()]
        return data

    def mrr(result, k):
        if result is None:
            return None
        rank = 0
        for j in range(len(result)):
            if result[j][1] == k:
                rank = j+1
        return 1 / rank
