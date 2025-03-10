import io
import torch

from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel

model_name = "microsoft/swin-large-patch4-window7-224"

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)    
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_embeddings(binaries_list: list):
    print('Calculating images embeddings')
    images = [Image.open(io.BytesIO(data)) for data in binaries_list]

    inputs = feature_extractor(images = images, return_tensors = 'pt', padding = True).to(device)

    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.cpu().numpy()

    return embeddings.reshape((-1,embeddings.shape[1]*embeddings.shape[2]))
    