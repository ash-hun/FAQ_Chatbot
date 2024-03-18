import os
import pathlib
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

class hfModelLoader:
    def __init__(self, model_name:str, path:str=None) -> None:
        """ Get HuggingFace Embedding Model and Download to your local path. 
        
            #Args

            model: model_name
            path: path where you want to save.

            #Usage Example

            loader = HuggingFaceEmbeddingDownLoader()
            loader.download()
        """
        self.model_name = model_name
        self.path = path

    def download(self):
        if self.path is None:
            self.path = pathlib.Path(__file__).parent.parent/'model'/self.model_name

        downloader = SentenceTransformer(model_name_or_path=self.model_name)
        os.makedirs(self.path, exist_ok=True)
        downloader.save(str(self.path))
        print(f'model {self.model_name} downloaded at path {self.path}.')

def getHFEmbedding(hf_model, device='mps'):
    """ Get model path from local and return with preset configuration. """
    model_path = pathlib.Path('model')/hf_model

    # check model exsists in model folder..
    if not model_path.exists():
        print(f"Model path {model_path} does not exist. Downloading the model...")
        loader = hfModelLoader(model_name=hf_model)
        loader.download()
    else:
        print(f"Model already exists at {model_path}. No need to download...")

    # Initialize and return
    hf_embedding = HuggingFaceEmbeddings(
        model_name=str(model_path), 
        model_kwargs={"device":device}, 
        encode_kwargs={"normalize_embeddings":True}
    )
    
    return hf_embedding