import os
import pathlib
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

class hfModelLoader:
    def __init__(self, model_name:str, path:str=None) -> None:
        """ 
            허깅페이스의 임베딩모델을 다운받아 로컬경로에 저장하는 클래스 객체를 만듭니다.
        
            # Args
            - model: HuggingFace의 임베딩모델 명을 입력합니다.
            - path: 저장할 로컬경로를 입력합니다.
        """
        self.model_name = model_name
        self.path = path

    def download(self):
        """
            정해진 임베딩모델을 다운로드 합니다.
        """
        if self.path is None:
            self.path = pathlib.Path(__file__).parent.parent/'model'/self.model_name

        downloader = SentenceTransformer(model_name_or_path=self.model_name)
        os.makedirs(self.path, exist_ok=True)
        downloader.save(str(self.path))
        print(f'model {self.model_name} downloaded at path {self.path}.')

def getHFEmbedding(hf_model, device='cpu'):
    """ 
        임베딩모델의 경로와 다운유무를 확인한 후 저장.
    """
    model_path = pathlib.Path('model')/hf_model

    if not model_path.exists():
        print(f" ►► Model path {model_path} does not exist. Downloading the model...")
        loader = hfModelLoader(model_name=hf_model)
        loader.download()
    else:
        print(f" ►► Model already exists at {model_path}. No need to download...")

    hf_embedding = HuggingFaceEmbeddings(
        model_name=str(model_path), 
        model_kwargs={"device":device}, 
        encode_kwargs={"normalize_embeddings":True}
    )
    
    return hf_embedding