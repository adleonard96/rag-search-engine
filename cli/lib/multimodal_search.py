from collections import defaultdict
from PIL import Image
from sentence_transformers import SentenceTransformer
from .semantic_search import cosine_similarity

class MultimodalSearch:
    
    def __init__(self, docs=[], model_name="clip-ViT-B-32", mapping= defaultdict(dict)):
        self.texts = docs
        self.mapping = mapping
        self.model = SentenceTransformer(model_name)
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        for i, movie in enumerate(docs):
            self.mapping[movie]["embedding"] = self.text_embeddings[i]
        
        
    def embed_image(self, image):
        return self.model.encode([Image.open(image)])[0]
    
    def search_with_image(self, image):
        image_embedding = self.embed_image(image)
        
        for i, embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(embedding, image_embedding)
            self.mapping[self.texts[i]]["score"] = similarity
            
        top = sorted(self.mapping.items(), key= lambda x: x[1]["score"], reverse=True)[:5]
        
        for i, result in enumerate(top):
            print(f"{i + 1}. {result[1]["title"]} (similarity: {result[1]["score"]:.3f})")
            print(f"{result[1]["description"][:100]}")
        
    
def verify_image_embedding(image_path):
    modal = MultimodalSearch()
    print(f"Embedding shape: {modal.embed_image(image_path).shape[0]} dimensions")