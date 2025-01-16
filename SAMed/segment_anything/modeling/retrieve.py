from transformers import DINOImageProcessor, DinoModel
import faiss
import torch

class DINOv2EmbeddingExtractor:
    def __init__(self, model_name="facebook/dino-v2-base"):
        """
        Initialize the DINOv2 model for embedding extraction.
        Args:
          model_name (str): The pretrained DINO model name.
        """
        self.processor = DINOImageProcessor.from_pretrained(model_name)
        self.model = DinoModel.from_pretrained(model_name)

    def extract_embedding(self, image):
        """
        Extract embeddings from an input image.
        Args:
          image (PIL.Image or torch.Tensor): Input image.

        Returns:
          torch.Tensor: Embedding of the image.
        """
        inputs = self.processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)


class RetrievalModule:
    def __init__(self, feature_dim, top_k=5):
        """
        Initialize the retrieval module.
        Args:
          feature_dim (int): Dimension of the features.
          top_k (int): Number of top matches to retrieve.
        """
        self.index = faiss.IndexFlatL2(feature_dim)
        self.top_k = top_k

    def add_to_index(self, embeddings):
        """
        Add embeddings to the FAISS index.
        Args:
          embeddings (torch.Tensor): Features to add.
        """
        self.index.add(embeddings.cpu().numpy())

    def query_index(self, query_embeddings):
        """
        Query the FAISS index for similar embeddings.
        Args:
          query_embeddings (torch.Tensor): Query features.

        Returns:
          (np.ndarray, np.ndarray): Distances and indices of the top matches.
        """
        distances, indices = self.index.search(query_embeddings.cpu().numpy(), self.top_k)
        return distances, indices
