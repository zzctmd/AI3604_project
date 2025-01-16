import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .memory_bank import MemoryBank
from .memory_attention import MemoryAttention
from .retrieve import RetrievalModule

class SamRAG(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        memory_feature_dim: int = 256,
        memory_max_size: int = 1000,
        retrieval_top_k: int = 5,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM with Memory Attention and Retrieval.

        Args:
          image_encoder (ImageEncoderViT): The backbone used to encode the image.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings.
          memory_feature_dim (int): Dimension of memory features.
          memory_max_size (int): Maximum size of the memory bank.
          retrieval_top_k (int): Number of top similar features to retrieve.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.memory_bank = MemoryBank(feature_dim=memory_feature_dim, max_size=memory_max_size)
        self.memory_attention = MemoryAttention(feature_dim=memory_feature_dim)
        self.retrieval_module = RetrievalModule(feature_dim=memory_feature_dim, top_k=retrieval_top_k)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, batched_input, multimask_output, image_size):
        input_images = self.preprocess(batched_input)
        image_embeddings = self.image_encoder(input_images)

        _, retrieved_indices = self.retrieval_module.query_index(image_embeddings)
        retrieved_features = self.memory_bank.retrieve(retrieved_indices.flatten())

        fused_features = self.memory_attention(image_embeddings, retrieved_features)

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=fused_features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=None,
            dense_prompt_embeddings=None,
            multimask_output=multimask_output,
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size),
        )

        self.memory_bank.add(image_embeddings)
        self.retrieval_module.add_to_index(image_embeddings)

        outputs = {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
        }
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize pixel values and pad to a square input.
        """
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x