"""
Frozen pretrained encoders shared by Quiz 2 and Quiz 3.

  image: ResNet-50 (ImageNet, 2048-d), ViT-B/16 (ImageNet, 768-d), CLIP ViT-B/32 (512-d projected)
  text : BERT-base-uncased (768-d, mean pooled), CLIP text encoder (512-d projected)

CLIP is called through vision_model / text_model + projection layers directly, which
behaves the same in transformers 4.x and 5.x.
"""

import numpy as np
import torch
import torchvision.transforms as T

from .common import DEVICE, IMAGENET_MEAN, IMAGENET_STD

CLIP_ID = 'openai/clip-vit-base-patch32'
BERT_ID = 'bert-base-uncased'
_TV_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
_CACHE = {}


def clip():
    if 'clip' not in _CACHE:
        from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer
        _CACHE['clip'] = (CLIPModel.from_pretrained(CLIP_ID).to(DEVICE).eval(),
                          CLIPImageProcessor.from_pretrained(CLIP_ID), CLIPTokenizer.from_pretrained(CLIP_ID))
    return _CACHE['clip']


class ImageEncoder:
    DIMS = {'resnet50': 2048, 'vit_b_16': 768, 'clip': 512}

    def __init__(self, name):
        self.name = name
        if name in ('resnet50', 'vit_b_16'):
            from torchvision import models
            if name == 'resnet50':
                m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); m.fc = torch.nn.Identity()
            else:
                m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1); m.heads = torch.nn.Identity()
            self.model = m.to(DEVICE).eval()
        elif name != 'clip':
            raise ValueError(name)

    @torch.no_grad()
    def __call__(self, images):
        images = [im.convert('RGB') for im in images]
        if self.name == 'clip':
            model, proc, _ = clip()
            px = proc(images=images, return_tensors='pt')['pixel_values'].to(DEVICE)
            return model.visual_projection(model.vision_model(pixel_values=px).pooler_output).float().cpu()
        x = torch.stack([_TV_TF(im) for im in images]).to(DEVICE)
        return self.model(x).float().cpu()


class TextEncoder:
    DIMS = {'bert': 768, 'clip': 512}

    def __init__(self, name):
        self.name = name
        if name == 'bert':
            from transformers import AutoModel, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(BERT_ID)
            self.model = AutoModel.from_pretrained(BERT_ID).to(DEVICE).eval()
        elif name != 'clip':
            raise ValueError(name)

    @torch.no_grad()
    def __call__(self, texts):
        if self.name == 'clip':
            model, _, tok = clip()
            t = tok(list(texts), padding=True, truncation=True, max_length=77, return_tensors='pt').to(DEVICE)
            out = model.text_model(input_ids=t['input_ids'], attention_mask=t['attention_mask'])
            return model.text_projection(out.pooler_output).float().cpu()
        t = self.tok(list(texts), padding=True, truncation=True, max_length=64, return_tensors='pt').to(DEVICE)
        h = self.model(**t).last_hidden_state
        mask = t['attention_mask'].unsqueeze(-1).float()
        return ((h * mask).sum(1) / mask.sum(1)).float().cpu()         # mean pooling over real tokens


@torch.no_grad()
def clip_patch_tokens(images):
    """Quiz 3 encoder: CLIP ViT-B/32 token features (B, 50, 768) = [CLS] + 7x7 patches."""
    model, proc, _ = clip()
    px = proc(images=[im.convert('RGB') for im in images], return_tensors='pt')['pixel_values'].to(DEVICE)
    return model.vision_model(pixel_values=px).last_hidden_state.float().cpu()


def batched_images(hf_split, batch=64):
    """Yields lists of PIL images from a Hugging Face split, decoding each image once."""
    for i in range(0, len(hf_split), batch):
        yield [hf_split[j]['image'] for j in range(i, min(i + batch, len(hf_split)))]
