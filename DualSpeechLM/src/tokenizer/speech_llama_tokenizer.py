import torch.nn as nn
import torch
import os
from typing import Any, Dict, List, Optional, Union
from transformers import LlamaTokenizer
from diffusers import DiffusionPipeline
from PIL import Image
from torchvision import transforms
import torchaudio
from .pipeline_stable_unclip_img2img import StableUnCLIPImg2ImgPipeline
from transformers import AutoTokenizer

WEIGHTS_NAME = 'seed_quantizer.pt'
DIFFUSION_NAME = 'diffusion_model'

from src.tokenizer.WavTokenizer.encoder.utils import convert_audio
from src.tokenizer.WavTokenizer.decoder.pretrained import WavTokenizer

class AudioTokenizer(nn.Module):
    def __init__(self,
                 model_path,
                 diffusion_model_path=None,
                 load_diffusion=False,
                 image_size=224,
                 device='cuda',
                 fp16=True,
                 **kwargs):
        super().__init__()
        from .qformer.qformer_quantizer import Blip2QformerQuantizer

        model = Blip2QformerQuantizer.from_pretrained(pretrained_model_path=model_path,
                                                      vit_precision='fp16' if fp16 else 'fp32',
                                                      **kwargs).eval()
        if diffusion_model_path is not None and load_diffusion:
            diffusion_model = StableUnCLIPImg2ImgPipeline.from_pretrained(diffusion_model_path,
                                                                          torch_dtype=torch.float16 if fp16 else torch.float32)
            self.diffusion_model = diffusion_model.to(device)
        else:
            self.diffusion_model = None

        model = model.to(device)

        processor = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=3),
            # transforms.Resize(image_size, interpolation=3),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        if fp16:
            model = model.half()

        shape_latents = torch.Size([1, 4, 96, 96])
        self.latents = torch.randn(shape_latents, generator=None, device=device, dtype=torch.float16, layout=torch.strided)

        shape_noise = torch.Size([1, 1024])
        self.noise = torch.randn(shape_noise, generator=None, device=device, dtype=torch.float16, layout=torch.strided)

        self.model = model
        self.processor = processor
        self.device = device
        self.fp16 = fp16

    def __len__(self):
        return self.model.n_embed

    def encode(self, image_torch):
        '''Convert a batch of img to code
        Args:
            model: The tokenizer model.
            img: [b, c, h, w]
        '''
        if len(image_torch.shape) == 3:
            image_torch = image_torch.unsqueeze(0)

        img = image_torch.to(self.device)
        if self.fp16:
            img = img.half()
        with torch.no_grad():
            id, _ = self.model.get_codebook_indices(img)
        return id.view(img.shape[0], -1)

    def decode(self, indices, negative_indices=None, guidance_scale=10, num_inference_steps=20):
        image_embeds = self.model.get_codebook_entry(indices)
        if negative_indices is not None:
            assert indices.shape == negative_indices.shape, 'Negative indices must have the same shape with indices'
            negative_image_embeds = self.model.get_codebook_entry(negative_indices)
        else:
            negative_image_embeds = None

        image = self.diffusion_model(
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            guidance_scale=guidance_scale,
            noise_level=0,
            num_inference_steps=num_inference_steps,
            latents=self.latents,
        ).images
        return image


class SpeechLlamaTokenizer(LlamaTokenizer):
    def __init__(self,
                 vocab_file,
                 unk_token="<unk>",
                 bos_token="<s>",
                 eos_token="</s>",
                 pad_token=None,
                 sp_model_kwargs: Optional[Dict[str, Any]] = None,
                 add_bos_token=True,
                 add_eos_token=False,
                 clean_up_tokenization_spaces=False,
                 device='cuda',
                 fp16=True,
                 load_diffusion=False,
                 **kwargs):
        super().__init__(vocab_file, unk_token, bos_token, eos_token, pad_token, sp_model_kwargs, add_bos_token, add_eos_token,
                         clean_up_tokenization_spaces, **kwargs)
        self.device = device
        self.fp16 = fp16
        self.pad_token = self.unk_token
        self.load_diffusion = load_diffusion

        wavtokenizer_config_path = "WavTokenizer/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        wavtokenizer_model_path = "WavTokenizer/WavTokenizer-large-unify-40token/wavtokenizer_large_unify_600_24k.ckpt"
        self._audio_tokenizer = WavTokenizer.from_pretrained0802(wavtokenizer_config_path, wavtokenizer_model_path)

    @property
    def audio_tokenizer(self):
        pass

    @property
    def num_image_tokens(self):
        return 500  # 

    def to(self, device):
        self.device = device
        if hasattr(self, '_audio_tokenizer'):
            self._audio_tokenizer.to(device=device)

    def encode_audio(
        self,
        audio_path=None,
        audio_pil=None,
        audio_torch=None,
        audio_size: int = 224,
    ):
        pass

    def decode_audio(self, indices, negative_indices=None, guidance_scale=10):
        audio_tokens = indices.to(self.device)
        features = self._audio_tokenizer.codes_to_features(audio_tokens)
        bandwidth_id = torch.tensor([0]).to(self.device) 
        audio_out = self._audio_tokenizer.decode(features, bandwidth_id=bandwidth_id)
        
        return audio_out

class SpeechLlama3BTokenizer(AutoTokenizer):
    def __init__(self,
                 device='cuda',
                 **kwargs):
        super().__init__(**kwargs)
        self.device = device

        wavtokenizer_config_path = "WavTokenizer/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        wavtokenizer_model_path = "WavTokenizer/WavTokenizer-large-unify-40token/wavtokenizer_large_unify_600_24k.ckpt"
        self._audio_tokenizer = WavTokenizer.from_pretrained0802(wavtokenizer_config_path, wavtokenizer_model_path)

    @property
    def audio_tokenizer(self):
        pass

    @property
    def num_image_tokens(self):
        return 500  # 

    def to(self, device):
        self.device = device
        if hasattr(self, '_audio_tokenizer'):
            self._audio_tokenizer.to(device=device)

    def encode_audio(
        self,
        audio_path=None,
        audio_pil=None,
        audio_torch=None,
        audio_size: int = 224,
    ):
        pass

    def decode_audio(self, indices, negative_indices=None, guidance_scale=10):
        audio_tokens = indices.to(self.device)
        features = self._audio_tokenizer.codes_to_features(audio_tokens)
        bandwidth_id = torch.tensor([0]).to(self.device) 
        audio_out = self._audio_tokenizer.decode(features, bandwidth_id=bandwidth_id)
        
        return audio_out

class MyWavTokenizer():
    def __init__(self,
                 device='cuda',
                 **kwargs):
        super().__init__(**kwargs)
        self.device = device

        wavtokenizer_config_path = "WavTokenizer/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        wavtokenizer_model_path = "WavTokenizer/WavTokenizer-large-unify-40token/wavtokenizer_large_unify_600_24k.ckpt"
        self._audio_tokenizer = WavTokenizer.from_pretrained0802(wavtokenizer_config_path, wavtokenizer_model_path)

    def to(self, device):
        self.device = device
        if hasattr(self, '_audio_tokenizer'):
            self._audio_tokenizer.to(device=device)

    def encode_audio(
        self,
        audio_path=None,
        audio_pil=None,
        audio_torch=None,
        audio_size: int = 224,
    ):
        pass

    def decode_audio(self, indices, negative_indices=None, guidance_scale=10):
        audio_tokens = indices.to(self.device)
        features = self._audio_tokenizer.codes_to_features(audio_tokens)
        bandwidth_id = torch.tensor([0]).to(self.device) 
        audio_out = self._audio_tokenizer.decode(features, bandwidth_id=bandwidth_id)
        
        return audio_out