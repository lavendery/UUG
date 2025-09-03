
import logging
import json
import contextlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizer, StoppingCriteriaList, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from transformers import LlamaForCausalLM 
from transformers import Phi3ForCausalLM 
from .modeling_whisper import WhisperModel
from .utils import StoppingCriteriaSub
# from vector_quantize_pytorch import ResidualVQ
# from .causal_att import AnyGPT
from .modules3.transformer import TransformerBlock
from .modules3.decoder import Decoder as CodecDecoder   
from .modules3.encoder import Encoder as CodecEncoder
from .modules3.conv_layer import Conv1d, ConvTranspose1d
from .modules3.quantizer import Quantizer

class Adapter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ):
        super(Adapter, self).__init__()

        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x

class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        # self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        self.gn = nn.Identity()

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]
    
    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)[...,:-self.causal_padding]


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=2):
        super().__init__()
        # print('in_channels ', in_channels)
        self.layers = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, groups=groups),
            GroupNorm(out_channels),
            Swish(),
            CausalConv1d(in_channels=out_channels, out_channels=out_channels,
                         kernel_size=kernel_size, groups=groups)
        )

    def forward(self, x):
        return x + self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, stride):
        super().__init__()
        out_channels = in_channels # * 2
        self.layers = nn.Sequential(
            CausalConvTranspose1d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=2*stride, stride=stride),
            GroupNorm(out_channels),
            Swish(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels),
            GroupNorm(out_channels),
            Swish(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels),
        )

    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, C, D, strides=[5, 4, 2]):
        super().__init__()
        self.layers = [
            CausalConv1d(in_channels=D, out_channels=D, kernel_size=3),
            Swish()
        ]
        for stride in strides:
            self.layers += [
                DecoderBlock(in_channels=D, stride=stride),
                GroupNorm(D),
                Swish()
            ]
        self.layers += [CausalConv1d(in_channels=D, out_channels=C, kernel_size=3)]
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x, g=None):
        x = x.transpose(1, 2)
        if g is not None:
            up_g = g.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = x + up_g
        
        h = self.layers[: -1](x) 
        y = self.layers[-1](h) 

        return y.transpose(1, 2) 

class USTokenizer(nn.Module):
    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def __init__(
        self,
        llama_path="",
        phi_path="",
        whisper_path="",
        freeze_whisper=True,
        lora=False,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,

        multi_prompt=False,
        prompt_path="",
        prompt_template="",
        max_txt_len=128,
        end_sym="<|end_of_text|>", 
        low_resource=False,  # use 8 bit
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        codec_config=None,  
    ):
        super().__init__()
        self.lora = lora
        self.multi_prompt = multi_prompt
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.low_resource = low_resource

        logging.info('Loading LLaMA Tokenizer')
        self.llama_path = llama_path
        self.phi_path = phi_path
        if self.llama_path:
            # self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_path, use_fast=False) # 
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, use_fast=False) #Llama-3.2-3B-Instruct or Phi-3.5-mini-instruct
            self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.llama_tokenizer.padding_side = "right"
        elif self.phi_path:
            self.llama_tokenizer = AutoTokenizer.from_pretrained(phi_path, use_fast=False)
            self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token
            self.llama_tokenizer.padding_side = "right"

        logging.info('Loading LLaMA Model')
        if self.low_resource:
            if self.llama_path:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_path,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map={"": device_8bit},
                )
            elif self.phi_path:
                self.llama_model = Phi3ForCausalLM.from_pretrained(
                    phi_path,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map={"": device_8bit},
                )
        else:
            if self.llama_path:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_path,
                    torch_dtype=torch.bfloat16,
                )
            elif self.phi_path:
                self.llama_model = Phi3ForCausalLM.from_pretrained(
                    phi_path,
                    torch_dtype=torch.bfloat16,
                )

        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLaMA Done')

        # if self.lora:
        #     self.peft_config = LoraConfig(
        #         task_type=TaskType.CAUSAL_LM, 
        #         inference_mode=False, 
        #         r=lora_rank, 
        #         lora_alpha=lora_alpha, 
        #         lora_dropout=lora_dropout,
        #     )
        #     self.llama_model = get_peft_model(self.llama_model, self.peft_config)
        #     self.llama_model.print_trainable_parameters()
        #     logging.info('LoRA Training')

        assert whisper_path
        logging.info('Loading Whisper Model')
        self.speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder
        # self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model) # 1280
        if freeze_whisper:
            for name, param in self.speech_encoder.named_parameters():
                param.requires_grad = False
            self.speech_encoder.eval()
            logging.info("freeze Whisper")
        

        # prepare prompts
        self.prompt_dict = {}
        if prompt_path:
            try:
                raw_prompts = json.load(open(prompt_path, "r"))
            except:
                print("Failed to load prompt! Try to use utf-8 encoding.")
                raw_prompts = json.load(open(prompt_path, "r", encoding='utf-8'))
            for task in raw_prompts.keys():
                filted_prompts = [raw_prompt for raw_prompt in raw_prompts[task] if "<SpeechHere>" in raw_prompt]
                self.prompt_dict[task] = [prompt_template.format(p) for p in filted_prompts]
            print("Loading training prompts done!")

        self.downsampling_conv = Conv1d(codec_config.input_channels, codec_config.input_channels, kernel_size=3, stride=2)
        self.codec_encoder = CodecEncoder(
            input_channels=codec_config.input_channels,
            encode_channels=codec_config.encode_channels,
            channel_ratios=codec_config.enc_ratios,
            strides=codec_config.enc_strides,
            kernel_size=codec_config.enc_kernel_size,
            bias=codec_config.bias,
            block_dilations=codec_config.enc_block_dilations,
            unit_kernel_size=codec_config.enc_block_kernel_size
        )     
        self.speech_llama_conv_proj = Conv1d(
            in_channels=self.codec_encoder.out_channels, # input_channels
            out_channels=codec_config.code_dim,    # out_channels
            kernel_size=3,
            stride=1,
            bias=False
        )
        self.vq = Quantizer(
            code_dim=codec_config.code_dim,
            codebook_num=codec_config.codebook_num,
            codebook_size=codec_config.codebook_size
        )
        self.codec_decoder = CodecDecoder(
            code_dim=codec_config.code_dim,
            output_channels=codec_config.output_channels,
            decode_channels=codec_config.decode_channels,
            channel_ratios=codec_config.dec_ratios,
            strides=codec_config.dec_strides,
            kernel_size=codec_config.dec_kernel_size,
            bias=codec_config.bias,
            block_dilations=codec_config.dec_block_dilations,
            unit_kernel_size=codec_config.dec_block_kernel_size
        )
        self.upsampling_conv = ConvTranspose1d(
            in_channels=codec_config.output_channels,
            out_channels=codec_config.output_channels,
            kernel_size=3,
            stride=2,
        )
        
        self.llama_linear = nn.Linear(codec_config.output_channels, self.llama_model.config.hidden_size)
        self.llama_conv = Conv1d(
            in_channels=self.llama_model.config.hidden_size,
            out_channels=self.llama_model.config.hidden_size,
            kernel_size=3,
            stride=1,
        )

        self.mse_loss = nn.MSELoss()

    def _encode_auditory_feature(self, whisper_embeds=None, return_USToken=False):
        with self.maybe_autocast():
            whisper_embeds = whisper_embeds.transpose(1, 2) # [B, T, whisper_dim] -> [B, whisper_dim, T]
            speech_embeds = self.downsampling_conv(whisper_embeds)  #  [B, whisper_dim, T/2]
            speech_embeds = self.codec_encoder(speech_embeds) # [B, whisper_dim, T/2]
            speech_embeds = self.speech_llama_conv_proj(speech_embeds) # [B, whisper_dim, T/2]
            quantized_features, commitment_loss, perplexity = self.vq(speech_embeds) # [B, whisper_dim, T/2]
            
            if return_USToken:
                ######### extract speech token during inference, you can only use the 'idx' in this code ########
                _, USToken = self.vq.codebook.forward_index(speech_embeds.transpose(2, 1))
                ######### extract speech token during inference, you can only use the 'idx' in this code ########
            else:
                USToken = None

            # reconstruct the whisper features
            recon_features = self.codec_decoder(quantized_features) # [B, whisper_dim, T/2]
            recon_features = self.upsampling_conv(recon_features) # [B, whisper_dim, T]
            time_dim = min(whisper_embeds.size(2), recon_features.size(2))
            recon_loss = self.mse_loss(recon_features[:, :, :time_dim], whisper_embeds[:, :, :time_dim])
            
            quantized_features = quantized_features.transpose(1, 2) # [B, whisper_dim, T] -> [B, T, whisper_dim]
            speech_atts = torch.ones(quantized_features.size()[:-1], dtype=torch.long).to(quantized_features.device) #torch.Size([2, 300])

        return quantized_features, speech_atts, commitment_loss, perplexity, recon_loss, USToken

    def encode_speech(self, spectrogram=None, raw_wav=None, audio_padding_mask=None, return_USToken=False):
        with self.maybe_autocast():
            whisper_output = self.speech_encoder(spectrogram, return_dict=True)
            speech_embeds = whisper_output.last_hidden_state 
            x_16k_len = int(raw_wav.shape[1] / 320)
            speech_embeds = speech_embeds[:,:x_16k_len,:]

        return self._encode_auditory_feature(whisper_embeds=speech_embeds, return_USToken=return_USToken)

    def prompt_wrap(self, embeds, atts, prompt, multi_prompt=False):
        if prompt:
            if multi_prompt:
                p_before = []
                p_after = []
                for i, p in enumerate(prompt):
                    b, a = p.split("<SpeechHere>")
                    p_before.append(b)
                    p_after.append(a)
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids)

                # speech_embeds wrapped with prompts_embeds are padded to the same length here
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", padding="longest", add_special_tokens=False
                ).to(embeds.device)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            else:
                batch_size = embeds.shape[0]
                p_before, p_after = prompt.split("<SpeechHere>")

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            return wrapped_embeds, wrapped_atts
        else:
            return embeds, atts

    def forward(self, samples, verbose=False):
        # detect whether there are multi tasks in this batch
        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:
                prompt = [self.prompt_dict[task][-1] for task in samples["task"]]
                if "Q" in samples:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, samples["Q"]) ]
            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])

        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)
        speech_embeds, speech_atts, commitment_loss, perplexity, recon_loss, _ = self.encode_speech(spectrogram=spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask, return_USToken=False)

        speech_embeds = self.llama_linear(speech_embeds) # [B, T, whisper_dim] -> [B, T, llama_dim]
        speech_embeds = self.llama_conv(speech_embeds.transpose(1, 2)).transpose(1, 2) # [B, T, llama_dim] -> [B, T, llama_dim]
        # wrap speech_embeds with prompts
        if self.prompt_dict:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)

        # prepare inputs for LLM
        text = [t + self.end_sym for t in samples["text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(spectrogram.device)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(
                [speech_atts.shape[0], speech_atts.shape[1] + 1],
                dtype=torch.long
            ).to(spectrogram.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts, to_regress_tokens.attention_mask], dim=1)

        # calulate loss
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

        if verbose:
            nvocab = self.llama_model.config.vocab_size
            results = outputs.logits[:, empty_targets.size(1) - 1: -1, :].contiguous().view(-1, nvocab).argmax(dim=-1)
            labels = targets[:, empty_targets.size(1):].contiguous().view(-1)
            mask = (labels != -100)
            correct = (results[mask] == labels[mask]).float().sum()
            total = len(labels[mask])
        llama_loss = loss
        commitment_loss = torch.sum(commitment_loss)
        loss = 1.0*commitment_loss + 45.0*recon_loss + 5.0*llama_loss
        if verbose:
            return {"loss": loss, "correct": correct, "total": total}
        return {
            "loss": loss,
            "llama_loss": llama_loss,
            "commitment_loss": commitment_loss,
            "recon_loss": recon_loss,
            "perplexity": perplexity,
        }

    @torch.no_grad()
    def generate(self, samples, generate_cfg, prompts=None):
        batch_size = samples["spectrogram"].shape[0]

        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)
        speech_embeds, speech_atts, commitment_loss, perplexity, recon_loss, USToken = self.encode_speech(spectrogram=spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask, return_USToken=True)
        speech_embeds = self.llama_linear(speech_embeds) # [B, T, whisper_dim] -> [B, T, llama_dim]
        speech_embeds = self.llama_conv(speech_embeds.transpose(1, 2)).transpose(1, 2) # [B, T, llama_dim] -> [B, T, llama_dim]

        if prompts is not None:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompts, multi_prompt=True)

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attns = torch.cat([atts_bos, speech_atts], dim=1)

        stop_words_ids = [torch.tensor([128001]).to(embeds.device)]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=generate_cfg.get("max_new_tokens", 200),
            stopping_criteria=stopping_criteria,
            num_beams=generate_cfg.get("num_beams", 4),
            do_sample=generate_cfg.get("do_sample", False),
            min_length=generate_cfg.get("min_length", 1),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
            length_penalty=generate_cfg.get("length_penalty", 1.0),
            attention_mask=attns,
            pad_token_id = self.llama_tokenizer.pad_token_id,
        )
        text = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return text, commitment_loss, recon_loss, perplexity, USToken

    @classmethod
    def from_config(cls, config):
        llama_path = config.get("llama_path", "")
        phi_path = config.get("phi_path", "")
        whisper_path = config.get("whisper_path")
        freeze_whisper = config.get("freeze_whisper", True)

        lora = config.get("lora", False)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.1)

        multi_prompt = config.get("multi_prompt", False)
        prompt_path = config.get("prompt_path", "")
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 128)
        end_sym = config.get("end_sym", "<|end_of_text|>")
        low_resource = config.get("low_resource", False)
        device_8bit = config.get("device_8bit", 0)

        codec_config = config.get("codec_config", {})

        model = cls(
            llama_path=llama_path,
            phi_path=phi_path,
            whisper_path=whisper_path,
            freeze_whisper=freeze_whisper,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            multi_prompt=multi_prompt,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            codec_config=codec_config,
        )

        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            logging.info("Load ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt['model'], strict=False)

        return model
