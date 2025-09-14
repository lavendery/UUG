"""
# Input: text.scp, wav.scp. We expect to directly quantized to speech into tokens
# Output: tar
# Author: DualSpeechLM team 
"""
import json
import numpy as np
import torch
import argparse
import sys
import time
import os
import webdataset as wds
from omegaconf import OmegaConf
import hydra
import glob
import uuid
import json
import pickle
from src.tokenizer.USTokenizer.models.USTokenizer import USTokenizer  
from src.tokenizer.WavTokenizer.encoder.utils import convert_audio
import torchaudio
from src.tokenizer.WavTokenizer.decoder.pretrained import WavTokenizer
USTokenizer_config_path = "src/tokenizer/USTokenizer/configs/decode_config.yaml"

def save_data_to_json(file_path, data):
    """
    Saves the given data to a JSON file.
    :param file_path: Path to the JSON file
    :param data: List of data entries to save
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving data to {file_path}: {e}")

def get_parser():
    parser = argparse.ArgumentParser(
        description="convert a data list, do tokenization and save as a torch .pt file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-text-file", type=str, default=None, help="text file in the format <exampe_id> <content>")
    parser.add_argument("--input-wav-file", type=str, default=None, help="token file in the format <exampe_id> <content>")
    parser.add_argument("--output-file", type=str, default=None, help="the patch of save pt or json")
    parser.add_argument("--rank", type=int, help="local GPU rank, if applicable")
    parser.add_argument("--tokenizer", type=str, default=None, help="what tokenizer to use")
    return parser

def read_text_file(file_path):
    ans = {}
    f = open(file_path)
    for line in f:
        tmp = line.strip().split(' ')
        ans[tmp[0]] = ' '.join(tmp[1:]) # add text content
    return ans 

def read_wav_file(file_path):
    ans = {}
    f = open(file_path)
    for line in f:
        tmp = line.strip().split(' ')
        ans[tmp[0]] = tmp[1] # add text content
    return ans 

def main(args):
    args = get_parser().parse_args(args)
    process_start_time = time.time()   
    args.rank -= 1 # run.pl starts from 1 but the exact jobid / gpuid starts from 0   
    max_gpu = torch.cuda.device_count()
    device_rank = (args.rank % max_gpu) #
    device = torch.device(f"cuda:{device_rank}")
    # GPU tokenizers 
    if args.tokenizer == "hubert":  
        tokenizer = HubertTokenizer(device=device)
    elif args.tokenizer == "USTokenizer": 
        USTokenizer_config = OmegaConf.load(USTokenizer_config_path)
        tokenizer = USTokenizer.from_config(USTokenizer_config.model)
    
    elif args.tokenizer == "WavTokenizer": 
        wavtokenizer_config_path = "WavTokenizer/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        wavtokenizer_model_path = "WavTokenizer/WavTokenizer-large-unify-40token/wavtokenizer_large_unify_600_24k.ckpt"
        wav_tokenizer = WavTokenizer.from_pretrained0802(wavtokenizer_config_path, wavtokenizer_model_path)
    elif args.tokenizer == "encodec":
        pass
    elif args.tokenizer == "stable-codec":
        stable_tokenizer = StableCodec(  
                        model_config_path="checkpoint/stabilityai/stable-codec-speech-16k/model_config.json",
                        ckpt_path="checkpoint/stabilityai/stable-codec-speech-16k/model.ckpt", # optional, can be `None`,
                        device = device
                    )
    elif args.tokenizer == "others":
        pass

    if args.tokenizer == "hubert" or args.tokenizer == "USTokenizer":
        tokenizer = tokenizer.to(device)
    elif args.tokenizer == "WavTokenizer":
        wav_tokenizer = wav_tokenizer.to(device)
    elif args.tokenizer == "stable-codec":
        stable_tokenizer.set_posthoc_bottleneck("1x46656_400bps") 
        stable_tokenizer = stable_tokenizer.to(device)
        

    text_dict = read_text_file(args.input_text_file) # read the content
    wav_dict = read_wav_file(args.input_wav_file) # read the audio path
    cnt = 0 # the number of item
    save_pattern = args.output_file + f"/%07d_{args.rank}.tar"
    with wds.ShardWriter(save_pattern, maxcount=100000) as sink:
        for bs_name in wav_dict.keys(): # travel the audio list
            tmp_dict = {}
            if bs_name not in text_dict.keys():
                continue
            content = text_dict[bs_name] 
            wav_path = wav_dict[bs_name]
            
            if args.tokenizer == "hubert" or args.tokenizer == "USTokenizer":
                try:
                    tmp_token = tokenizer.tokenize(wav_path)
                except Exception as e:
                    print(f"Error in {wav_path} with error: {e}")
                    continue
            elif args.tokenizer == "WavTokenizer":
                try:
                    wav, sr = torchaudio.load(wav_path)
                    if wav.shape[0] > 2: 
                        wav = wav[0:1, :]
                    wav = convert_audio(wav, sr, 24000, 1) 
                    bandwidth_id = torch.tensor([0])
                    wav=wav.to(device)
                    features, wavtokenizer_discrete_code= wav_tokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
                except:
                    print(f"Error in {wav_path}")
                    continue
                tmp_token = wavtokenizer_discrete_code.squeeze()
            elif args.tokenizer == "stable-codec":
                try:
                    latents, stable_tokens = stable_tokenizer.encode(wav_path, posthoc_bottleneck = True)
                except:
                    print(f"Error in {wav_path}")
                    continue
                    
                tmp_token = stable_tokens[0][0].squeeze()

            if isinstance(tmp_token, torch.Tensor):
                assert tmp_token.dim() == 1
                tmp_token = tmp_token.cpu()
            key_str = uuid.uuid4().hex
            tmp  = {}
            tmp['id'] = bs_name
            tmp['tokenizer'] = args.tokenizer

            sample = {'audio_ids': tmp_token.view(-1).cpu().tolist(), 'text': content, 'metadata': tmp}

            # # only wav tokenizer
            # sample = {'audio_ids': tmp_token.view(-1).cpu().tolist(), 'text': content, 'metadata': tmp, 'wav_tokenizer': wavtokenizer_discrete_code.view(-1).cpu().tolist()}
            sink.write({'__key__': key_str, 'pkl': pickle.dumps(sample)})

if __name__ == "__main__":
    main(sys.argv[1:])
