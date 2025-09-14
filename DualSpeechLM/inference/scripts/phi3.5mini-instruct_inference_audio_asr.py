import hydra
import os
import torch

from omegaconf import OmegaConf
import json
from typing import Optional
import transformers
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import webdataset as wds
import pickle
import numpy as np
from transformers import StoppingCriteriaList

from inference.scripts.utils import StoppingCriteriaSub
import argparse

BOI_TOKEN = '<audio>'
EOI_TOKEN = '</audio>'
AUDIO_TOKEN = '<a_{}>'

IMG_FLAG = '<audio>'
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 500
image_id_shift = 32000




def generate(tokenizer, input_tokens, generation_config, model):

    input_ids = tokenizer(input_tokens, add_special_tokens=False, return_tensors='pt').input_ids
    input_ids = input_ids.to("cuda")

    target_vocab_size = 4096
    special_decoder_id = target_vocab_size + 1 
    target_audio_ids = torch.ones(1, dtype=torch.long) * special_decoder_id 
    decoder_start = target_vocab_size + 2  # 4098
    decoder_end = target_vocab_size + 3    # 4099
    target_audio_ids = torch.cat([torch.tensor([decoder_start], dtype=torch.long), target_audio_ids, torch.tensor([decoder_end], dtype=torch.long)], dim=0)
    target_audio_ids = target_audio_ids.unsqueeze(0).cuda() 
    spk_emb = torch.zeros(512, dtype=torch.bfloat16).cuda().unsqueeze(0)

    stop_words_ids = [torch.tensor([tokenizer.eos_token_id]).cuda()]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    generate_ids = model.generate(
        input_ids=input_ids,
        stopping_criteria=stopping_criteria,
        pad_token_id = tokenizer.pad_token_id,
        target_audio_ids = target_audio_ids,
        spk_emb = spk_emb,
        **generation_config
    )
    generate_ids = generate_ids[0][input_ids.shape[1]:]
    
    return generate_ids

def decode_image_text(generate_ids, tokenizer, save_path=None):
    # import pdb; pdb.set_trace()

    boi_list = torch.where(generate_ids == tokenizer(BOI_TOKEN, add_special_tokens=False).input_ids[0])[0]
    eoi_list = torch.where(generate_ids == tokenizer(EOI_TOKEN, add_special_tokens=False).input_ids[0])[0]
    # print('boi_list ', boi_list)
    # print('eoi_list ', eoi_list)
    # assert 1==2
    if len(boi_list) == 0 and len(eoi_list) == 0:
        text_ids = generate_ids
        texts = tokenizer.decode(text_ids, skip_special_tokens=True)
        #print(texts)
        return texts

    else:
        boi_index = boi_list[0]
        eoi_index = eoi_list[0]

        text_ids = generate_ids[:boi_index]
        if len(text_ids) != 0:
            texts = tokenizer.decode(text_ids, skip_special_tokens=True)
            print(texts)
            
        image_ids = (generate_ids[boi_index+1:eoi_index] - image_id_shift).reshape(1,-1)

        images = tokenizer.decode_image(image_ids)

        images[0].save(save_path)


generation_config = {
        'temperature': 0.6,
        'num_beams': 4,
        'max_new_tokens': 300,
        'top_p': 0.9,
        # 'top_k': 20,
        'do_sample': True
    }

s_token = "USER:"
e_token = "ASSISTANT:"
sep = "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cfg-path", type=str, required=True, help='path to configuration file')
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the evaluation data directory")
    parser.add_argument("--asr-gen-file", type=str, required=True, help="Path to the result of asr prediction")
    parser.add_argument("--asr-gt-file", type=str, required=True, help="Path to the ground truth of asr")
    args = parser.parse_args()

    # load tokenizer
    device = "cuda"
    tokenizer_cfg_path = 'configs/tokenizer/speech_phi3.5mini-instruct_tokenizer.yaml'
    tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=device, load_diffusion=True)
    tokenizer.pad_token = tokenizer.unk_token

    # load model
    # model_cfg = OmegaConf.load('inference/configs/llm/phi3.5mini-instruct.yaml')
    model_cfg = OmegaConf.load(args.model_cfg_path)
    model = hydra.utils.instantiate(model_cfg, torch_dtype=torch.bfloat16)
    model = model.eval().to(device)
    print('model ', model)

    # process data and generate results
    data_dir = args.data_dir
    all_dataset_path = [f for f in os.listdir(data_dir) if f.endswith('.tar')]
    cnt = 0
    with open(args.asr_gen_file, 'w') as f_out, open(args.asr_gt_file, 'w') as f_gt:
        for dataset_path in all_dataset_path:
            dataset_path = os.path.join(data_dir, dataset_path)
            dataset = wds.WebDataset(dataset_path)
            for sample in dataset:
                try:
                    key = sample['__key__']
                    pkl_data = sample['pkl']
                    data = pickle.loads(pkl_data)
                    audio_ids = data['audio_ids']
                    current_tokenizer = data['metadata']['tokenizer']

                    text = data['text']
                    metadata = data['metadata']

                    img_ids = np.array(audio_ids)
                    audio_tokens = BOI_TOKEN + ' ' + ' '.join([AUDIO_TOKEN.format(int(item)) for item in audio_ids]) + ' ' + EOI_TOKEN
                    question = "Recognize the speech and give me the transcription. "

                    input_tokens = tokenizer.bos_token + question + s_token + " " + audio_tokens + sep + e_token + ' '
                    generate_ids = generate(tokenizer, input_tokens, generation_config, model)
                    text_gen = decode_image_text(generate_ids, tokenizer)
                    print(text_gen)
                    print(text)
                    cnt += 1
                    file_name = metadata['id']
                    f_out.write(file_name+'\t'+text_gen+'\n')
                    f_gt.write(file_name+'\t'+text+'\n')
                except Exception as e:
                    print('Error: ', e)
                    continue

    print('cnt ', cnt)

