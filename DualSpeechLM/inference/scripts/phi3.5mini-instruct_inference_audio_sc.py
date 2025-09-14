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
import torchaudio

from inference.scripts.utils import StoppingCriteriaSub
from src.tokenizer.speech_llama_tokenizer import MyWavTokenizer
import argparse

BOI_TOKEN = '<audio>'
EOI_TOKEN = '</audio>'
AUDIO_TOKEN = '<a_{}>'

IMG_FLAG = '<audio>'
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 500
image_id_shift = 32000


def generate(tokenizer, question, generation_config, model, gt_target_audio_ids, audio_tokens, gt_semantic_tokens, spk_emb=None):

    input_ids = tokenizer(question, add_special_tokens=False, return_tensors='pt').input_ids
    # input_ids2 = tokenizer.encode(question, add_special_tokens=False)
    input_ids = input_ids.to("cuda")

    stop_words_ids = [torch.tensor([tokenizer.eos_token_id]).cuda(), torch.tensor([32012]).cuda()] 
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    generate_ids = model.generate(
        input_ids=input_ids,
        stopping_criteria=stopping_criteria,
        pad_token_id = tokenizer.pad_token_id,
        target_audio_ids = None,
        output_hidden_states = True,
        return_dict_in_generate = True,
        **generation_config
    )
    lmdecoder_input_ids = generate_ids[0] 

    if len(lmdecoder_input_ids.shape) == 1:
        lmdecoder_input_ids = lmdecoder_input_ids.unsqueeze(0)
    attention_mask = [1] * lmdecoder_input_ids.shape[1]
    attention_mask = torch.tensor(attention_mask).unsqueeze(0).cuda()
    
    # attention_mask_question
    attention_mask_question = attention_mask.clone()
    attention_mask_question[:, input_ids.shape[1]:] = 0
    attention_mask_answer = attention_mask.clone()
    attention_mask_answer[:, :input_ids.shape[1]] = 0
    
    target_vocab_size = 4096
    decoder_start = target_vocab_size + 2  # 4098
    decoder_end = target_vocab_size + 3    # 4099
    target_audio_ids = torch.cat([torch.tensor([decoder_start], dtype=torch.long), torch.tensor([decoder_end], dtype=torch.long)], dim=0)
    target_audio_ids = target_audio_ids.unsqueeze(0).cuda() 

    gt_semantic_labels = None
    if lmdecoder_input_ids.shape[1] - len(gt_semantic_tokens) - 2 - input_ids.shape[1] >= 0:
        gt_semantic_tokens_label = [-100] * (input_ids.shape[1]) + [32011] + gt_semantic_tokens + [32012] + [-100] * (lmdecoder_input_ids.shape[1] - len(gt_semantic_tokens) - 2 - input_ids.shape[1])
        gt_semantic_labels = torch.tensor(gt_semantic_tokens_label).unsqueeze(0).cuda()
    lm_decodec_id = model.lm_decoder_generate(input_ids=lmdecoder_input_ids, attention_mask=attention_mask, attention_mask_question=attention_mask_question, attention_mask_answer=attention_mask_answer, target_audio_ids = target_audio_ids, gt_semantic_labels=gt_semantic_labels, spk_emb=spk_emb)
    
    return lm_decodec_id

def decode_image_text(generate_ids, tokenizer, save_path=None):
    audio_ids = generate_ids[generate_ids < 4096].unsqueeze(0)
    if audio_ids.shape[1] == 0:
        print('audio_ids is empty')
        return

    audio_out = tokenizer.decode_audio(audio_ids)

    torchaudio.save(save_path, audio_out.cpu(), sample_rate=24000, encoding='PCM_S', bits_per_sample=16)

generation_config = {
        'temperature': 0.6,
        'num_beams': 1,
        'max_new_tokens': 400, 
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
    parser.add_argument("--npy_dir", type=str, required=True, help="Path to the speaker embedding file directory")
    parser.add_argument("--sc-gen-dir", type=str, required=True, help="Path to the result of asr prediction")
    args = parser.parse_args()
    os.makedirs(args.sc_gen_dir, exist_ok=True)

    # load tokenizer
    device = "cuda"
    tokenizer_cfg_path = 'configs/tokenizer/speech_phi3.5mini-instruct_tokenizer.yaml'
    tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=device, load_diffusion=True)
    tokenizer.pad_token = tokenizer.unk_token
    print('tokenizer ', len(tokenizer))
    mywav_tokenizer = MyWavTokenizer()
    mywav_tokenizer._audio_tokenizer.to(device)

    model_cfg = OmegaConf.load(args.model_cfg_path)
    model = hydra.utils.instantiate(model_cfg, torch_dtype=torch.bfloat16)
    model = model.eval().to(device)
    print('model ', model)

    # process data and generate results
    data_dir = args.data_dir
    all_dataset_path = [f for f in os.listdir(data_dir) if f.endswith('.tar')]
    cnt = 0
    for dataset_path in all_dataset_path:
        dataset_path = os.path.join(data_dir, dataset_path)
        dataset = wds.WebDataset(dataset_path)
        for sample in dataset:
            try:
                key = sample['__key__']
                pkl_data = sample['pkl']
                
                data = pickle.loads(pkl_data)
                audio_ids = data['audio_ids']
                speech_question_ids = data['question_ids']

                gt_target_audio_ids = data['target_audio_ids']
                text = data['answer_text']
                metadata = data['metadata']

                audio_tokens = BOI_TOKEN + ' ' + ' '.join([AUDIO_TOKEN.format(int(item)) for item in audio_ids]) + ' ' + EOI_TOKEN
                speech_question_tokens = BOI_TOKEN + ' ' + ' '.join([AUDIO_TOKEN.format(int(item)) for item in speech_question_ids]) + ' ' + EOI_TOKEN
                system_message = "Please listen to the speech content and provide a spoken answer to the question. "

                question = tokenizer.bos_token + system_message + s_token + " The speech content is: " + audio_tokens + sep + "The question is: " + speech_question_tokens + sep + e_token + ' '
                gt_semantic_ids = data['answer_ids']
                gt_semantic_tokens = [item + 32013 for item in gt_semantic_ids]
                file_id = data['metadata']['id']
                
                npy_path = os.path.join(args.npy_dir, file_id + '.npy')
                if not os.path.exists(npy_path):
                    print('npy file not exists: ', npy_path)
                    continue
                spk_emb = np.load(npy_path)
                spk_emb = torch.tensor(spk_emb, dtype=torch.bfloat16).unsqueeze(0).cuda()
                generate_ids = generate(tokenizer, question, generation_config, model, gt_target_audio_ids, audio_tokens, gt_semantic_tokens, spk_emb)
                file_name = metadata['id']
                save_path = os.path.join(args.sc_gen_dir, file_name + '.wav')
                decode_image_text(generate_ids, mywav_tokenizer, save_path)
                print(file_id, text) 
                cnt += 1
            except Exception as e:
                print(f"Error processing sample: {e}")
                torch.cuda.empty_cache()
                continue

    print('cnt ', cnt)

