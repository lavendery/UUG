import torchdata.datapipes as dp
import functools
import numpy as np
import torch
import pickle
from braceexpand import braceexpand
import hydra
from torch.nn.utils.rnn import pad_sequence
import os

task2id = {
    'asr': 1,   
    'tts': 2, 
    'vc': 3, 
    't2st': 4, 
    'sc': 5,
    'sqa': 6,
    's2tt': 7,
    'ser': 8,
}

BOI_TOKEN = '<audio>'
EOI_TOKEN = '</audio>'
AUDIO_TOKEN = '<a_{}>'

s_token = "USER:"
e_token = "ASSISTANT:"

def decode_ASR_pair_for_llm(item,
                            tokenizer=None,
                            max_length=128,
                            caption_prompt=None,
                            reverse_ratio=0.5,
                            mask_left_label=False,
                            use_caption_in_metadata=False,
                            caption_key_in_metadata='',
                            target_vocab_size=None):
    key, value = item
    sep = '\n'
    if key.endswith(".pkl"):
        sample = pickle.load(value)
        audio_ids = sample['audio_ids']
        text = sample['text']
        audio_tokens = BOI_TOKEN + ' ' + ' '.join([AUDIO_TOKEN.format(int(item)) for item in audio_ids]) + ' ' + EOI_TOKEN 
        system_message = 'Recognize the speech and give me the transcription. '
        question = system_message + s_token + " " + audio_tokens + sep + e_token + ' '
        question_ids = tokenizer.encode(question, add_special_tokens=False)
        answer_ids = tokenizer.encode(text, add_special_tokens=False)
        # print('answer_ids ', len(answer_ids), answer_ids)
        labels = [-100] * len(question_ids) + answer_ids
        input_ids = question_ids + answer_ids
        input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] + labels + [tokenizer.eos_token_id]
        if len(input_ids) >= max_length:
            return key, None
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        attention_mask_question = attention_mask
        attention_mask_answer = attention_mask
        labels = torch.tensor(labels, dtype=torch.long)

        special_decoder_id = target_vocab_size + 1 
        target_audio_ids = torch.ones(1, dtype=torch.long) * special_decoder_id 
        decoder_start = target_vocab_size + 2  # 4098
        decoder_end = target_vocab_size + 3    # 4099
        target_audio_ids = torch.cat([torch.tensor([decoder_start], dtype=torch.long), target_audio_ids, torch.tensor([decoder_end], dtype=torch.long)], dim=0)

        spk_emb = torch.zeros(512, dtype=torch.float32)
        task = torch.tensor([task2id['asr']], dtype=torch.long)
        return key, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'attention_mask_question': attention_mask_question,
            'attention_mask_answer': attention_mask_answer,
            'labels': labels,
            'target_audio_ids': target_audio_ids,
            'spk_emb': spk_emb,
            'task_id': task,
        }
    else:
        return key, None

def decode_TTS_pair_for_llm(item,
                            tokenizer=None,
                            max_length=128,
                            caption_prompt=None,
                            reverse_ratio=0.5,
                            mask_left_label=False,
                            use_caption_in_metadata=False,
                            caption_key_in_metadata='',
                            target_vocab_size=None,
                            train_LibriTTS_R_npy_dir=None,
                            train_libriheavy_large_npy_dir=None):
    key, value = item
    sep = '\n'
    if key.endswith(".pkl"):
        sample = pickle.load(value)
        audio_ids = sample['audio_ids']
        text = sample['text']

        audio_tokens = BOI_TOKEN + ' ' + ' '.join([AUDIO_TOKEN.format(int(item)) for item in audio_ids]) + ' ' + EOI_TOKEN # prepare the sequence: '<audio><a_1><a_2><a_3>...</audio>'
        system_message = 'Please read this sentence out loud. '
        question = system_message + s_token + " " + text + sep + e_token + ' '
        question_ids = tokenizer.encode(question, add_special_tokens=False)
        answer_ids = tokenizer.encode(audio_tokens, add_special_tokens=False)
        labels = [-100] * len(question_ids) + answer_ids
        input_ids = question_ids + answer_ids
        input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] + labels + [tokenizer.eos_token_id]
        if len(input_ids) >= max_length:
            return key, None
        attention_mask_question = [1]*(1 + len(question_ids)) + [0]*(len(answer_ids) + 1)
        attention_mask_answer = [0]*(1 + len(question_ids)) + [1]*(len(answer_ids) + 1)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        attention_mask_question = torch.tensor(attention_mask_question, dtype=torch.long)
        attention_mask_answer = torch.tensor(attention_mask_answer, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        target_audio_ids = sample['target_audio_ids']
        target_audio_ids = torch.tensor(target_audio_ids, dtype=torch.long)
        decoder_start = target_vocab_size + 2  # 4098
        decoder_end = target_vocab_size + 3    # 4099
        target_audio_ids = torch.cat([torch.tensor([decoder_start], dtype=torch.long), target_audio_ids, torch.tensor([decoder_end], dtype=torch.long)], dim=0)

        file_id = sample['metadata']['id']
        if 'large' in file_id:
            npy_dir = train_libriheavy_large_npy_dir
        else:
            npy_dir = train_LibriTTS_R_npy_dir
        npy_path = os.path.join(npy_dir, file_id + '.npy')
        if not os.path.exists(npy_path):
            print('npy file not exists: ', npy_path)
            return key, None
        spk_emb = np.load(npy_path)
        spk_emb = torch.tensor(spk_emb, dtype=torch.float32)
        if spk_emb.size(0) != 512:
            print('spk_emb size is not 512: ', spk_emb.size(0))
            return key, None
        
        task = torch.tensor([task2id['tts']], dtype=torch.long)
        return key, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'attention_mask_question': attention_mask_question,
            'attention_mask_answer': attention_mask_answer,
            'labels': labels,
            'target_audio_ids': target_audio_ids,
            'spk_emb': spk_emb,
            'task_id': task,
        }
    else:
        return key, None

def decode_VC_pair_for_llm(item,
                            tokenizer=None,
                            max_length=128,
                            caption_prompt=None,
                            reverse_ratio=0.5,
                            target_vocab_size=None,
                            train_libriheavy_large_npy_dir=None):
    key, value = item
    sep = '\n'
    if key.endswith(".pkl"):
        sample = pickle.load(value)
        audio_ids = sample['audio_ids']
        current_tokenizer = sample['metadata']['tokenizer']
        text = sample['text']
        audio_tokens = BOI_TOKEN + ' ' + ' '.join([AUDIO_TOKEN.format(int(item)) for item in audio_ids]) + ' ' + EOI_TOKEN # prepare the sequence: '<audio><a_1><a_2><a_3>...</audio>'
        system_message = "Without altering the spoken content, transform the speaker's voice in this speech to match the target voice. "
        question = system_message + s_token + " " + audio_tokens + sep + e_token + ' '
        question_ids = tokenizer.encode(question, add_special_tokens=False)
        answer_ids = tokenizer.encode(audio_tokens, add_special_tokens=False)
        labels = [-100] * len(question_ids) + answer_ids
        input_ids = question_ids + answer_ids
        input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] + labels + [tokenizer.eos_token_id]

        if len(input_ids) >= max_length:
            return key, None
        
        attention_mask_question = [1]*(1 + len(question_ids)) + [0]*(len(answer_ids) + 1)
        attention_mask_answer = [0]*(1 + len(question_ids)) + [1]*(len(answer_ids) + 1)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        attention_mask_question = torch.tensor(attention_mask_question, dtype=torch.long)
        attention_mask_answer = torch.tensor(attention_mask_answer, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        target_audio_ids = sample['target_audio_ids']
        target_audio_ids = torch.tensor(target_audio_ids, dtype=torch.long)
        decoder_start = target_vocab_size + 2  # 4098
        decoder_end = target_vocab_size + 3    # 4099
        target_audio_ids = torch.cat([torch.tensor([decoder_start], dtype=torch.long), target_audio_ids, torch.tensor([decoder_end], dtype=torch.long)], dim=0)

        file_id = sample['metadata']['id']
        npy_dir = train_libriheavy_large_npy_dir
        npy_path = os.path.join(npy_dir, file_id + '.npy')
        if not os.path.exists(npy_path):
            print('npy file not exists: ', npy_path)
            return key, None
        spk_emb = np.load(npy_path)
        spk_emb = torch.tensor(spk_emb, dtype=torch.float32)
        if spk_emb.size(0) != 512:
            print('spk_emb size is not 512: ', spk_emb.size(0))
            return key, None

        task = torch.tensor([task2id['vc']], dtype=torch.long)
        return key, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'attention_mask_question': attention_mask_question,
            'attention_mask_answer': attention_mask_answer,
            'labels': labels,
            'target_audio_ids': target_audio_ids,
            'spk_emb': spk_emb,
            'task_id': task,
        }
    else:
        return key, None

def decode_T2ST_pair_for_llm(item,
                            tokenizer=None,
                            max_length=128,
                            caption_prompt=None,
                            reverse_ratio=0.5,
                            target_vocab_size=None,
                            npy_dir=None):
    key, value = item
    sep = '\n'
    if key.endswith(".pkl"):
        sample = pickle.load(value)
        audio_ids = sample['audio_ids']
        current_tokenizer = sample['metadata']['tokenizer']
        input_text = sample['input_text']
        system_message = "Please translate the " + sample['metadata']['source_language'] + " text into " + sample['metadata']['target_language'] + " speech. "
        question = system_message + s_token + " " + input_text + sep + e_token + ' '
        
        question_ids = tokenizer.encode(question, add_special_tokens=False)
        answer_audio_ids = sample['answer_ids']
        answer_audio_tokens = BOI_TOKEN + ' ' + ' '.join([AUDIO_TOKEN.format(int(item)) for item in answer_audio_ids]) + ' ' + EOI_TOKEN # prepare the sequence: '<audio><a_1><a_2><a_3>...</audio>'
        answer_ids = tokenizer.encode(answer_audio_tokens, add_special_tokens=False)
        # print('answer_ids ', answer_ids)
        labels = [-100] * len(question_ids) + answer_ids
        input_ids = question_ids + answer_ids
        input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] + labels + [tokenizer.eos_token_id]

        if len(input_ids) >= max_length:
            return key, None

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        attention_mask_question = attention_mask
        attention_mask_answer = attention_mask
        labels = torch.tensor(labels, dtype=torch.long)

        target_audio_ids = sample['target_audio_ids']
        target_audio_ids = torch.tensor(target_audio_ids, dtype=torch.long)
        decoder_start = target_vocab_size + 2  # 4098
        decoder_end = target_vocab_size + 3    # 4099
        target_audio_ids = torch.cat([torch.tensor([decoder_start], dtype=torch.long), target_audio_ids, torch.tensor([decoder_end], dtype=torch.long)], dim=0)

        file_id = sample['metadata']['id']
        npy_path = os.path.join(npy_dir, file_id + '.npy')
        if not os.path.exists(npy_path):
            print('npy file not exists: ', npy_path)
            return key, None
        spk_emb = np.load(npy_path)
        spk_emb = torch.tensor(spk_emb, dtype=torch.float32)
        if spk_emb.size(0) != 512:
            print('spk_emb size is not 512: ', spk_emb.size(0))
            return key, None

        task = torch.tensor([task2id['t2st']], dtype=torch.long)
        return key, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'attention_mask_question': attention_mask_question,
            'attention_mask_answer': attention_mask_answer,
            'labels': labels,
            'target_audio_ids': target_audio_ids,
            'spk_emb': spk_emb,
            'task_id': task,
        }
    else:
        return key, None

def decode_SC_pair_for_llm(item,
                            tokenizer=None,
                            max_length=128,
                            caption_prompt=None,
                            reverse_ratio=0.5,
                            target_vocab_size=None,
                            npy_dir=None):
    key, value = item
    sep = '\n'
    if key.endswith(".pkl"):
        sample = pickle.load(value)
        audio_ids = sample['audio_ids']
        speech_question_ids = sample['question_ids']

        current_tokenizer = sample['metadata']['tokenizer']
        audio_tokens = BOI_TOKEN + ' ' + ' '.join([AUDIO_TOKEN.format(int(item)) for item in audio_ids]) + ' ' + EOI_TOKEN # prepare the sequence: '<audio><a_1><a_2><a_3>...</audio>'
        speech_question_tokens = BOI_TOKEN + ' ' + ' '.join([AUDIO_TOKEN.format(int(item)) for item in speech_question_ids]) + ' ' + EOI_TOKEN # prepare the sequence: '<audio><a_1><a_2><a_3>...</audio>'
        system_message = "Please listen to the speech content and provide a spoken answer to the question. "
        question = system_message + s_token + " The speech content is: " + audio_tokens + sep + "The question is: " + speech_question_tokens + sep + e_token + ' '
        question_ids = tokenizer.encode(question, add_special_tokens=False)
        answer_audio_ids = sample['answer_ids']
        answer_audio_tokens = BOI_TOKEN + ' ' + ' '.join([AUDIO_TOKEN.format(int(item)) for item in answer_audio_ids]) + ' ' + EOI_TOKEN # prepare the sequence: '<audio><a_1><a_2><a_3>...</audio>'
        answer_ids = tokenizer.encode(answer_audio_tokens, add_special_tokens=False)
        labels = [-100] * len(question_ids) + answer_ids
        input_ids = question_ids + answer_ids
        input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] + labels + [tokenizer.eos_token_id]
        if len(input_ids) >= max_length:
            return key, None
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        attention_mask_question = attention_mask
        attention_mask_answer = attention_mask
        labels = torch.tensor(labels, dtype=torch.long)

        target_audio_ids = sample['target_audio_ids']
        target_audio_ids = torch.tensor(target_audio_ids, dtype=torch.long)
        decoder_start = target_vocab_size + 2  # 4098
        decoder_end = target_vocab_size + 3    # 4099
        target_audio_ids = torch.cat([torch.tensor([decoder_start], dtype=torch.long), target_audio_ids, torch.tensor([decoder_end], dtype=torch.long)], dim=0)

        file_id = sample['metadata']['id']
        npy_path = os.path.join(npy_dir, file_id + '.npy')
        if not os.path.exists(npy_path):
            print('npy file not exists: ', npy_path)
            return key, None
        spk_emb = np.load(npy_path)
        spk_emb = torch.tensor(spk_emb, dtype=torch.float32)
        if spk_emb.size(0) != 512:
            print('spk_emb size is not 512: ', spk_emb.size(0))
            return key, None

        task = torch.tensor([task2id['sc']], dtype=torch.long)
        return key, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'attention_mask_question': attention_mask_question,
            'attention_mask_answer': attention_mask_answer,
            'labels': labels,
            'target_audio_ids': target_audio_ids,
            'spk_emb': spk_emb,
            'task_id': task,
        }
    else:
        return key, None

def decode_SQA_pair_for_llm(item,
                            tokenizer=None,
                            max_length=128,
                            caption_prompt=None,
                            reverse_ratio=0.5,
                            target_vocab_size=None):
    key, value = item
    sep = '\n'
    if key.endswith(".pkl"):
        sample = pickle.load(value)
        audio_ids = sample['audio_ids']
        text_question = sample['question']

        audio_tokens = BOI_TOKEN + ' ' + ' '.join([AUDIO_TOKEN.format(int(item)) for item in audio_ids]) + ' ' + EOI_TOKEN # prepare the sequence: '<audio><a_1><a_2><a_3>...</audio>'
        system_message = "Based on the content, provide a text-based answer to the question. "
        question = system_message + s_token + " The content is: " + audio_tokens + sep + "The question is: " + text_question + sep + e_token + ' '
        question_ids = tokenizer.encode(question, add_special_tokens=False)
        text_answer = sample['text']
        answer_ids = tokenizer.encode(text_answer, add_special_tokens=False)
        # print('answer_ids ', answer_ids)
        labels = [-100] * len(question_ids) + answer_ids
        input_ids = question_ids + answer_ids
        input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] + labels + [tokenizer.eos_token_id]
        if len(input_ids) >= max_length:
            return key, None

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        attention_mask_question = attention_mask
        attention_mask_answer = attention_mask
        labels = torch.tensor(labels, dtype=torch.long)

        special_decoder_id = target_vocab_size + 1 
        target_audio_ids = torch.ones(1, dtype=torch.long) * special_decoder_id 
        decoder_start = target_vocab_size + 2  # 4098
        decoder_end = target_vocab_size + 3    # 4099
        target_audio_ids = torch.cat([torch.tensor([decoder_start], dtype=torch.long), target_audio_ids, torch.tensor([decoder_end], dtype=torch.long)], dim=0)

        spk_emb = torch.zeros(512, dtype=torch.float32)
        task = torch.tensor([task2id['sqa']], dtype=torch.long)
        return key, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'attention_mask_question': attention_mask_question,
            'attention_mask_answer': attention_mask_answer,
            'labels': labels,
            'target_audio_ids': target_audio_ids,
            'spk_emb': spk_emb,
            'task_id': task,
            # 'text': text,
        }
    else:
        return key, None

def decode_S2TT_pair_for_llm(item,
                            tokenizer=None,
                            max_length=128,
                            caption_prompt=None,
                            reverse_ratio=0.5,
                            mask_left_label=False,
                            use_caption_in_metadata=False,
                            caption_key_in_metadata='',
                            target_vocab_size=None):
    key, value = item
    sep = '\n'
    if key.endswith(".pkl"):
        sample = pickle.load(value)
        audio_ids = sample['audio_ids']
        text = sample['text']

        audio_tokens = BOI_TOKEN + ' ' + ' '.join([AUDIO_TOKEN.format(int(item)) for item in audio_ids]) + ' ' + EOI_TOKEN # prepare the sequence: '<audio><a_1><a_2><a_3>...</audio>'
        system_message = "Please translate the " + sample['metadata']['source_language'] + " speech into " + sample['metadata']['target_language'] + " text transcription. "
        question = system_message + s_token + " " + audio_tokens + sep + e_token + ' '
        question_ids = tokenizer.encode(question, add_special_tokens=False)

        answer_ids = tokenizer.encode(text, add_special_tokens=False)
        # print('answer_ids ', len(answer_ids), answer_ids)
        labels = [-100] * len(question_ids) + answer_ids
        input_ids = question_ids + answer_ids
        input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] + labels + [tokenizer.eos_token_id]

        if len(input_ids) >= max_length:
            return key, None
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        attention_mask_question = attention_mask
        attention_mask_answer = attention_mask
        labels = torch.tensor(labels, dtype=torch.long)

        special_decoder_id = target_vocab_size + 1 
        target_audio_ids = torch.ones(1, dtype=torch.long) * special_decoder_id 
        decoder_start = target_vocab_size + 2  # 4098
        decoder_end = target_vocab_size + 3    # 4099
        target_audio_ids = torch.cat([torch.tensor([decoder_start], dtype=torch.long), target_audio_ids, torch.tensor([decoder_end], dtype=torch.long)], dim=0)

        spk_emb = torch.zeros(512, dtype=torch.float32)
        task = torch.tensor([task2id['s2tt']], dtype=torch.long)
        return key, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'attention_mask_question': attention_mask_question,
            'attention_mask_answer': attention_mask_answer,
            'labels': labels,
            'target_audio_ids': target_audio_ids,
            'spk_emb': spk_emb,
            'task_id': task,
            # 'text': text,
        }
    else:
        return key, None

def decode_SER_pair_for_llm(item,
                            tokenizer=None,
                            max_length=128,
                            caption_prompt=None,
                            reverse_ratio=0.5,
                            mask_left_label=False,
                            use_caption_in_metadata=False,
                            caption_key_in_metadata='',
                            target_vocab_size=None):
    key, value = item
    sep = '\n'
    if key.endswith(".pkl"):
        sample = pickle.load(value)
        audio_ids = sample['audio_ids']
        text = sample['text']

        audio_tokens = BOI_TOKEN + ' ' + ' '.join([AUDIO_TOKEN.format(int(item)) for item in audio_ids]) + ' ' + EOI_TOKEN # prepare the sequence: '<audio><a_1><a_2><a_3>...</audio>'
        system_message = "Please describe the emotion of the speaker. "
        question = system_message + s_token + " " + audio_tokens + sep + e_token + ' '
        question_ids = tokenizer.encode(question, add_special_tokens=False)
        answer_ids = tokenizer.encode(text, add_special_tokens=False)
        # print('answer_ids ', len(answer_ids), answer_ids)
        labels = [-100] * len(question_ids) + answer_ids
        input_ids = question_ids + answer_ids
        input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] + labels + [tokenizer.eos_token_id]
        if len(input_ids) >= max_length:
            return key, None
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        attention_mask_question = attention_mask
        attention_mask_answer = attention_mask
        labels = torch.tensor(labels, dtype=torch.long)

        special_decoder_id = target_vocab_size + 1 
        target_audio_ids = torch.ones(1, dtype=torch.long) * special_decoder_id 
        decoder_start = target_vocab_size + 2  # 4098
        decoder_end = target_vocab_size + 3    # 4099
        target_audio_ids = torch.cat([torch.tensor([decoder_start], dtype=torch.long), target_audio_ids, torch.tensor([decoder_end], dtype=torch.long)], dim=0)

        spk_emb = torch.zeros(512, dtype=torch.float32)
        task = torch.tensor([task2id['ser']], dtype=torch.long)
        return key, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'attention_mask_question': attention_mask_question,
            'attention_mask_answer': attention_mask_answer,
            'labels': labels,
            'target_audio_ids': target_audio_ids,
            'spk_emb': spk_emb,
            'task_id': task,
        }
    else:
        return key, None

def unwarp_data(item):
    unwarpped = {}
    for key, value in item.items():
        if isinstance(value, dict):
            unwarpped.update(value)
        elif value is not None:
            unwarpped[key] = value
    return unwarpped


def filter_data_for_llm(item):
    if 'input_ids' in item:
        return True
    else:
        print('A sample has been filtered out.')
        return False


def filter_data_with_image_text(item):
    if 'pixel_values' in item and 'input_ids' in item:
        return True
    else:
        print('A sample has been filtered out.')
        return False

def custom_collate_fn(batch_samples, tokenizer, target_vocab_size):
    collated_batch = {}
    
    for key in batch_samples[0].keys():
        samples = [sample[key] for sample in batch_samples]
        if key in ['input_ids', 'attention_mask', 'labels', 'spk_emb', 'attention_mask_question', 'attention_mask_answer']:
            if key == 'input_ids':
                pad_value = tokenizer.pad_token_id
            elif key == 'labels':
                pad_value = -100
            elif key == 'attention_mask' or key == 'attention_mask_question' or key == 'attention_mask_answer':
                pad_value = 0
            elif key == 'spk_emb':
                pad_value = 0

            padded = pad_sequence(
                samples,
                batch_first=True,
                padding_value=pad_value
            )
        elif key == 'target_audio_ids':
            padded = pad_sequence(
                samples,
                batch_first=True,
                padding_value=target_vocab_size  
            )
        elif key == '__key__':
            continue
        else:
            padded = torch.stack(samples)
        
        collated_batch[key] = padded
    
    return collated_batch

def build_ASR_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.5,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None,
                                 target_vocab_size=None):
    """
    datapipe of ASR dataset (such as LibriSpeech, MLS...) with webdataset format
    """
    decode_partial = functools.partial(decode_ASR_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio,
                                       target_vocab_size=target_vocab_size)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)

        collate_fn_with_args = functools.partial(
            custom_collate_fn,
            tokenizer=tokenizer,
            target_vocab_size=target_vocab_size
        )
        datapipe = datapipe.collate(collate_fn=collate_fn_with_args)
    return datapipe

def build_TTS_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.5,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None,
                                 target_vocab_size=None,
                                 train_LibriTTS_R_npy_dir=None,
                                 train_libriheavy_large_npy_dir=None):
    """
    datapipe of ASR dataset (such as LibriSpeech, MLS...) with webdataset format
    """
    decode_partial = functools.partial(decode_TTS_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio,
                                       target_vocab_size=target_vocab_size,
                                       train_LibriTTS_R_npy_dir=train_LibriTTS_R_npy_dir,
                                       train_libriheavy_large_npy_dir=train_libriheavy_large_npy_dir)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)

        collate_fn_with_args = functools.partial(
            custom_collate_fn,
            tokenizer=tokenizer,
            target_vocab_size=target_vocab_size
        )
        datapipe = datapipe.collate(collate_fn=collate_fn_with_args)
    return datapipe

def build_VC_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.5,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None,
                                 target_vocab_size=None,
                                 train_libriheavy_large_npy_dir=None):
    """
    datapipe of ASR dataset (such as LibriSpeech, MLS...) with webdataset format
    """
    decode_partial = functools.partial(decode_VC_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio,
                                       target_vocab_size=target_vocab_size,
                                       train_libriheavy_large_npy_dir=train_libriheavy_large_npy_dir)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)

        collate_fn_with_args = functools.partial(
            custom_collate_fn,
            tokenizer=tokenizer,
            target_vocab_size=target_vocab_size
        )
        datapipe = datapipe.collate(collate_fn=collate_fn_with_args)
    return datapipe

def build_T2ST_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.5,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None,
                                 target_vocab_size=None,
                                 npy_dir=None):
    """
    datapipe of ASR dataset (such as LibriSpeech, MLS...) with webdataset format
    """
    decode_partial = functools.partial(decode_T2ST_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio,
                                       target_vocab_size=target_vocab_size,
                                       npy_dir=npy_dir)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)

        collate_fn_with_args = functools.partial(
            custom_collate_fn,
            tokenizer=tokenizer,
            target_vocab_size=target_vocab_size
        )
        datapipe = datapipe.collate(collate_fn=collate_fn_with_args)
    return datapipe

def build_SC_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.5,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None,
                                 target_vocab_size=None,
                                 npy_dir=None):
    """
    datapipe of ASR dataset (such as LibriSpeech, MLS...) with webdataset format
    """
    decode_partial = functools.partial(decode_SC_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio,
                                       target_vocab_size=target_vocab_size,
                                       npy_dir=npy_dir)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)

        collate_fn_with_args = functools.partial(
            custom_collate_fn,
            tokenizer=tokenizer,
            target_vocab_size=target_vocab_size
        )
        datapipe = datapipe.collate(collate_fn=collate_fn_with_args)
    return datapipe

def build_SQA_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.5,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None,
                                 target_vocab_size=None):
    """
    datapipe of ASR dataset (such as LibriSpeech, MLS...) with webdataset format
    """
    decode_partial = functools.partial(decode_SQA_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio,
                                       target_vocab_size=target_vocab_size)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)

        collate_fn_with_args = functools.partial(
            custom_collate_fn,
            tokenizer=tokenizer,
            target_vocab_size=target_vocab_size
        )
        datapipe = datapipe.collate(collate_fn=collate_fn_with_args)
    return datapipe

def build_S2TT_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.5,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None,
                                 target_vocab_size=None):
    """
    datapipe of ASR dataset (such as LibriSpeech, MLS...) with webdataset format
    """
    decode_partial = functools.partial(decode_S2TT_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio,
                                       target_vocab_size=target_vocab_size)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)

        collate_fn_with_args = functools.partial(
            custom_collate_fn,
            tokenizer=tokenizer,
            target_vocab_size=target_vocab_size
        )
        datapipe = datapipe.collate(collate_fn=collate_fn_with_args)
    return datapipe

def build_SER_datapipes_for_llm(data_dir,
                                 tokenizer=None,
                                 max_length=512,
                                 reverse_ratio=0.5,
                                 recursive=True,
                                 batch_size=None,
                                 caption_prompt=None,
                                 cycle_count=None,
                                 target_vocab_size=None):
    """
    datapipe of ASR dataset (such as LibriSpeech, MLS...) with webdataset format
    """
    decode_partial = functools.partial(decode_SER_pair_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       caption_prompt=caption_prompt,
                                       reverse_ratio=reverse_ratio,
                                       target_vocab_size=target_vocab_size)
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=recursive)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_for_llm)
    datapipe = datapipe.shuffle(buffer_size=4096)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)

        collate_fn_with_args = functools.partial(
            custom_collate_fn,
            tokenizer=tokenizer,
            target_vocab_size=target_vocab_size
        )
        datapipe = datapipe.collate(collate_fn=collate_fn_with_args)
    return datapipe

def build_multi_datapipes(datapipes, tokenizer=None, concat_type='sample', sample_weights=None):
    assert concat_type in ['concat', 'mux_longest', 'sample']
    if sample_weights is None:
        sample_weights = [1] * len(datapipes)
    else:
        assert len(sample_weights) == len(datapipes)

    datapipes = [hydra.utils.instantiate(datapipe, tokenizer=tokenizer) for datapipe in datapipes]

    if concat_type == 'concat':
        datapipe = dp.iter.Concater(*datapipes)
    elif concat_type == 'mux_longest':
        datapipe = dp.iter.MultiplexerLongest(*datapipes)
    elif concat_type == 'sample':
        datasets_to_weights_dict = {}
        for dataset, sample_weight in zip(datapipes, sample_weights):
            datasets_to_weights_dict[dataset] = sample_weight
        datapipe = dp.iter.SampleMultiplexer(datasets_to_weights_dict)

    else:
        raise NotImplementedError

    return datapipe

