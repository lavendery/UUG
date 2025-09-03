# Benchmark evaluation for MLLM undersanding
# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os 
import torch
from transformers import WhisperFeatureExtractor

from config import Config
from models.USTokenizer import USTokenizer
from utils import prepare_one_sample
# from infer_visual_mel import plot_mel_spectrogram

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--test_json_path", type=str, help='the json path for evaluation')
parser.add_argument("--result_file", type=str, help='the path for result')
parser.add_argument("--gt_file", type=str, help='the path for result')
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)

args = parser.parse_args()
cfg = Config(args)

prompt_dict = {}
raw_prompts = json.load(open(cfg.config.model.test_prompt_path, "r"))
for task in raw_prompts.keys():
    prompt_dict[task] = cfg.config.model.prompt_template.format(raw_prompts[task].strip())
model = USTokenizer.from_config(cfg.config.model)
model.to(args.device)
model.eval()

wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)
test_set = json.load(open(args.test_json_path, "r"))["annotation"]
f_out = open(args.result_file, 'w')
f_gt = open(args.gt_file, 'w')
for test_itm in test_set:
    wav_path = test_itm['path']
    bs_name = os.path.basename(wav_path).replace('.wav', '')
    samples = prepare_one_sample(wav_path, wav_processor)

    if prompt_dict:
        prompt = prompt_dict[test_itm["task"]]
    if "Q" in test_itm and '{}' in prompt:
        prompt = prompt.format(test_itm["Q"])
    prompt = [prompt]
    with torch.cuda.amp.autocast(dtype=torch.float16):
        print('gt ', test_itm['text'])
        text, commitment_loss, recon_loss, perplexity, USToken = model.generate(samples, cfg.config.generate, prompts=prompt)
        print('text ', ''.join(text))
        f_out.write(bs_name+'\t'+ ''.join(text) +'\n')
        f_gt.write(bs_name+'\t'+test_itm['text']+'\n')
        # plot_mel_spectrogram(pre_mel.squeeze().detach().cpu().numpy(), gt_mel.squeeze().detach().cpu().numpy())