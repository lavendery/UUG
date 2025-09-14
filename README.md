<p align="center">

  <h2 align="center"> DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling with Large Language Models </h2>
  <p align="center">
        <a href="https://arxiv.org/abs/2508.08961">
        <img src='https://img.shields.io/badge/arXiv-red' alt='Paper Arxiv'></a> &nbsp; &nbsp;  &nbsp; 
        <a href='https://lavendery.github.io/Unified-Understanding-and-Generalization-Demo/'>
        <img src='https://img.shields.io/badge/Project_Page-green' alt='Project Page'></a> &nbsp;&nbsp; &nbsp; 
        <a href="https://github.com/lavendery/UUG">
          <img src="https://img.shields.io/badge/Code-black?logo=github&logoColor=white" alt="Code">
        </a>&nbsp;&nbsp; &nbsp; 
  </p>
    </p>

This repo contains our official implementation of <strong> DualSpeechLM </strong>. For the generated audio, Please refer to [[Demo]](https://lavendery.github.io/Unified-Understanding-and-Generalization-Demo/). You can find our paper from [[Paper]](https://arxiv.org/abs/2508.08961).

## Abstract
Extending pre-trained Large Language Models (LLMs)'s speech understanding or generation abilities by introducing various effective speech tokens has attracted great attention in the speech community. However, building a unified speech understanding and generation model still faces the following challenges: (1) Due to the huge modality gap between speech tokens and text tokens, extending text LLMs to unified speech LLMs relies on large-scale paired data for fine-tuning, and (2) Generation and understanding tasks prefer information at different levels, e.g., generation benefits from detailed acoustic features, while understanding favors high-level semantics. This divergence leads to difficult performance optimization in one unified model. To solve these challenges, in this paper, we present two key insights in speech tokenization and speech language modeling. Specifically, we first propose an Understanding-driven Speech Tokenizer (USTokenizer), which extracts high-level semantic information essential for accomplishing understanding tasks using text LLMs. In this way, USToken enjoys better modality commonality with text, which reduces the difficulty of modality alignment in adapting text LLMs to speech LLMs. Secondly, we present DualSpeechLM, a dual-token modeling framework that concurrently models USToken as input and acoustic token as output within a unified, end-to-end framework, seamlessly integrating speech understanding and generation capabilities. Furthermore, we propose a novel semantic supervision loss and a Chain-of-Condition (CoC) strategy to stabilize model training and enhance speech generation performance. Experimental results demonstrate that our proposed approach effectively fosters a complementary relationship between understanding and generation tasks, highlighting the promising strategy of mutually enhancing both tasks in one unified model.
<p align="center">
<!-- <img src="figs/USTokenizer.png"/> -->
<img src="figs/DualSpeechLM.png"/>
</p>

## 📣 News & TODOs
- [x] **[2025.08.12]** Release paper and project page.
- [x] **[2025.09.03]** Release USTokenizer code.
- [x] **[2025.09.14]** Release DualSpeechLM code.
- [ ] Release pretrained weights (coming soon).


## USTokenizer
### 1.Installation Environment
```
# Clone repository
git clone https://github.com/lavendery/UUG.git
cd USTokenizer

# Create and activate conda environment
conda create -n USTokenizer python=3.10
conda activate USTokenizer

# Install dependencies
pip install -r requirements.txt
```
### 2. Training the Tokenizer
(1) Configure your settings in configs/config.yaml.

(2) Prepare the required public checkpoints: `Llama` and `Whisper`.  
For example, you can download [whisper-medium](https://huggingface.co/openai/whisper-medium) and [llama](https://huggingface.co/meta-llama/Llama-3.2-3B).
After that, you need to modify the `llama_path` and `whisper_path` in configs/config.yaml

(3) **Prepare Training/Dev/Test Dataset**

Create your training/dev/test dataset file meta.json with the following format:
```
{
    "annotation": [
        {
            "path": "audio_path/4693/17508/4693-17508-0000.flac",
            "text": "chapter eleven the eagle screams despite the glories of the amalfi road",
            "task": "asr"
        },
        ...
    ]
}
```
(4) **Run Train**
```
cd USTokenizer
bash run.sh
```

### 3. Inference/Generating Discrete Code (USToken)
To generate discrete codes using the trained tokenizer:
```
bash infer.sh
```

## DualSpeechLM
### 1. Installation Environment
```
# Clone repository
git clone https://github.com/lavendery/UUG.git
cd DualSpeechLM

# Create and activate conda environment
conda create -n DualSpeechLM python=3.8
conda activate DualSpeechLM

# Install dependencies
pip install -r requirements.txt
```
### 2. Training the SpeechLM
(1) **Prepare Text LLM**

Download [Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) as the text backbone, and place it in:
```
checkpoint/microsoft/Phi-3.5-mini-instruct
```

(2) **Prepare Tokenizer**

Expand the vocabulary of the text LLM. For example, with USTokenizer, you need to add **1024 new tokens**.
After that, Place the expanded vocabulary into:
```
checkpoint/tokenizer/Phi-3.5-mini-instruct-tokenizer-audio-1024
```

(3) **Prepare WavTokenizer**

Download [config file](https://github.com/jishengpeng/WavTokenizer/tree/main/configs) and [checkpoint](https://huggingface.co/novateur/WavTokenizer-large-unify-40token) of WavTokenizer, and place them in: 
```
WavTokenizer/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml
WavTokenizer/WavTokenizer-large-unify-40token/wavtokenizer_large_unify_600_24k.ckpt
```

(4) **Dataset Preparation Example**

The `pre_data.sh` script is provided as an **example for ASR (Automatic Speech Recognition)**.
It demonstrates how to construct `.tar` files from raw audio and transcription data.
You need to create `.tar` dataset files based on prepare raw_wav.scp and text.scp:
```
bash pre_data.sh
```

The script ensures that the data format matches the requirements of **DualSpeechLM** training.
If you want to prepare datasets for other tasks (e.g., TTS, Speech Translation), you can follow the same structure, but adjust the script to handle different input/output modalities accordingly.

(5) **Run Train**
```
bash run-multi_task.sh
```

### 3. Inference
(1) **Merge checkpoint**

First, you need to merge checkpoint:
```
bash merge.sh
```
After merging, you will get: `exp/DualSpeechLM/checkpoint-merged-60000.`

(2) **Run inference**

```
cd inference/bash
# Perform different tasks using the provided scripts:
bash infer_asr.sh
bash infer_tts.sh
...
```

## Citation
```bibtex
@misc{wang2025dualspeechlm,
      title={DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling with Large Language Models}, 
      author={Yuanyuan Wang and Dongchao Yang and Yiwen Shao and Hangting Chen and Jiankun Zhao and Zhiyong Wu and Helen Meng and Xixin Wu},
      year={2025},
      eprint={2508.08961},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2508.08961}, 
}
```

## Acknowledgments
This repository is developed based on the following repos, and we thank them for open-sourcing their great code!
* [SALMONN](https://github.com/bytedance/SALMONN/tree/salmonn)
* [RepCodec](https://github.com/mct10/RepCodec)
* [AudioDec](https://github.com/facebookresearch/AudioDec/tree/main)
