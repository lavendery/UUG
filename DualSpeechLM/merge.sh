. ./path.sh
CUDA_VISIBLE_DEVICES=2 python3 src/tools/merge_lora_weights.py \
  --model_cfg configs/model/Phi-3.5-mini-instruct_lora_formerge.yaml \
  --tokenizer_cfg configs/tokenizer/speech_phi3.5mini-instruct_tokenizer.yaml \
  --lora_model exp/DualSpeechLM/checkpoint-60000 \
  --save_path exp/DualSpeechLM/checkpoint-merged-60000
