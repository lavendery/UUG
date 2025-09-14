CUDA_VISIBLE_DEVICES=3 python3 inference/scripts/phi3.5mini-instruct_inference_audio_sc.py \
    --model-cfg-path inference/configs/llm/phi3.5mini-instruct.yaml \
    --data-dir data/SC/test/4splits_ustokenizer \
    --npy_dir data/SC/test/speaker_embedding/embeddings \
    --sc-gen-dir exp/llm_eval/sc/v0 \
