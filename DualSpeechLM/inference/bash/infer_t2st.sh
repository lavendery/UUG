CUDA_VISIBLE_DEVICES=2 python3 inference/scripts/phi3.5mini-instruct_inference_audio_t2st.py \
    --model-cfg-path inference/configs/llm/phi3.5mini-instruct.yaml \
    --data-dir data/T2ST/fr_en_test/4splits_ustokenizer \
    --npy_dir data/T2ST/fr_en_test/speaker_embedding/embeddings \
    --t2st-gen-dir exp/llm_eval/t2st/fr_en

CUDA_VISIBLE_DEVICES=2 python3 inference/scripts/phi3.5mini-instruct_inference_audio_t2st.py \
    --model-cfg-path inference/configs/llm/phi3.5mini-instruct.yaml \
    --data-dir data/T2ST/es_en_test/4splits_ustokenizer \
    --npy_dir data/T2ST/es_en_test/speaker_embedding/embeddings \
    --t2st-gen-dir exp/llm_eval/t2st/es_en
