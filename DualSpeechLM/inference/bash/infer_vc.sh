CUDA_VISIBLE_DEVICES=1 python3 inference/scripts/phi3.5mini-instruct_inference_audio_vc.py \
    --model-cfg-path inference/configs/llm/phi3.5mini-instruct.yaml \
    --data-dir data/VC/test_vctk/1splits_ustokenizer \
    --npy_dir data/VC/test_vctk/speaker_embedding/embeddings \
    --tts-gen-dir exp/llm_eval/vc_new/v0