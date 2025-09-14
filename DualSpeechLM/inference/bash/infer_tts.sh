CUDA_VISIBLE_DEVICES=0 python3 inference/scripts/phi3.5mini-instruct_inference_audio_tts.py \
    --model-cfg-path inference/configs/llm/phi3.5mini-instruct.yaml \
    --data-dir data/TTS/LibriTTS_R_test-clean/8splits_ustokenizer_WavTokenizer \
    --npy_dir data/TTS/LibriTTS_R_test-clean/speaker_embedding/embeddings \
    --tts-gen-dir exp/llm_eval/tts/test_clean

CUDA_VISIBLE_DEVICES=0 python3 inference/scripts/phi3.5mini-instruct_inference_audio_tts.py \
    --model-cfg-path inference/configs/llm/phi3.5mini-instruct.yaml \
    --data-dir data/TTS/LibriTTS_R_test-other/8splits_ustokenizer_WavTokenizer \
    --npy_dir data/TTS/LibriTTS_R_test-other/speaker_embedding/embeddings \
    --tts-gen-dir exp/llm_eval/tts/test_other