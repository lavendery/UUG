# CUDA_VISIBLE_DEVICES=0 python3 inference/scripts/phi3.5mini-instruct_inference_audio_asr_batch.py \
CUDA_VISIBLE_DEVICES=0 python3 inference/scripts/phi3.5mini-instruct_inference_audio_asr.py \
    --model-cfg-path inference/configs/llm/phi3.5mini-instruct.yaml \
    --data-dir data/ASR/LibriSpeech_test-clean/8splits_ustokenizer \
    --asr-gen-file exp/llm_eval/asr/libri_test_clean_res.scp \
    --asr-gt-file exp/llm_eval/asr/libri_test_clean_gt.scp \


# CUDA_VISIBLE_DEVICES=0 python3 inference/scripts/phi3.5mini-instruct_inference_audio_asr_batch.py \
CUDA_VISIBLE_DEVICES=0 python3 inference/scripts/phi3.5mini-instruct_inference_audio_asr.py \
    --model-cfg-path inference/configs/llm/phi3.5mini-instruct.yaml \
    --data-dir data/ASR/LibriSpeech_test-other/8splits_ustokenizer \
    --asr-gen-file exp/llm_eval/asr/libri_test_other_res.scp \
    --asr-gt-file exp/llm_eval/asr/libri_test_other_gt.scp \
    # --batch-size 5