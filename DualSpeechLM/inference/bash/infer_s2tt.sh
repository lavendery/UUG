CUDA_VISIBLE_DEVICES=2 python3 inference/scripts/phi3.5mini-instruct_inference_audio_s2tt.py \
    --model-cfg-path inference/configs/llm/phi3.5mini-instruct.yaml \
    --data-dir data/S2TT/en_zh-CN_test/8splits_ustokenizer \
    --asr-gen-file exp/llm_eval/s2tt/en_zh_res.scp \
    --asr-gt-file exp/llm_eval/s2tt/en_zh_gt.scp \

CUDA_VISIBLE_DEVICES=2 python3 inference/scripts/phi3.5mini-instruct_inference_audio_s2tt.py \
    --model-cfg-path inference/configs/llm/phi3.5mini-instruct.yaml \
    --data-dir data/S2TT/en_de_test/8splits_ustokenizer \
    --asr-gen-file exp/llm_eval/s2tt/en_de_res.scp \
    --asr-gt-file exp/llm_eval/s2tt/en_de_gt.scp \
    # --batch-size 5