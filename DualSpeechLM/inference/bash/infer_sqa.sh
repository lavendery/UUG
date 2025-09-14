CUDA_VISIBLE_DEVICES=3 python3 inference/scripts/phi3.5mini-instruct_inference_audio_sqa.py \
    --model-cfg-path inference/configs/llm/phi3.5mini-instruct.yaml \
    --data-dir data/SQA/test/1splits_ustokenizer \
    --asr-gen-file exp/llm_eval/sqa/sqa_res.scp \
    --asr-gt-file exp/llm_eval/sqa/sqa_gt.scp \