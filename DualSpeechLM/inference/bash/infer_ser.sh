CUDA_VISIBLE_DEVICES=1 python3 inference/scripts/phi3.5mini-instruct_inference_audio_ser.py \
    --model-cfg-path inference/configs/llm/phi3.5mini-instruct.yaml \
    --data-dir data/SER/IEMOCAP/4splits_ustokenizer \
    --asr-gen-file exp/llm_eval/ser/iemocap_res.scp \
    --asr-gt-file exp/llm_eval/ser/iemocap_gt.scp \
    # --batch-size 5
