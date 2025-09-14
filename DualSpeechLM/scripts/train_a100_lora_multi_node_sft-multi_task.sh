export HOST_GPU_NUM=4
export HOST_NUM=1
export NODE_NUM=1
export INDEX=0
export CHIEF_IP="localhost"
export CUDA_LAUNCH_BLOCKING=1

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=INFO


torchrun --nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM --master_addr=$CHIEF_IP --master_port=20026 --node_rank=$INDEX src/train/train.py \
    --model configs/model/Phi-3.5-mini-instruct_lora.yaml \
    --tokenizer configs/tokenizer/speech_phi3.5mini-instruct_tokenizer.yaml \
    --train_data configs/data/multi_torchdata_speech_sft.yaml \
    --output_dir exp/DualSpeechLM \
    --deepspeed configs/deepspeed/stage2_bf16.json \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --max_steps 500000 \
    --min_lr_ratio 0.1 \
    --learning_rate 1e-4 \
    --weight_decay 5e-2 \
    --warmup_ratio 0.007 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 3 \
    --report_to "tensorboard" \
    --gradient_checkpointing \
    --dataloader_num_workers 8 \
    --logging_steps 1 \
    --log_level 'info' \
    --logging_nan_inf_filter "no" \
    --bf16 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --ignore_data_skip \
    # --resume_from_checkpoint xxx \

