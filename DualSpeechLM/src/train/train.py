import transformers
from typing import Optional
from dataclasses import dataclass, field
from omegaconf import OmegaConf
import hydra
import pyrootutils
import logging
import random
import numpy as np
import torch

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
from src.train.trainer import CustomTrainer, compute_metrics

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger(__name__)


@dataclass
class ConfigPathArguments:
    model: Optional[str] = field(default=None, metadata={"help": "config path of model used to initialize LM model"})
    tokenizer: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer used to initialize tokenizer"})
    train_data: Optional[str] = field(default=None, metadata={"help": "config path of train dataset"})
    eval_data: Optional[str] = field(default=None, metadata={"help": "config path of eval dataset"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(
        default=None, metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    overwrite_output_dir: bool = field(default=False, metadata={"help": "Overwrite the content of the output directory"})
    optim: str = field(default="adamw_hf")
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    min_lr_ratio: float = field(
        default=0.1, metadata={"help": "The min lr ratio reqpect to the learning rate, only used to cosine lr scheduler"})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1, metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."})

    lr_scheduler_type: str = field(default='cosine', metadata={"help": "The scheduler type to use."})
    # report_to: Optional[List[str]] = field(default=['tensorboard'],
    #                                        metadata={"help": "The list of integrations to report the results and logs to."})
    save_steps: int = field(default=1000, metadata={"help": "The interval between saving the model checkpoint."})
    bf16: bool = field(default=False,
                       metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"})
    fp16: bool = field(default=False,
                       metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"})
    dataloader_num_workers: int = field(default=8, metadata={"help": "The number of workers to use for data loading."})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/CPU for training."})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/CPU for evaluation."})
    run_name: str = field(default=None, metadata={"help": "The name of the run."})

    torch_compile: bool = field(default=False, metadata={"help": "Whether to use torch.jit.trace to compile the model."})
    coco_caption_root: str = field(
        default=None, metadata={"help": "root path of coco karpathy which is used to comput caption metrics during training."})

id2task = {1: 'asr', 2: 'tts', 3: 'vc', 4: 't2st', 5: 'sc', 6: 'sqa', 7: 's2tt', 8: 'ser'}

class Trainer(CustomTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        input_labels = inputs.get('labels', None)

        target_audio_ids = inputs.get('target_audio_ids', None)
        spk_emb = inputs.get('spk_emb', None)
        attention_mask_question = inputs.get('attention_mask_question', None)
        attention_mask_answer = inputs.get('attention_mask_answer', None)
        task_id = inputs.get('task_id', None)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_labels, target_audio_ids=target_audio_ids, spk_emb=spk_emb, attention_mask_question=attention_mask_question, attention_mask_answer=attention_mask_answer)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if "multi_loss" in outputs and outputs["multi_loss"] is not None and self.state.is_world_process_zero:  
            if self.state.global_step % self.args.gradient_accumulation_steps == 0: 
                multi_loss = outputs["multi_loss"]
                current_task = id2task[task_id[0][0].item()]  
                task_suffix = f"_{current_task}" 
                log_dict = {}
                llama_key = f"llama{task_suffix}"
                if "loss_llama" in multi_loss:
                    log_dict[llama_key] = multi_loss["loss_llama"].mean().item()
                lm_decoder_key = f"lm_decoder{task_suffix}"
                if "loss_lm_decoder" in multi_loss:
                    log_dict[lm_decoder_key] = multi_loss["loss_lm_decoder"].mean().item()
                    acc_key = f"acc_lm_decoder{task_suffix}"
                    if "acc_lm_decoder" in multi_loss:
                        log_dict[acc_key] = multi_loss["acc_lm_decoder"].mean().item()
                if log_dict:
                    self.log(log_dict)
                
        return (loss, outputs) if return_outputs else loss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train():
    global local_rank

    parser = transformers.HfArgumentParser((ConfigPathArguments, TrainingArguments))
    cfg_path, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    set_seed(7)

    train_data_cfg = OmegaConf.load(cfg_path.train_data)
    model_cfg = OmegaConf.load(cfg_path.model)
    tokenizer_cfg = OmegaConf.load(cfg_path.tokenizer)

    use_peft = 'peft' in model_cfg._target_ or 'lora' in model_cfg._target_
    print('Use peft or not: ', use_peft)

    print('Init tokenizer')
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)
    tokenizer.pad_token = tokenizer.unk_token
    print('Init train data')
    train_data = hydra.utils.instantiate(train_data_cfg, tokenizer=tokenizer)
    print('Init model')
    if use_peft:
        
        model = hydra.utils.instantiate(model_cfg, tokenizer=tokenizer)
    else:
        model = hydra.utils.instantiate(model_cfg)
        print(f'Length of tokenizer and resize embedding: {len(tokenizer)}')
        model.resize_token_embeddings(len(tokenizer))

    if cfg_path.eval_data is not None:
        eval_data_cfg = OmegaConf.load(cfg_path.eval_data)
        eval_data = hydra.utils.instantiate(eval_data_cfg, tokenizer=tokenizer)
    else:
        eval_data = None

    print('Init done.')

    model.config.use_cache = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == '__main__':
    train()
