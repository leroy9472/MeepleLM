---
library_name: peft
license: other
base_model: /inspire/hdd/global_user/lizizhen-240108540152/model/Qwen3-8B
tags:
- base_model:adapter:/inspire/hdd/global_user/lizizhen-240108540152/model/Qwen3-8B
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qwen3-8b-bgg-direct1
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen3-8b-bgg-direct1

This model is a fine-tuned version of [/inspire/hdd/global_user/lizizhen-240108540152/model/Qwen3-8B](https://huggingface.co//inspire/hdd/global_user/lizizhen-240108540152/model/Qwen3-8B) on the bgg_direct_train dataset.
It achieves the following results on the evaluation set:
- Loss: 2.3618

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- gradient_accumulation_steps: 8
- total_train_batch_size: 128
- total_eval_batch_size: 64
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.03
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 2.472         | 0.1985 | 200  | 2.4718          |
| 2.3813        | 0.3971 | 400  | 2.4322          |
| 2.3833        | 0.5956 | 600  | 2.4135          |
| 2.4314        | 0.7941 | 800  | 2.4010          |
| 2.3911        | 0.9927 | 1000 | 2.3906          |
| 2.3026        | 1.1906 | 1200 | 2.3842          |
| 2.3657        | 1.3891 | 1400 | 2.3780          |
| 2.3443        | 1.5877 | 1600 | 2.3729          |
| 2.3564        | 1.7862 | 1800 | 2.3684          |
| 2.3232        | 1.9847 | 2000 | 2.3649          |
| 2.298         | 2.1827 | 2200 | 2.3648          |
| 2.3028        | 2.3812 | 2400 | 2.3636          |
| 2.3169        | 2.5797 | 2600 | 2.3623          |
| 2.3347        | 2.7783 | 2800 | 2.3619          |
| 2.2895        | 2.9768 | 3000 | 2.3618          |


### Framework versions

- PEFT 0.17.1
- Transformers 4.57.1
- Pytorch 2.7.1+cu126
- Datasets 3.1.0
- Tokenizers 0.22.1