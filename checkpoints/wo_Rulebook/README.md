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
- name: qwen3-8b-bgg-norule1
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen3-8b-bgg-norule1

This model is a fine-tuned version of [/inspire/hdd/global_user/lizizhen-240108540152/model/Qwen3-8B](https://huggingface.co//inspire/hdd/global_user/lizizhen-240108540152/model/Qwen3-8B) on the bgg_norule_train dataset.
It achieves the following results on the evaluation set:
- Loss: 2.4634

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
| 2.575         | 0.1985 | 200  | 2.5757          |
| 2.4894        | 0.3971 | 400  | 2.5379          |
| 2.485         | 0.5956 | 600  | 2.5192          |
| 2.5385        | 0.7941 | 800  | 2.5064          |
| 2.4994        | 0.9927 | 1000 | 2.4952          |
| 2.4092        | 1.1906 | 1200 | 2.4885          |
| 2.4787        | 1.3891 | 1400 | 2.4816          |
| 2.4563        | 1.5877 | 1600 | 2.4758          |
| 2.4685        | 1.7862 | 1800 | 2.4711          |
| 2.4328        | 1.9847 | 2000 | 2.4675          |
| 2.4091        | 2.1827 | 2200 | 2.4668          |
| 2.4141        | 2.3812 | 2400 | 2.4651          |
| 2.4392        | 2.5797 | 2600 | 2.4639          |
| 2.4469        | 2.7783 | 2800 | 2.4635          |
| 2.3995        | 2.9768 | 3000 | 2.4634          |


### Framework versions

- PEFT 0.17.1
- Transformers 4.57.1
- Pytorch 2.7.1+cu126
- Datasets 3.1.0
- Tokenizers 0.22.1