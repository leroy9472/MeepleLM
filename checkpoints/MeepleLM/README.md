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
- name: qwen3-8b-bgg-persona1
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen3-8b-bgg-persona1

This model is a fine-tuned version of [/inspire/hdd/global_user/lizizhen-240108540152/model/Qwen3-8B](https://huggingface.co//inspire/hdd/global_user/lizizhen-240108540152/model/Qwen3-8B) on the bgg_persona_train dataset.
It achieves the following results on the evaluation set:
- Loss: 1.6187

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
| 1.7789        | 0.1985 | 200  | 1.7694          |
| 1.6935        | 0.3971 | 400  | 1.7158          |
| 1.684         | 0.5956 | 600  | 1.6901          |
| 1.6918        | 0.7941 | 800  | 1.6729          |
| 1.665         | 0.9927 | 1000 | 1.6593          |
| 1.5908        | 1.1906 | 1200 | 1.6499          |
| 1.6181        | 1.3891 | 1400 | 1.6410          |
| 1.6018        | 1.5877 | 1600 | 1.6342          |
| 1.6306        | 1.7862 | 1800 | 1.6288          |
| 1.5952        | 1.9847 | 2000 | 1.6245          |
| 1.5744        | 2.1827 | 2200 | 1.6227          |
| 1.6107        | 2.3812 | 2400 | 1.6206          |
| 1.5768        | 2.5797 | 2600 | 1.6194          |
| 1.5975        | 2.7783 | 2800 | 1.6188          |
| 1.5819        | 2.9768 | 3000 | 1.6187          |


### Framework versions

- PEFT 0.17.1
- Transformers 4.57.1
- Pytorch 2.7.1+cu126
- Datasets 3.1.0
- Tokenizers 0.22.1