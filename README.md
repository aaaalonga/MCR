# MCR

Multi-task training on [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct): LoRA SFT, then sampling and filtering for GRPO, then GRPO. Tasks: GMNER, MNER, MNER-MI, and gRefCOCO.

## Setup

Run from the repository root. `requirements.txt` installs the bundled `ms-swift` and the training and evaluation dependencies.

```bash
conda create -n mcr python=3.10 -y
conda activate mcr
pip install -r requirements.txt
```



### Dataset Preparation

Download from the following link(images are not included):

- **Download:** [pan.baidu.com/s/11_i0bRaRv2l70sNqfH87WQ?pwd=ywp3](https://pan.baidu.com/s/11_i0bRaRv2l70sNqfH87WQ?pwd=ywp3)
- **Extraction code:** `ywp3`

Annotations are in `data/{gmner,mner,mner-mi,grefcoco}/`:

| File                    | Use  |
| ----------------------- | ---- |
| `sft_train_data.json` | SFT  |
| `filtered_data.json`  | GRPO |

Images are not included. Please download them separately. Set the paths in `custom/dataset_*.py`:

- `gmner_path`: Twitter2015 / Twitter2017 images, shared by GMNER and MNER
- `mner_mi_path`: MNER-MI images
- `coco_path`: COCO2014 `train2014`
- `MIN_PIXELS` and `MAX_PIXELS`: `78400`
- For a different model, update the coordinate conversion in `custom/dataset_*.py`

## Run

Two training stages, then inference. After each stage, point the next script at the checkpoint just produced.

**1. SFT**

```bash
bash script/Qwen2.5VL/sft_cot_swift.sh
bash script/merge_lora.sh
```

**2. Sample and GRPO**

```bash
bash script/Qwen2.5VL/sample.sh
bash script/Qwen2.5VL/grpo_multi_type.sh
bash script/merge_lora.sh
```

**3. Inference and evaluation**

```bash
bash script/Qwen2.5VL/infer.sh
bash script/eval.sh
```
