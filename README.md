## Context-aware Biases for Length Extrapolation 

The source code of [(Context-aware Biases for Length Extrapolation)]()

<p align="center">
 <img src="https://raw.githubusercontent.com/axiomlab/Cable/refs/heads/main/Figures/pull_figure.png"  width="400" height="300"/>
</p>

### 🚀 News
- [2025.02.3] Code release

#### Upcoming
- [x] Cleaning codebase
- [ ] Efficient Implementation of method
- [ ] Adding scripts for training ALiBi, RoPE, T5-bias

### Datasets and Models
- Fineweb-Edu [ :link: ](https://arxiv.org/abs/2406.17557) [:hugs:](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- Fineweb [ :link: ](https://arxiv.org/abs/2406.17557) [:hugs:](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- WikiText-103 [ :link: ](https://arxiv.org/abs/1609.07843) [:hugs:](https://huggingface.co/datasets/iohadrubin/wikitext-103-raw-v1)
- WikiText-2 [ :link: ](https://arxiv.org/abs/1609.07843) [:hugs:](https://huggingface.co/datasets/mindchain/wikitext2)

Download the datasets from HuggingFace and use use ```src/dataset_preparation.py``` for saving tokenized dataset.


Some of trained models:
| Dataset | Model | Parameters |Sequence Length | Checkpoint |
| -------- | :-------: | :-------: | :-------: | :-------: |
| Fineweb-Edu(10B) | GPT-Medium | 334M | 1024 | [Link]() |
| Fineweb-Edu(10B) | GPT-Medium | 334M | 512 | [Link]() |
| WikiText-103 | GPT-Tiny | 44M | 1024 | [Link]() |
| WikiText-103 | GPT-Tiny | 44M | 512 | [Link]() |


### Training
- Single GPU
  ```shell
  python train.py --dataset-dir "path to dataset" --model "medium or small or tiny" --save-dir "dir for logs"
  ```

- Multiple GPUs
  ```shell
  torchrun --standalone --nproc_per_node=2 train.py
  ```

For Hellaswag benchmark and evaluating extrapolation please use ```src/evaluation.ipynb``` notebook.


### Length Extrapolation

<p align="center">
 <img src="https://raw.githubusercontent.com/axiomlab/Cable/refs/heads/main/Figures/ablation.png"  width="400" height="300"/>
</p>

A Cable model trained on T=1024 can extrapolate on T=8192, achieving a better performance (PPL=22.22) compared to the sinusoidal model (PPL=22.81) trained on T=8192.

### Runtime and Memory Overhead
<p align="center">
 <img src="https://raw.githubusercontent.com/axiomlab/Cable/refs/heads/main/Figures/time.png"  width="800" height="300"/>
</p>

Cable improves the model's extrapolation ability significantly with a negligible burden in time and memory compared to the vanilla transformer. Furthermore, compared to existing RPE methods, our approach maintains nearly identical training time and GPU memory usage, while its inference overhead remains either negligible or comparable, depending on the sequence length.
 
 ## Citation
If you use this repository for your research or wish to refer to our distillation method, please use the following BibTeX entry:
```bibtex

```

### Acknowledgement
This repo is based on [Karpathy/Build-NanoGPT](https://github.com/karpathy/build-nanogpt). Thanks for their excellent work.
