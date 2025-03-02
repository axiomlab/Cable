## Context-aware Biases for Length Extrapolation 

The source code of [(Context-aware Biases for Length Extrapolation)]()

<p align="center">
 <img src="https://raw.githubusercontent.com/axiomlab/Cable/refs/heads/main/Figures/pull_figure.png"  width="400" height="300"/>
</p>

### 🚀 News
- [2025.02.3] Code release

#### Upcoming
- [ ] Cleaning codebase
- [ ] Adding scripts for training ALiBi, RoPE, T5-bias

### Requirements

- fire>=0.1.3
- regex==2017.4.5
- requests==2.21.0
- tqdm==4.31.1
- tiktoken==0.9.0 

### Datasets and Models
- Fineweb-Edu [ :link: ](https://arxiv.org/abs/2406.17557) [:hugs:](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- Fineweb [ :link: ](https://arxiv.org/abs/2406.17557) [:hugs:](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- WikiText-103 [ :link: ](https://arxiv.org/abs/1609.07843) [:hugs:](https://huggingface.co/datasets/iohadrubin/wikitext-103-raw-v1)
- WikiText-2 [ :link: ](https://arxiv.org/abs/1609.07843) [:hugs:](https://huggingface.co/datasets/mindchain/wikitext2)

Download the datasets from HuggingFace and use use ```dataset_preparation.py``` for saving tokenized dataset.


Some of trained models:
| Dataset | Model | Parameters |Sequence Length | Checkpoint |
| -------- | :-------: | :-------: | :-------: | :-------: |
| Fineweb-Edu | GPT-Medium | 334M | 1024 | [Link]() |
| Fineweb-Edu | GPT-Medium | 334M | 512 | [Link]() |
| WikiText-103 | GPT-Tiny | 44M | 1024 | [Link]() |
| WikiText-103 | GPT-Tiny | 44M | 512 | [Link]() |


### Training
- Single GPU
  ```shell
  python Cable.py --dataset-dir "path to dataset" --model "medium or small or tiny" --save-dir "dir for logs"
  ```

- Multiple GPUs
  ```shell
  export OMP_NUM_THREADS=8; CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 Cable.py
  ```

For Hellaswag benchmark and evaluating extrapolation please use ```evaluation.ipynb``` notebook.


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
This codebase is heavily borrowed from [Language Models are Unsupervised Multitask Learners](https://github.com/openai/gpt-2). Thanks for their excellent work.
