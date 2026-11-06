<div align="center">

<h2 align="center">
<a href="https://github.com/ernie-research/AoH">Autonomy-of-Heads: Data-Free Sparse Attention from Frozen Query-Key Geometry </a>
</h2>

<h5 align="center">If you like our project, please give us a star ⭐ on GitHub for latest update.</h5>

<p align="center">
  <a href="https://undground.fun/aoh/">
    <img src="https://img.shields.io/badge/Homepage-Project%20Page-1f6feb.svg" alt="AOH Homepage"/>
  </a>
  <a href="https://arxiv.org/abs/2608.06849">
    <img src="https://img.shields.io/badge/Paper-arXiv%3A2608.20473-b31b1b.svg" alt="arXiv Paper"/>
  </a>
  <!-- <a href="https://huggingface.co/ernie-research/AVIOT">
    <img src="https://img.shields.io/badge/Model-%F0%9F%A4%97%20AVIOT-ffd21e.svg" alt="Hugging Face Model"/>
  </a> -->
</p>

</div>

## Performance

**AoH (Autonomy-of-Heads)**, a training-free method that classifies attention heads from the frozen query-key projection geometry.
<div align="center">
  <img src="./assets/figures/aoh-accuracy.png" width="46%">
  <img src="./assets/figures/aoh-efficiency.png" width="51%">
</div>

## Demo
https://github.com/user-attachments/assets/081e901d-3fec-4d57-bc47-cb619e748221

## Method
<div align="center">
  <img src="./assets/figures/aoh_vs_Input-based-0723.png" width="58%">
</div>

For each query head, AoH measures the effective rank of the query-key interaction matrix using a small head-dimension Gram matrix. Low-rank heads are classified as retrieval heads and retain full attention; higher-rank heads are classified as streaming heads and can use a local attention window.

Both multi-head attention (MHA) and grouped-query attention (GQA) models are supported. For GQA, per-query-head scores are aggregated within each KV group using the mean, minimum, or maximum score.

## Installation

Python 3.10 or newer is required.

```bash
pip install -e .
```

Install plotting dependencies when visualization is needed:

```bash
pip install -e ".[visualization]"
```

Install LongBench evaluation dependencies before running the result scripts:

```bash
pip install -e ".[evaluation]"
```

Model weights are loaded through Hugging Face Transformers. Access to gated models must be configured separately with Hugging Face.

## Command Line

```bash
python -m aoh \
  --model_path Qwen/Qwen2.5-7B \
  --device cuda:0 \
  --sparsity 0.5 \
  --agg mean \
  --output_dir outputs/qwen25-7b
```

The installed `aoh` command provides the same interface:

```bash
aoh --model_path Qwen/Qwen2.5-7B --output_dir outputs/qwen25-7b
```

Useful options:

- `--sparsity`: fraction of heads classified as streaming in each layer.
- `--agg {mean,min,max}`: GQA score aggregation strategy.
- `--binary`: use binary projection weights for accelerated rank estimation.
- `--torch_dtype {auto,float32,float16,bfloat16}`: checkpoint loading dtype; `auto` preserves the stored dtype and is the default. AoH still performs the Q/K rank computation in float32.

Run `python -m aoh --help` for the complete interface.

## Python API

```python
from aoh import AoHClassifier

classifier = AoHClassifier(
    model_path="Qwen/Qwen2.5-7B",
    device="cuda:0",
    agg="mean",
)
labels = classifier.classify(sparsity=0.5)
classifier.save("outputs/qwen25-7b")
```

`labels` has shape `(num_layers, num_kv_heads)` for GQA models and `(num_layers, num_heads)` for MHA models. A value of `1` denotes a retrieval head and `0` denotes a streaming head.

## Outputs

Each run writes the following files to `--output_dir`:

- `head_classification.npy`: binary retrieval/streaming labels.
- `eff_rank.npy`: per-query-head effective ranks.
- `eff_rank_kv.npy`: effective ranks aggregated by KV group.
- `results.json`: model and run metadata.

## Examples

```bash
python examples/qwen25_7b.py
python examples/qwen3.py
```

The examples load full model weights. Adjust the model name, device, and output path in each script for your environment.


## Results
```bash
bash scripts/run_result_eval.sh
```
We evaluate AoH on LongBench, a long-context understanding suite covering 21 tasks across six categories. The evaluated models are Qwen3-8B, Llama-3.1-8B-Instruct, and Qwen2.5-7B. Unless otherwise stated, streaming heads retain 128 sink tokens and a 256-token recent window, and the main experiments use 50% per-layer sparsity.

`eval/results/run_longbench.py` runs the FlashAttention-2 AoH runtime by default (`--protocol aoh`, requires `flash-attn>=2.7`): the prompt except its last 50 tokens (`--decoding-simulation-length`) is prefilled with `flash_attn_func`, streaming heads are trimmed to sink + recent KV, and the remaining tokens are decoded one by one. `--protocol eager` keeps the reference eager kernels with chunked prefill (`--chunk-size 256`). Both paths compute attention scores in fp32.

![results](./assets/figures/aoh-main-results.png)


## Efficiency
```bash
bash scripts/run_efficiency_eval.sh
```
AoH reduces KV-cache memory nearly proportionally to the fraction of streaming heads, and its latency benefits become more pronounced at longer contexts where attention and cache access dominate runtime. As shown in Figure 5 and Table 11, at 75% sparsity, AoH achieves up to 3.24x prefill speedup and 9.14x decode speedup at 256K context, while reducing KV-cache memory by up to 3.98x on Llama-3.1-8B-Instruct-128K.
![efficiency](./assets/figures/efficiency-figure.png)
![efficiency](./assets/figures/aoh-efficiency-table.png)


## Visualization

After running AoH:

```bash
python visualization/plot_head_classification.py outputs/qwen25-7b
python visualization/plot_effective_rank.py outputs/qwen25-7b
```

## Run the Demo
After setting up the environment, you can run the following script to execute the PassKey retrieval demo on the Llama-3.1-8B-Instruct-128K model. The demo is designed to run on a single A100 GPU and supports a context length of up to 128K tokens.

```bash
bash scripts/demo.sh
```


## Citation
If you find AoH useful or relevant to your project and research, please kindly cite our paper:

```
@misc{yang2026autonomy,
  title={Autonomy-of-Heads: Data-Free Sparse Attention from Frozen Query-Key Geometry},
  author={Yang, Yehan and Shang, Junyuan and Li, Yang and Zhao, Guanqun and Wang, Shuohuan and Yu, Dianhai},
  year={2026},
  eprint={2608.06849},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2608.06849}
}
```
