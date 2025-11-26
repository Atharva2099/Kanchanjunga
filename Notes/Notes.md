Need to build a 100 M model atleast with 50M dense and 5x10M experts.

<u>Experts in:</u>

- Math `GSM8k` or `harvardnlp/math_dataset`
- Python code `bigcode/the-stack-v2` and `codeparrot/codeparrot-clean`
- Web search `HuggingFaceFW/fineweb`
- Science `Arxiv`
- Legal `ukp-world/us-patents`, `big_patent`, `pile_of_law`, `free_law`/`case_law`
- Chat `OpenAssistant/oasst1`, `databricks/databricks-dolly-15k`, `google/flan`
---
<u>Imp tips:</u>

- Scale down sequence length but keep MoEs up in case of RAM shortage during training
- Keep each expert’s shard to 5–50M tokens for Colab-scale runs
- Warm-start routing by sampling mini-batches per expert in round-robin for the first few thousand steps. Then switch to mixed sampling; the aux loss will take over.
---
**Day 1: Tokenisers and data streaming**

Used [allenai/c4](https://huggingface.co/datasets/allenai/c4) dataset with HF streaming used tiktoken for tokenising and a buffer to make sure everything works smoothly on a mac. Tokenised into 3 files train.bin , val.bin and a config file called meta.pkl

**Day 2: understanding attention**

Input tokens are converted to vectors and stored in matrix call it query, we also have every word precomputed as keys 

$$attention\;score  =Query(Q) \circledast Key(K)$$
Where $\circledast$ denotes element-wise dot product

$$attention\;weights(\alpha) = softmax(attention\;score /\sqrt{dK})$$

attention weights are just normalised attention score.

**The rationale behind scaled-dot product attention**
The reason for the normalization by the embedding dimension size is to
improve the training performance by avoiding small gradients. For instance,when scaling up the embedding dimension, which is typically greater thant housand for GPT-like LLMs, large dot products can result in very small gradients during backpropagation due to the softmax function applied to them. As dot products increase, the softmax function behaves more like a step function, resulting in gradients nearing zero. These small gradients can drastically slow down learning or cause training to stagnate.The scaling by the square root of the embedding dimension is the reason why this self-attention mechanism is also called scaled-dot product attention.

**Computing the context vector**

To compute context vectors you multiple them with values

$$Context\;vector = \alpha\;\circledast\;Value(V)$$
Everything can be optimised with matrix multiplication