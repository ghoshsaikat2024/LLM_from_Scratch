# LLM From Scratch Using Pytorch

A hands-on implementation and study repository for understanding Large Language Models from the ground up using PyTorch.

The goal of this repository is to move beyond using pretrained models and understand how modern language models work internally — from processing raw text and implementing attention mechanisms to building a GPT-style Transformer, training it, and exploring reasoning models.

# What This Repository Covers

The repository follows a progressive path from the fundamentals of language modeling to complete GPT-style architectures.

1. Working with Text Data

* Raw text preprocessing
* Tokenization
* Token IDs
* Vocabulary construction
* Input-target pair creation
* Context windows
* Dataset and DataLoader construction
* Batch generation

3. Attention Mechanisms

Implementing attention mechanisms step by step:

* Basic attention
* Query, Key, and Value representations
* Attention scores
* Scaled dot-product attention
* Causal attention
* Multi-head attention
* Masking
* Dropout in attention

3. Transformer Architecture

Understanding and implementing the main components of a Transformer decoder:

* Token embeddings
* Positional embeddings
* Layer Normalization
* Feed-forward networks
* Residual connections
* Multi-head causal self-attention
* Transformer blocks
* Output projection
* Weight tying

4. GPT From Scratch

Building a GPT-style decoder-only language model in PyTorch.

The implementation follows this architecture:

```text
Input Tokens
     ↓
Token Embeddings
     +
Positional Embeddings
     ↓
Transformer Blocks
     │
     ├── LayerNorm
     ├── Causal Self-Attention
     ├── Residual Connection
     ├── LayerNorm
     ├── Feed-Forward Network
     └── Residual Connection
     ↓
Output Projection
     ↓
Logits
     ↓
Next Token Prediction
```

5. Training and Pretraining

* Next-token prediction
* Cross-entropy loss
* Training and validation loss
* Optimizers
* Gradient computation
* Backpropagation
* Model evaluation
* Autoregressive generation
  
6. Text Generation

Understanding how a trained language model generates text one token at a time.

Topics include:

* Autoregressive generation
* Context windows
* Logits and probability distributions
* Sampling
* Temperature
* Top-k sampling
* Sequential token generation
  
7. Reasoning Models

The repository also explores the foundations of reasoning-oriented language models and how reasoning can be incorporated into language-model training and inference.

## Repository Structure

```text
LLM_from_Scratch/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_working_with_text_data.ipynb
│   ├── 02_coding_attention_mechanism.ipynb
│   ├── 03_implementing_gpt_from_scratch.ipynb
│   └── 04_pretraining_llm.ipynb
│
├── src/
│   └── gpt_components.py
│
├── assets/
│   └── pics/
│
├── data/
│   └── the-verdict.txt
│
├── docs/
│   └── environment_setup.md
│
└── references/
    └── README.md
```

# References

In this repository I have used Sebastian Raschka's books as the primary references for learning and implementing the concepts covered here.
1. Sebastian Raschka — Build a Large Language Model (From Scratch)
2. Sebastian Raschka — Build a Reasoning Model (From Scratch)
3. Also checkout his code repo as well: [rasbt](https://github.com/rasbt/LLMs-from-scratch.git)



The exact structure may evolve as the implementation grows.
