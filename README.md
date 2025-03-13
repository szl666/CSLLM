
# CSLLM (Crystal Synthesis Large Language Models)
An LLM system for the ultra-accurate (TPR=98.8%) prediction of the synthesizability and precursors of crystal structures.

![图片1](https://github.com/user-attachments/assets/eaa81aab-afff-4c00-a677-3ad433f9955f)

This is the repository for the paper "Is Large Language Model All You Need to Predict the Synthesizability and Precursors of Crystal Structures?"
https://arxiv.org/abs/2407.07016
The code will be released soon.

The GUI example of using CSLLM

![CSLLM(11)](https://github.com/user-attachments/assets/43d0aca7-c16e-406e-bd64-51fff5efd9e8)


## Overview

CSLLM is a powerful tool that leverages three specialized large language models to predict crystal structure synthesis:

1. **synthesis_llm**: Predicts whether a given crystal structure can be synthesized
2. **method_llm**: Recommends methods for synthesizing the crystal structure
3. **precursor_llm**: Suggests precursors for the synthesis process

## Features

- Upload and analyze CIF or POSCAR crystal structure files
- Visualize structures using VESTA
- Get combined predictions from multiple specialized LLMs
- Interactive chat interface

## Models

All three models can be downloaded from the HuggingFace repository:
[https://huggingface.co/zhilong777/csllm](https://huggingface.co/zhilong777/csllm)

The repository contains the following models:
- `synthesis_llm`: For synthesizability prediction
- `method_llm`: For synthesis method recommendation
- `precursor_llm`: For precursor suggestion

## Installation

### Environment Setup

All required dependencies are specified in the `environment.yml` file. You can create the conda environment using:

```bash
conda env create -f environment.yml
conda activate csllm
```

### Download Models

Download the three specialized models from HuggingFace:

```bash
# Install git-lfs if you haven't already
git lfs install

# Clone the models repository
git clone https://huggingface.co/zhilong777/csllm
```

Alternatively, you can download each model individually from the HuggingFace platform at [https://huggingface.co/zhilong777/csllm](https://huggingface.co/zhilong777/csllm).

### Setup VESTA (Optional)

Install VESTA for structure visualization:
- Download from [VESTA website](https://jp-minerals.org/vesta/en/)
- Update the path in `gui.py` if necessary

## Usage

### Basic Usage

To use all three specialized models for comprehensive crystal synthesis analysis:

```bash
python gui.py \
  --model_paths ./csllm/synthesis_llm,./csllm/method_llm,./csllm/precursor_llm \
  --device cuda \
```

### Command Line Arguments

- `--model_paths`: Comma-separated paths to the three model directories (synthesis_llm,method_llm,precursor_llm)
- `--device`: Device to run models on (`cuda` or `cpu`)

