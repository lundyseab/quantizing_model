# Quantizing Model

A project for quantizing and running the Qwen3-14B language model using 4-bit quantization with BitsAndBytes.

## Overview

This project consists of two main scripts:
- **`quantizing_model.py`**: Quantizes the Qwen3-14B model to 4-bit precision and saves it to disk
- **`model_runner.py`**: Loads the quantized model and provides an interactive interface for generating responses

## Features

- 4-bit quantization using BitsAndBytes (NF4 quantization type)
- Automatic device mapping (GPU/CPU)
- Mixed precision inference for reduced memory usage
- Interactive prompt interface with issue/solution formatting

## Requirements

- Python 3.12-3.14
- CUDA-capable GPU (recommended)
- Poetry for dependency management

## Installation

```bash
poetry install
```

## Usage

### Step 1: Quantize the Model

Run the quantization script to download and quantize the Qwen3-14B model:

```bash
poetry run python quantizing_model.py
```

This will save the quantized model to `./quantized_model`.

### Step 2: Run Inference

Start the interactive model runner:

```bash
poetry run python model_runner.py
```

Enter prompts when prompted. The model will format your input as "issue: {your_input} solution:" and generate a response. Type `exit` to quit.

## Dependencies

- transformers (>=4.57.1,<5.0.0)
- bitsandbytes (>=0.48.2,<0.49.0)
- accelerate (>=1.11.0,<2.0.0)
- torch (>=2.9.0,<3.0.0)
- torchvision (>=0.24.0,<0.25.0)

