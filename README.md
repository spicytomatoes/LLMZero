# LLMZero

Idea: replace deep learning models in [MuZero](https://arxiv.org/abs/1911.08265) with LLM models.

## Install

### conda environment

create conda environment with python 3.11 (any versions between 3.9 and 3.11 should work, but only 3.11 is tested).

```
conda create -n llmzero python=3.11
conda activate llmzero
```

### pytorch

The code is tested to work with pytorch 2.5.1 with cuda. Check https://pytorch.org/get-started/locally/ to install for your machine.

Install other dependencies **after** installing pytorch with cuda.

```
pip install -r requirements.txt
```

### pyRDDLGym

This implementation of LLMZero supports environments created using [PyRDDLGym](https://github.com/tasbolat1/pyRDDLGym), install it with

```
pip install -q git+https://github.com/tasbolat1/pyRDDLGym.git --force-reinstall
pip install numpy==1.24.2 --force-reinstall
```

### OpenAI API Key

Create `.env` file in the root folder and add

```
OPENAI_API_KEY=<your api key>
```