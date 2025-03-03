# Building/Training a GPT-2 model from Scratch

Welcome to the GPT-2 from Scratch project! This repository contains all the necessary code and instructions to build, train, and use a GPT-2 model from scratch.

## Setup

To get started with this project, follow the steps below to set up your environment and dependencies.

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/your-username/Building-a-GPT-based-LLM-from-Scratch.git
cd Building-a-GPT-based-LLM-from-Scratch
```

### 2. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Download or Set Up the Pretrained Model

If you want to use a pretrained model for inference or fine-tuning, download or set it up:

```bash
python main.py
```

## Usage

Once the dependencies are installed, you can load and use the GPT-2 model for text generation. Here's an example:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model_name = "path_to_your_model"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Encode input text
input_text = "Once upon a time"
inputs = tokenizer.encode(input_text, return_tensors='pt')

# Generate text
output = model.generate(inputs, max_length=50, num_return_sequences=1)

# Decode and print the result
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

## Training

To train the model from scratch, use the provided `main.py` script. This script sets up the data, model, and training loop:

```bash
python main.py
```

### Training Notes

- **Positional Embeddings**: In GPT-2, positional embeddings are trained from scratch like other parameters.
- **Architecture**: GPT-2 is a decoder-only transformer and uses no bias for the final projection.
- **Normalization**: In GPT-2, normalization is applied before the attention mechanism.
- **GELU Activation**: GPT-2 uses the tanh approximation for the GELU activation function.
- **Attention Masking**: Use `att.masked_fill` for attention masking.
- **Weight Initialization**: Proper weight initialization is crucial. For example, the embedding layers should not be initialized with a uniform distribution.

## Configuration

The model and training configurations are defined in the [`config.py`](config.py) file. You can adjust the hyperparameters and other settings as needed.

### Example `config.py`

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size = 50257    # Vocabulary size
    context_length = 1024 # Context length
    emb_dim = 768         # Embedding dimension
    n_heads = 12          # Number of attention heads
    n_layers = 12         # Number of layers
    drop_rate = 0.1       # Dropout rate
    qkv_bias = False      # Query-key-value bias
    weight_tying = False

@dataclass
class TrainingConfig:
    max_steps = 19073 
    warmup_steps = 715
    max_lr = 6e-4  # GPT-3 small 
    min_lr = 6e-5  # GPT-3 small   (0.1 * 6e-4)
    weight_decay = 0.1

    batch_size = 2 
    max_new_token = 50
    temperature = 1
    k_top = 50 
    save_path = "."
```

## Repository Structure

The project has the following structure:

```
dataset/
    data_module.py
    dataset.py
models/
    feed_forward.py
    gelu.py
    gpt.py
    layer_norm.py
    multi_head_self_attention.py
    transformer_block.py
ckpt/
config.py
trainer.py
main.py
README.md
```


- `dataset/`: Contains the dataset-related modules.
  - `data_module.py`: Handles data loading and preprocessing.
  - `dataset.py`: Defines the dataset class.
- `models/`: Contains the model-related modules.
  - `feed_forward.py`: Defines the feed-forward network.
  - `gelu.py`: Defines the GELU activation function.
  - `gpt.py`: Defines the GPT model.
  - `layer_norm.py`: Defines the layer normalization.
  - `multi_head_self_attention.py`: Defines the multi-head self-attention mechanism.
  - `transformer_block.py`: Defines the transformer block.
- `config.py`: Contains the configuration classes for the model and training.
- `trainer.py`: Contains the training loop and related functions.
- `main.py`: Main script for training the model.

## License

This project is licensed under the MIT License.

## Acknowledgements

We would like to extend our gratitude to the following resources and individuals who made this project possible:

- **Sebastian Raschka**: Author of [Build a Large Language Model (From Scratch)](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167?crid=228R4JI0P0QFR&dib=eyJ2IjoiMSJ9.XvZyIer9iV133BWXqNiVt_OOJXZheO54dvZtQly8MC25PNYZrN3OWsGLjbg3I0G9hI3LkjwhsORxvHIob3nvCZFgdSSQEFe07VkehijGxT03n4Amdw7lnXxnsOUuWXeglfHnewCcV3DjL9zWHELfh5DG1ZErzFym3S6ZxSuFzNvoPkaq0uDlD_CKwqHdC0KM_RdvIqF0_2RudgvzRli0V155KkusHRck3pG7ybp5VyqKDC_GgL_MEywLwLhFgX6kOCgV6Rq90eTgSHFd6ac8krpIYjsHWe6H3IXbfKGvMXc.473O1-iUZC0z2hdx8L5Z5ZTNxtNV9gNPw_mE7QZ5Y90&dib_tag=se&keywords=raschka&qid=1730250834&sprefix=raschk,aps,162&sr=8-1&linkCode=sl1&tag=rasbt03-20&linkId=84ee23afbd12067e4098443718842dac&language=en_US&ref_=as_li_ss_tl), for providing valuable insights and guidance on building large language models. Reference: Raschka, Sebastian. Build A Large Language Model (From Scratch). Manning, 2024. ISBN: 978-1633437166.
- **Andrej Karpathy**: For his educational YouTube videos, especially [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU&t=12025s) and [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY), which were instrumental in understanding and implementing GPT-2.
- **Hugging Face**: For their [Transformers library](https://github.com/huggingface/transformers), which provides pre-trained models and tokenizers that significantly accelerated our development process.
- **PyTorch**: For their [deep learning framework](https://pytorch.org/), which served as the backbone for building and training our model.

We are grateful for the contributions of these resources and individuals, which have been invaluable to the success of this project.