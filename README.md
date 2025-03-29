# Building and Training a GPT-2 Model from Scratch

This project contains all the necessary code and instructions to build, train, and use a GPT-2 model from scratch. While some of the training hyperparameters used in the GPT-2 paper are not explicitly stated, this project uses configurations inspired by GPT-3 (see the [References](#References) section). This GPT-2 model has 124M parameters. See [this file ](./Overview.pdf) to better understand the project. Additionally, this project wouldn't have been possible without the resources listed in the [Acknowledgements](#Acknowledgements) section.

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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 3. Train the Model

To train the model from scratch, use the provided `main.py` script. This script sets up the data, model, and training loop:

```bash
python main.py
```

The training script will save the best model checkpoint in the `ckpt/` directory, which can later be used for text generation.

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
    weight_tying = True

@dataclass
class TrainingConfig:
    max_steps = 3000 
    warmup_steps =100 # 715  
    max_lr = 6e-4   
    min_lr = 6e-5  
    weight_decay = 0.1  
    batch_size = 64 
    max_new_token = 100
    temperature = 1  
    k_top = 50   
    grad_accum_steps = 8 
```

## Inference 

### Generating Text

Once the model is trained, you can use the `generate_outputs.py` script to generate text based on a given input prompt. The script allows you to control the randomness and diversity of the generated text using the `--temperature` and `--k_top` arguments.

#### Command

```bash
python generate_outputs.py "sample_input_text" --temperature 0.7 --k_top 50
```

#### Arguments

- `sample_input_text`: The input text prompt to generate text from.
- `--temperature`: Controls the randomness of the generated text. Lower values (e.g., 0.5) make the output more deterministic, while higher values (e.g., 1.0) make it more random. Default is `0.7`.
- `--k_top`: Controls the top-k sampling for text generation. Only the top-k tokens with the highest probabilities are considered for sampling. Default is `50`.

#### Example 

**sample_input_text:**
*Tommy had a little puppy named Max. Every day, they went to the park to play. Tommy threw a ball, and Max ran to get it.*

**Output:** 
*Tommy had a little puppy named Max. Every day, they went to the park to play. Tommy threw a ball, and Max ran to get it. <u>**But one day, Max saw a squirrel and chased after it. Tommy got angry and said, "Max, you are naughty! You should not chase squirrels!" Max looked sad and said, "I'm sorry, Tommy. I just wanted to play with you." Tommy hugged Max and said, "It's okay, Max. I forgive you. Let's play together again." And they played happily in the park.</u>***  



- Ensure that the `requirements.txt` dependencies are installed before running the scripts.
- The `generate_outputs.py` script requires the trained model checkpoint (`best_model.pth`) to be present in the `ckpt/` directory.

## Notes

- **Dataset**: The *TinyStories* dataset (see the [References](#References) section), consisting of short and simple stories, was used for training. 

- **Positional Embeddings**: In GPT-2, positional embeddings are trained from scratch like other parameters.
- **Bias**: GPT-2 uses no bias for the final projection.
- **Architecture**: GPT-2 is a decoder-only transformer and uses no bias for the final projection.
- **Normalization**: In GPT-2, normalization is applied before the attention mechanism.
- **GELU Activation**: GPT-2 uses the tanh approximation for the GELU activation function.
- **Weight Initialization**: Proper weight initialization is crucial. For example, the embedding layers are better not initialized with a uniform distribution.


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
generate_outputs.py
generate_tokens.py
Overview.pdf
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
- `generate_outputs.py`: Script for generating text from a trained model.
- `generate_tokens.py`: Script for generating raw token sequences from the dataset.
- `Overview.pdf`: A document providing a high-level overview of the project and its components.
- `config.py`: Contains the configuration classes for the model and training.
- `trainer.py`: Contains the training loop and related functions.
- `main.py`: Main script for training the model.

## License

This project is licensed under the MIT License.

## Acknowledgements

We would like to extend our gratitude to the following resources and individuals who made this project possible:

- **Sebastian Raschka**: Author of [Build a Large Language Model (From Scratch)](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167?crid=228R4JI0P0QFR&dib=eyJ2IjoiMSJ9.XvZyIer9iV133BWXqNiVt_OOJXZheO54dvZtQly8MC25PNYZrN3OWsGLjbg3I0G9hI3LkjwhsORxvHIob3nvCZFgdSSQEFe07VkehijGxT03n4Amdw7lnXxnsOUuWXeglfHnewCcV3DjL9zWHELfh5DG1ZErzFym3S6ZxSuFzNvoPkaq0uDlD_CKwqHdC0KM_RdvIqF0_2RudgvzRli0V155KkusHRck3pG7ybp5VyqKDC_GgL_MEywLwLhFgX6kOCgV6Rq90eTgSHFd6ac8krpIYjsHWe6H3IXbfKGvMXc.473O1-iUZC0z2hdx8L5Z5ZTNxtNV9gNPw_mE7QZ5Y90&dib_tag=se&keywords=raschka&qid=1730250834&sprefix=raschk,aps,162&sr=8-1&linkCode=sl1&tag=rasbt03-20&linkId=84ee23afbd12067e4098443718842dac&language=en_US&ref_=as_li_ss_tl), for providing valuable insights and guidance on building large language models. Reference: Raschka, Sebastian. Build A Large Language Model (From Scratch). Manning, 2024. ISBN: 978-1633437166.
- **Andrej Karpathy**: For his educational YouTube videos, especially [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU&t=12025s) and [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY), which were instrumental in understanding and implementing GPT-2.


## References

- Radford, Alec, et al. "Language Models are Unsupervised Multitask Learners." OpenAI Blog (2019). [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Brown, Tom, et al. "Language Models are Few-Shot Learners." Advances in Neural Information Processing Systems 33 (2020). [GPT-3 Paper](https://arxiv.org/abs/2005.14165)
- Eldan, Ronen, and Yuanzhi Li. "Tinystories: How small can language models be and still speak coherent english?." arXiv preprint arXiv:2305.07759 (2023).
