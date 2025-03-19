import torch
import argparse
import tiktoken
from models.gpt import GPT
from config import GPTConfig


device = 'cuda'
tokenizer = tiktoken.get_encoding("gpt2")
gpt_config = GPTConfig()

torch.set_float32_matmul_precision('high') 

# Create the GPT model
model = GPT(gpt_config)
model.to(device) # Move the model to the devic
model = torch.compile(model) # Compile it

# Load the best model
state_dict = torch.load("./ckpt/best_model.pth", weights_only=True) 
model.load_state_dict(state_dict)

# Switch a model to evaluation mode
model.eval()

def generate_text(sample_context, temperature=0.7, k_top=50):
    """ Generate text from a given prompt using the trained model. """
    print(sample_context, n, temperature, k_top)
    completion = []
    tokens = tokenizer.encode(sample_context, allowed_special={'<|endoftext|>'})
    while True:
            tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
            with torch.no_grad():
                # Generate logits from the model for the current token sequence
                with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
                    logits = model(tokens_tensor)
                # Extract logits corresponding to the last token in the sequence (shape: [vocab_size])
                last_seq_logits = logits[0, -1, :]
                # Select the indices of the top-k highest logits
                _, top_k_indices = torch.topk(last_seq_logits, k_top)
                # Create a boolean mask where top-k logits are True, others are False
                mask = torch.zeros_like(last_seq_logits, dtype=torch.bool)
                mask[top_k_indices] = True
                # Set logits outside the top-k to negative infinity to exclude them from sampling
                last_seq_logits[~mask] = float('-inf')
                # Scale logits using temperature to control randomness in sampling
                scaled_logits = last_seq_logits / temperature
                # Convert scaled logits to probabilities using softmax
                probs = torch.softmax(scaled_logits, dim=0)
                # Sample the next token based on the probability distribution
                next_token = torch.multinomial(probs, num_samples=1)

                if next_token == 50256: # if the next token is <|endoftext|>, stop generating
                    completion.append(tokens)                
                    break
                
                tokens = tokens + [next_token.item()]

    for item in completion:
        decoded_text = tokenizer.decode(item).replace("\n", " ")
        print(f'> {decoded_text}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a trained model.")
    parser.add_argument("sample_context", type=str, help="Input sentence to generate text from")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--k_top", type=int, help="Top-p nucleus sampling")
    args = parser.parse_args()
    generate_text(args.sample_context, args.n, args.temperature, args.k_top)

