import tiktoken
import multiprocessing
from datasets import load_dataset


tokenizer = tiktoken.get_encoding("gpt2")

batch_size = 1000  # Number of text samples processed at a time
num_workers = multiprocessing.cpu_count()  # Use all available CPU cores
eot = tokenizer._special_tokens['<|endoftext|>']

def tokenize_batch(text_list):
    """Tokenizes a batch of text samples using tiktoken."""
    return [tokenizer.encode_ordinary(text) + [eot] for text in text_list]

def save_tokens(output_file, split):
  fw = load_dataset("roneneldan/TinyStories", split=split)  # Small dataset
  token_count = 0
  with open(output_file, "w", encoding="utf-8") as f:
      batch = []  # Store batches for processing

      for sample in fw:
          batch.append(sample["text"])  # Collect text samples

          # Process batch when full
          if len(batch) >= batch_size:
              with multiprocessing.Pool(num_workers) as pool:
                  tokenized_batches = pool.map(tokenize_batch, [batch])

              for tokenized in tokenized_batches[0]:  
                  token_count += len(tokenized)  
                  f.write(" ".join(map(str, tokenized)) + "\n")  # Save tokens as space-separated numbers

              batch.clear()  
              
  print(f"Collected {token_count} tokens for the {split} set and saved to {output_file}")

save_tokens("train_tokens.txt", 'train')
save_tokens("val_tokens.txt", 'validation')


