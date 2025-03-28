import torch
from models.gpt import GPT
from dataset.data_module import DataModule
from config import GPTConfig, TrainingConfig
from trainer import Trainer
import time



# Load the tiny stories dataset 
def load_tiny_stories():
    tokens_train = []
    tokens_val = []

    with open("train_tokens.txt", "r", encoding="utf-8") as f:
        for line in f:
            # Convert each space-separated token back into an integer list and append to loaded_tokens
            token_list = list(map(int, line.strip().split()))
            tokens_train.extend(token_list)

    with open("val_tokens.txt", "r", encoding="utf-8") as f:
        for line in f:
            # Convert each space-separated token back into an integer list and append to loaded_tokens
            token_list = list(map(int, line.strip().split()))
            tokens_val.extend(token_list)
    return tokens_train, tokens_val



def main():
    """ Main function to set up and train the GPT model. """
    
    # Load dataset
    tokens_train, tokens_val = load_tiny_stories()

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}...')

    # Initialize configurations
    gpt_config = GPTConfig()
    training_config = TrainingConfig()


    # Initialize Data Module and Data Loaders
    dm = DataModule(training_config.batch_size, gpt_config.context_length, tokens_train, tokens_val)
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    # Enable TF32 for faster training on Ampere GPUs
    torch.set_float32_matmul_precision('high') 

    # Create the GPT model
    model = GPT(gpt_config)
    model.to(device) # Move the model to the device

    # Compile the model for faster training
    model = torch.compile(model) 

    # Set the sample context
    sample_context = "Once, a cat sees a dog and says, 'Hi puppy! How's life going?' "

    # Create the trainer
    trainer = Trainer(
        tokenizer = dm.tokenizer,
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader,
        model = model,
        config = training_config,
        device = device,
        sample_context = sample_context
    )
    print('Start training')
    # Start training
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_time_minutes = (end_time - start_time)/60
    print(f'\n Training completed in {training_time_minutes:.2f} minutes.')

if __name__ == "__main__":
    main()



