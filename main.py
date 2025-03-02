import torch
from models.gpt import GPT
from dataset.data_module import DataModule
from config import GPTConfig, TrainingConfig
from trainer import Trainer
import time

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the configuration objects
gpt_config = GPTConfig()
training_config = TrainingConfig()

# Create the data module
dm = DataModule(training_config.batch_size, gpt_config.context_length)

# Create the data loaders
train_dataloader = dm.train_dataloader()
val_dataloader = dm.val_dataloader()

# Enable TF32 for faster training
torch.set_float32_matmul_precision('high') 

# Create the GPT model
model = GPT(gpt_config)
model.to(device) # Move the model to the device

# Compile the model for faster training
model = torch.compile(model) 

# Set the sample context
sample_context = "Once a cat sees a dog and asks,"

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
start_time = time.time()
trainer.train()
# end_time = time.time()
# training_time_minutes = (end_time - start_time)/60
# print(f'\n Training completed in {training_time_minutes:.2f} minutes.')
# print(f' Best Epoch is {trainer.best_epoch} with validation loss {trainer.best_val_loss:.3f}')


### in GP2 positional embedding are just parapmeters and trained from scratch like other parameters
### GPT 2 is a deoder only transformer
### GPT 2 use no bias for the final projection
## IN gpt2 THE NORMALIZATION IS BEFORE  (compared the attention paper)
#### Attention: communication operation  [reduce]  communicate
#### FF: for every single token individually. nothing between tokens . [map]  think individually

### for GELU the tanH method is an appriximation and GPT 2 USED TAN H
#### use att.masked_fill 
#### use **-0.5

###sd[k].copy_(sd_hf[k].t())   or sd[key] = sd_hf[key].clone()   | probably use torch.no_grad   and he just change the dictionary and did not load it
### x.to(device) is wrong becase x is not stateful  (it is just a tensor)


### in (before)attention:  wte and lm_head the same value . note that the shapes are: [50257, 768] for both of them 
### similiray betwee ntoken, nearby token embeddigns . an vice versa   we expect this     weight tying
### 1.11.54

######################### 1.14.27 initialization of weights and bias 

######################### 1.19.0 residula probelm

## 1.34.15    torch sync 
## 1.36.40     token/sec
### utilizaion > 60% is very well
### inrease speed: enab;e tf32: 1.37.00'   3x
#### 2:   1.43.25
#333. 1.48   compile  ### 2.3x impoervement
###  44. flash attention 2.0.40  2.05.21      27 percent

#### 5. uge number    4 percenrt


######################### gpt 3 :   2.20 adamw
######################### 2.20 grad clip
######################### 2.4.20 schedular
######################### 2.29 weight decay
######## 2.38  grad accmulation
######## 2.37 batch size

######## 2.33   fused adam






### 2^19 = 542,288 tokens per step 
### dataset: 10 Bilion tokens 
### max_step = 10e9 / 2e19 = 19073 steps
### in gpt 3 warm up over 375e6 tokens so 375e6/2e19  steps for warm up = 715 


# batch size should be 542,288 which is 2e19, ~0.5M in nmber of tokes


#B = 16 T = 1024 so grad_acc = 32

'''
My lessons:
1. before wiegh tying, the training started from 9 but after weight tying it started from 114 and did no get to the best
brefore. however, initilizaion o the mbed with normal, solve this problem. it seems that the FC layers should not be initialize with
uniform distritubution (which embedding layers init with).
 In fact I have a unifroms dist from -0.3 to 0.3. which has the std of 1.2
 xavier = mean = 0  std = 1/(sqrt(n))    n:inoming features



'''
