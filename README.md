# Stable_Diffusion_Trainer: Text to Image Stable Diffusion model(train from scratch or finetune from a checkpoint)
Contain a single script to train stable diffusion from scratch. Easy to modify with advanced libraries support.

## Key Features

- Custom UNet and VAE architecture
- Resume from checkpoints
- Can be used to finetune
- EMA (Exponential Moving Average) for model stability
- Mixed precision training support
- Integration with TensorBoard and Weights & Biases
- HuggingFace Hub compatibility
- Multi GPU support
- Mixed precision support with accelerate


## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```
and source install of diffusers
```
pip install git+https://github.com/huggingface/diffusers 
```
2. Prepare your dataset
See dataset dataloader documentation from huggingface
You can use local file or some dataset at hugging face hub.(See example implementation at the end.)

3. Run the training script:

```bash

!git clone 
cd

```
To run the training script with all the hyperparameters
```bash

```
