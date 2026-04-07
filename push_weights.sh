#!/bin/bash
echo "Pushing model weights to Hugging Face..."
python -c "
from huggingface_hub import HfApi
import sys

# Replace this string with your actual Hugging Face username!
HF_USERNAME = 'YOUR_HF_USERNAME' 

if HF_USERNAME == 'YOUR_HF_USERNAME':
    print('Error: Please edit push_weights.sh and put your actual Hugging Face username in!')
    sys.exit(1)

api = HfApi()
api.upload_folder(
    folder_path='lora_weights/',
    repo_id=f'{HF_USERNAME}/llm-diffusion-lora-weights',
    repo_type='model'
)
print('Done. Weights safely backed up.')"
