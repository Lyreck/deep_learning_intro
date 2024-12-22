## experimenting with Stable Diffusion through HF.

from huggingface_hub import notebook_login
notebook_login() #loading compute resource

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image

experimental_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True) 
experimental_pipe = experimental_pipe.to("cuda")

description_1 = "Peppa pig on Mars"
with autocast("cuda"):
  image_1 = experimental_pipe(description_1).images[0]
  
