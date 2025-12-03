from diffusers import WanTransformer3DModel
from peft import LoraConfig, get_peft_model
import torch

# 1. Load the Base Model
model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
transformer = WanTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)

# 2. Define LoRA Configuration
# Adjust 'target_modules' based on Step 1 if the names differ.
lora_config = LoraConfig(
    r=16,                       # Rank
    lora_alpha=32,              # Alpha scaling
    target_modules=[
        "to_q", "to_k", "to_v", # Attention projections
        "to_out.0"              # Output projection
    ],
    lora_dropout=0.05,
    bias="none",
)

# 3. Inject LoRA Adapters
# This wraps the original model and makes only the LoRA layers trainable
peft_model = get_peft_model(transformer, lora_config)

# 4. Verify Trainable Parameters
peft_model.print_trainable_parameters()
# Output example: "trainable params: 10,485,760 || all params: 1,300,000,000 || trainable%: 0.8%"

# 5. (Optional) Save the Adapter after training
# peft_model.save_pretrained("./wan-lora-adapter")
