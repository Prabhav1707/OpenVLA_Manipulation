from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import torch
import numpy as np

print("Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True
)

model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True
)

print("Model loaded!")

# Create a dummy image — red cube on white background
# In the real project this will come from PyBullet's camera
image = Image.fromarray(
    np.uint8(
        np.concatenate([
            np.ones((224, 224, 1)) * 220,  # R channel — red
            np.ones((224, 224, 1)) * 50,   # G channel — low
            np.ones((224, 224, 1)) * 50,   # B channel — low
        ], axis=2)
    )
)

# The instruction
instruction = "pick up the red cube"

# Format prompt exactly as OpenVLA expects
prompt = f"In: What action should the robot take to {instruction}?\nOut:"

# Prepare inputs
print(f"\nRunning inference...")
print(f"Instruction: '{instruction}'")

inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

# Run inference — predict 7 action tokens
with torch.no_grad():
    action = model.predict_action(
        **inputs,
        unnorm_key="bridge_orig",
        do_sample=False
    )

print(f"\nPredicted action:")
print(f"  Delta X (forward/back) : {action[0]:.4f}")
print(f"  Delta Y (left/right)   : {action[1]:.4f}")
print(f"  Delta Z (up/down)      : {action[2]:.4f}")
print(f"  Delta Roll             : {action[3]:.4f}")
print(f"  Delta Pitch            : {action[4]:.4f}")
print(f"  Delta Yaw              : {action[5]:.4f}")
print(f"  Gripper (0=open,1=close): {action[6]:.4f}")