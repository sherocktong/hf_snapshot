import torch
from safetensors.torch import load_file
import argparse


def convert_safetensors_to_ckpt(safetensors_path, output_ckpt_path):
    # Load the SafeTensors file
    state_dict = load_file(safetensors_path)

    # Save as checkpoint file
    torch.save(state_dict, output_ckpt_path)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .safetensors to .ckpt format")
    parser.add_argument("safetensors_path", type=str, help="Path to the input .safetensors file")
    parser.add_argument("ckpt_path", type=str, help="Path to save the output .ckpt file")
    args = parser.parse_args()
    convert_safetensors_to_ckpt(args.safetensors_path, args.ckpt_path)
