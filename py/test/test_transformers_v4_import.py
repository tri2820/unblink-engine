"""Test importing from transformers v4.53-release"""
import sys
import os

# Add transformers-v4.53/src to path relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
transformers_v4_path = os.path.join(script_dir, '..', 'transformers-v4.53', 'src')
sys.path.insert(0, transformers_v4_path)

print("Testing import from transformers v4.53...")
from transformers import AutoModelForCausalLM
import transformers

print(f"Successfully imported transformers version: {transformers.__version__}")
print(f"Transformers location: {transformers.__file__}")
print(f"AutoModelForCausalLM: {AutoModelForCausalLM}")
