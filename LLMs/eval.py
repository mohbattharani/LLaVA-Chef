
'''
- Use latest version of transformers

'''


import argparse

from models.instructBLIP import InstructBlip
from models.gpt2 import GTP2
from models.phi2 import Phi2
from models.mistral import Mistral
from models.llama import LLAMA


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/llava-7b-finetune")
    parser.add_argument("--dataset_dir", type=str, default="/data/datasets/Food/Recipe1M/")
    parser.add_argument("--save_dir", type=str, default="results/")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--model_type", type=str, default="llava")
    parser.add_argument("--eval_type", type=str, default="recipe1m")
    parser.add_argument("--partition", type=str, default="test_image")
    parser.add_argument("--prompt_type", type=str, default="i__instruct")

    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()

    model = eval(args.model_type)(args)

    model.predict()
