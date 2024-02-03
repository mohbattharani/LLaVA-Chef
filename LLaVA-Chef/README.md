# LLaVA-Chef

This code evaluates LLaVA and LLaVA-Chef models. 


args: 

'''
    --model_path               # Path to model weights
    --dataset_dir              # Path to dataset
    --save_dir                 # Path where results should be saved
    --eval_type                # Decides input/output 
    --model_type               # Which model to evaluate - model name
    --partition                # Dataset partition on which model needs to be evaluated 
    --prompt_type              # Input/Ouput relation key used for dataset
    --start=0                  # Start index for list of data samples
    --end=-1                   # Last index  for list of data samples of the dataset, -1 means all sample of the partition

''''

