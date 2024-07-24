### Title
LLaVA-Chef: A Multi-modal Generative Model for Food Recipes


### Abstract

In the rapidly evolving landscape of online recipe sharing within a globalized context, there has been a notable surge in research towards comprehending and generating food recipes. Recent advancements in large language models (LLMs) like GPT-2 and LLaVA have paved the way for Natural Language Processing (NLP) approaches to delve deeper into various facets of food-related tasks, encompassing ingredient recognition and comprehensive recipe generation. Despite impressive performance and multi-modal adaptability of LLMs, domain-specific training remains paramount for their effective application. This work evaluates existing LLMs for recipe generation. We propose LLaVA-Chef, a novel model trained on a curated dataset of diverse recipe prompts in a multi-stage approach. First, we refine the mapping of visual food image embeddings to the language space. Second, we adapt LLaVA to the food domain by fine-tuning it on relevant recipe data. Third, we utilize diverse prompts to enhance the model's recipe comprehension. Finally, we improve the linguistic quality of  generated recipes by penalizing the model with a custom loss function. LLaVA-Chef demonstrates impressive improvements over pretrained LLMs and prior works. A detailed qualitative analysis reveals that LLaVA-Chef generates more detailed recipes with precise ingredient mentions, compared to extant approaches.

### Download dataset:
- Download dataset from: [Recipe1M](http://im2recipe.csail.mit.edu). Note that dataset may be avaialble publically. 
- Once dataset is downloaded, download images for each recipe from links given with each recipe sample. 
- Discard samples that do not contain at least one image. 
- We have also provided ids of the samples used in our both test sets: $test1k$ and $test50k$ in ```datasets``` folder.
- The prompts used for training and evaluation of the model as also given in ```dataset/prompts.json```

### Run code:

Run the bash file in each folder LLM and LLaVA-Chef as:

    $ cd LLMs 
    $ bash eval.sh
    $ cd LLaVA-Chef
    $ bash eval.sh

Each $eval.sh$ bash file runs calls eval.py to evaluate the model on given inputs setting. 

    python eval.py \
        --model_path  = path to model weights\
        --dataset_dir = path to Recipe1M dataset\
        --save_dir    = path to save generated recipes \
        --eval_type   = "im_t_ing" \
        --model_type  = model_name_for_results_file_name \
        --partition   = "test_1k" \
        --prompt_type = "ing__instruct"\   # What to predict: here ingredient is input to generate recipe instructions 
        --batch_size    1 




```bibtex
@article{mohbat2024llavachef,
  title={LLaVA-Chef: A Multi-modal Generative Model for Food Recipes},
  author={Fnu Mohbat, Mohammed J. Zaki},
  booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM)},
  year      = {2024}
}

```