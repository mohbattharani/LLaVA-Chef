from .model import Model
from PIL import Image
import json, random, os
from pprint import pprint
import torch, json
#from llava.data.data import RecipeM
from tqdm import tqdm

import os 
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class RecipeM ():
    def __init__ (self, dataset_dir, partition = "train", input_type="im_t_ing", prompt_type = None):
        self.dataset_dir = dataset_dir
        self.partition = partition
        self.image_dir = os.path.join (dataset_dir, "images", partition.split ("_")[0])
        self.data = json.load (open(os.path.join(dataset_dir, f"{partition}.json")))
        self.ids = list(self.data.keys())
        self.prompts = json.load (open(os.path.join (dataset_dir, "prompts.json")))
        self.prompt_keys = list(self.prompts.keys())
        self.prompt_type = prompt_type
        self.input_type = input_type
        self.targets = {
                 "ing_i_t__instruct": "instructions: <instructions>",
                 "i__instruct": "instructions: <instructions>",
                 "i_ing__t": "title: <title>",
                 "i__ing": "ingredients: <ingredients>",
                 "i_t_recipe": "title: <title>\ningredients: <ingredients>\n instructions: <instructions>", 
            }

    def __len__(self):
        return len(self.ids)

    def prompt_input_mapping (self, key):
        select = 'image'
        
        if (key in ['ing_i_t__instruct']):
            select = ['ingredients'] + random.sample (['image', 'title'], random.choice ([0,1,2]))
        elif key in ['i_ing__t']: # must select at least one attr
            select = random.sample (['image', 'ingredients'], random.choice ([1,2]))
        elif key in ['i_t_recipe']:
            select = random.sample (['image', 'title'], random.choice ([0,1,2]))
    
        return select
    
    def get_sample (self, idx):
        id = self.ids[idx]
        sample = self.data[id]
        if ("image" in sample.keys()):
            images = []
            for im in sample['image']:
                im = os.path.join (self.image_dir, im)
                if os.path.isfile(im):
                    images.append(im)
            if (len(images)>0):        
                sample['image_path']= random.sample (images, 1)[0]
            else:
                sample['image_path']= None
            
        return sample

    def conversation (self, idx):

        sample = self.get_sample (idx)
        chat = { 
            "id": sample['id'],
            "conversations": []
        }

        
        if (self.prompt_type is None):
            selected_prompt_keys = random.choices(self.prompt_keys)[0]
        else:
            selected_prompt_keys = self.prompt_type
        q = random.sample (self.prompts[selected_prompt_keys], 1)[0]

        target = self.targets [selected_prompt_keys]

        if ("<title>" in target):
            target = target.replace ("<title>", sample['title'])
        if ("<ingredients>" in target):
            target = target.replace ("<ingredients>", "\n".join (sample['ingredients']))
        if ("<instructions>" in target):
            instructions = '\n'.join(f'{i + 1}. {line}' for i, line in enumerate(sample['instructions']))
            target = target.replace ("<instructions>", instructions)
        
        if ("t" in self.input_type.split ("_")):
            q = q.replace("<name>", sample['title']) if "<name>" in q else q + "The food is:" + sample['title'] 
        else: 
            q = q.replace("<name>", random.choice (["food", "dish"])) 

        if ("ing" in self.input_type.split ("_")):
            ingredients = "\n".join (sample['ingredients'])
            q = q.replace("<ingredients>", ingredients) if "<ingredients>" in q else q + "Use ingredients:" + ingredients
        else:
            q = q.replace("<ingredients>", "") 

        if ("im" in self.input_type.split ("_")):
            if "image_path" in sample.keys():
                if (os.path.isfile(sample['image_path'])):
                    chat['image'] = sample['image_path']
        else:
            chat['image'] = "/data/datasets/Food/Recipe1M/images/empty.jpg"


        chat ["conversations"].append(
                    {
                        "from": "human",
                        "value": q  
                    })

        chat ["conversations"].append(
                    {
                        "from": "gpt",
                        "value": target
                    })
        
        return chat  


class EvalModel ():
    def __init__(self, args) -> None:
        
        self.args = args
        self.save_dir =  os.path.join (self.args.save_dir, f"{args.model_type}_{args.eval_type}_{args.prompt_type}.json")
        
        if (True):
                
            self.recipe = RecipeM (args.dataset_dir, partition=args.partition, input_type=self.args.eval_type, prompt_type=args.prompt_type)
            self.model = Model ( args.model_path)

            if (not args.mm_project_path is None):
                mm_projector_weights = torch.load(args.mm_project_path)
                mm_projector_weights = {k.split(".")[-1]: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                self.model.model.model.mm_projector.load_state_dict(mm_projector_weights, strict=False)

        
        print ("test partition:", args.partition)
        print ("test dataset size:", len(self.recipe) )

        print ("save dir:", self.save_dir)



    def save_results (self, results, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


    def predict_loop (self):
            
        for i in tqdm (range (self.args.start, self.args.end)):

            chat = self.recipe.conversation (i)
            
            # if id was processed, then do not process it again
            if chat['id'] in self.done:
                continue
            image_path = chat['image'] if "image" in chat.keys() else None
            
            if (image_path is None ): # process foods with images 
                print ("No image found. Skip this sample.") 
                continue
            
            q = chat['conversations'][0]['value']
            t = chat['conversations'][1]['value']

            output = self.model.step (q, [image_path])
            
            self.outputs[chat['id']] = {   
                                    "id": chat['id'],
                                    "gt": t,
                                    "pred": output,
                                    "q" : q
                                  }
            
            if (i%10 == 0):
                self.save_results (self.outputs, self.save_dir)       

        return self.outputs

    def predict (self):
        self.outputs = {}
        self.done = []
        if (self.args.end<1):
           self.args.end = len(self.recipe)
        self.args.end = min (self.args.end, len(self.recipe))
        print (f"processeing from {self.args.start} to {self.args.end} index of dataset size: {len(self.recipe)}")


        if (self.args.start>self.args.end):
            return
            
        if (os.path.exists(self.save_dir)):
            self.outputs = json.load (open(self.save_dir))
            self.done = list (self.outputs.keys())


        print ("Already processed samples:", len(self.done))
        
        results = self.predict_loop ()
        self.save_results (results, self.save_dir)  

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/llava-7b-finetune")
    parser.add_argument("--mm_project_path", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default="/data/datasets/Food/Recipe1M/")
    parser.add_argument("--queries_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="results/")
    parser.add_argument("--model_type", type=str, default="llava")
    parser.add_argument("--eval_type", type=str, default="recipe1m")
    parser.add_argument("--partition", type=str, default="test_image")
    parser.add_argument("--prompt_type", type=str, default="i__instruct")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=-1)

    args = parser.parse_args()

    eval = Eval (args)
    eval.predict_save()
'''