

from torch.utils.data import Dataset
import json, random, os



class RecipeM (Dataset):
    def __init__ (self, dataset_dir, partition = "train", input_type="im_t_ing", prompt_type = "ing_i_t__instruct"):
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

    def __getitem__(self, idx):
        chat = self.conversation (idx) 
        q = chat['conversations'][0]['value']
        t = chat['conversations'][1]['value']
        #image = Image.open(chat['image'])

        return q, t, chat['image'], chat['id']
    
