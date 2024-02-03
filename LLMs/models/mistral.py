


from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

import json, os , torch
from tqdm import tqdm

from .dataset import RecipeM

class Mistral ():
    def __init__(self, args) -> None:
        
        self.args = args
        kwargs = {"device_map": "auto"}

        self.save_dir =  os.path.join (self.args.save_dir, f"{args.model_type}_{args.eval_type}_{args.prompt_type}.json")
        
        self.recipe = RecipeM (args.dataset_dir, partition=args.partition, input_type=self.args.eval_type, prompt_type=args.prompt_type)
        self.model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.tokenizer.model_max_length = 512
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.temperature = 0.2
        self.num_beams = 1
        print ("test partition:", args.partition)
        print ("test dataset size:", len(self.recipe) )

        print ("save dir:", self.save_dir)

            
    def save_results (self, results, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


    def predict_batch (self, batch):
        outputs = []
        batch2 = self.tokenizer(batch[0], padding='max_length', truncation=True, return_tensors="pt").to(self.model.device)
        input_len = batch2['input_ids'].shape[1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **batch2,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                num_beams=self.num_beams,
                max_new_tokens=self.args.max_length,
                use_cache=True
            )

        for i in range (output_ids.shape[0]):
            output = self.tokenizer.batch_decode(output_ids[i][input_len:].unsqueeze(0), clean_up_tokenization_spaces=False, skip_special_tokens=True)
            outputs.append ({ "id": batch[-1][i],
                            "gt":  batch[1][i],
                            "pred": output[0].split ("\n\n")[-1],
                            "prompt": batch[0][i]
                            })

        return outputs

    def predict (self):
        self.outputs = []
        self.done = []
        if (os.path.exists(self.save_dir)):
            self.outputs = json.load (open(self.save_dir))
            self.done = [o['id'] for o in self.outputs]
        test_dataloader = DataLoader(self.recipe, batch_size=self.args.batch_size, shuffle=False)
        
        batch_id = 0

        for batch in tqdm (test_dataloader):

            batch_id += 1
            ids = batch[-1]
            done_counter = 0
            for id in ids:
                if (id in self.done):
                    done_counter= done_counter + 1
            if (done_counter == len(ids)): # skip if whole batch is already done
                continue
        
            outputs = self.predict_batch (batch)

            for o in outputs:
                self.outputs.append (o)

            #if (batch_id%10 == 0):
            self.save_results (self.outputs, self.save_dir)  
  