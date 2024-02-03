import json, os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset

from .dataset import RecipeM
       
class Phi2 ():
    def __init__(self, args) -> None:
        
        self.args = args
        self.save_dir =  os.path.join (self.args.save_dir, self.args.model_type+"_"+self.args.eval_type+".json")
        print ("Results will be save at: ", self.save_dir)
        
        kwargs = {"device_map": "auto"}
        
        self.model =  AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True,trust_remote_code=True, **kwargs)
        self.tokenizer =  AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.temperature = 0.2
        self.num_beams = 1

        self.recipe = RecipeM (args.dataset_dir, partition=args.partition)
            
    def save_results (self, results, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)



    def predict_batch (self):
        self.outputs = {}
        self.done = []
        if (os.path.exists(self.save_dir)):
            self.outputs = json.load (open(self.save_dir))
            self.done = list (self.outputs.keys())
        test_dataloader = DataLoader(self.recipe, batch_size=self.args.batch_size, shuffle=False)
        
        print ("Number of already processed files:", len(self.done))

        for j, batch in tqdm (enumerate(test_dataloader)):
            ids = batch[-1]
            done_counter = 0
            for id in ids:
                if (id in self.done):
                    done_counter= done_counter + 1
            if (done_counter == len(ids)): # skip if whole batch is already done
                continue
            
            batch2 = self.tokenizer(batch[0], truncation=True, return_tensors="pt").to(self.model.device)

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
                input_token_len = len(batch[0][i])
                
                output = self.tokenizer.batch_decode(output_ids[i][input_token_len:].unsqueeze(0), clean_up_tokenization_spaces=False, skip_special_tokens=True)
                out = { "id": batch[-1][i],
                                "gt":  batch[1][i],
                                "pred": output[0].split ("Answer")[-1] ,
                                "prompt": batch[0][i]
                    }
                self.outputs[batch[-1][i]] = out
                if (j%10 == 0):
                    self.save_results (self.outputs, self.save_dir)  
        
    def predict (self):
        self.predict_batch()