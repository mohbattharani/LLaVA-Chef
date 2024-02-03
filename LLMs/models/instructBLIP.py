from PIL import Image
import json, os
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from torch.utils.data import DataLoader

from tqdm import tqdm
 

from .dataset import RecipeM

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class InstructBlip ():
    def __init__(self, args) -> None:
        
        self.args = args
        kwargs = {"device_map": "auto"}

        self.save_dir =  os.path.join (self.args.save_dir, f"{args.model_type}_{args.eval_type}_{args.prompt_type}.json")
        
        self.recipe = RecipeM (args.dataset_dir, partition=args.partition, input_type=self.args.eval_type, prompt_type=args.prompt_type)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(args.model_path, low_cpu_mem_usage=True, **kwargs)
        self.processor = InstructBlipProcessor.from_pretrained(args.model_path)
        self.temperature = 0.2
        self.num_beams = 1
        print ("test partition:", args.partition)
        print ("test dataset size:", len(self.recipe) )

        print ("save dir:", self.save_dir)


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
            images = [Image.open(im) for im in batch[2]]

            batch2 = self.processor(images=images, text=batch[0], return_tensors="pt").to(self.model.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **batch2,
                    do_sample=True if self.temperature>0 else False,
                    temperature=self.temperature,
                    num_beams=self.num_beams,
                    max_new_tokens=self.args.max_length,
                    min_length=10,
                    use_cache=True,
                    top_p=0.9,
                    repetition_penalty=1.5
                )

            for i in range (output_ids.shape[0]):
                input_token_len = len(batch[0][i])
                
                output = self.processor.batch_decode(output_ids[i].unsqueeze(0), clean_up_tokenization_spaces=False, skip_special_tokens=True)
                out = { "id": batch[-1][i],
                                "gt":  batch[1][i],
                                "pred": output[0].strip(),#split ("Answer")[-1] ,
                                "prompt": batch[0][i]
                    }
                self.outputs[batch[-1][i]] = out
                if (j%10 == 0):
                    self.save_results (self.outputs, self.save_dir) 

                    break

    def predict (self):
        self.predict_batch()
