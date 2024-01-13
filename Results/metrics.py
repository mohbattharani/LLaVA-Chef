import json
import numpy as np 

# ====================================================================================
# ====================================================================================

def save_results (results, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
            
def remove_duplicate_sentences(paragraph):
    if isinstance(paragraph, list):
        paragraph = paragraph[0]
    
    paragraph = steps_to_para (paragraph)
    
    sentences = [s.strip() for s in paragraph.split('.') if s.strip()]  # Split the paragraph into sentences

    unique_sentences = set(sentences)  # Remove duplicates by converting to a set
    result_paragraph = '. '.join(unique_sentences) + '.'  # Rejoin the unique sentences into a paragraph

    return result_paragraph

def steps_to_para (steps):
    sentences = [s.strip() for s in steps.split('\n') if s.strip()]  # Split the paragraph into sentences

    unique_sentences = set(sentences)  # Remove duplicates by converting to a set
    paragraph = '. '.join(unique_sentences) # Rejoin the unique sentences into a paragraph
    
    return paragraph

# ====================================================================================
# ====================================================================================

def compute_metrics_hf (file, N = -1):
    metric_hf = Metrics_Huggingface ()

    results = json.load (open (file))
    if not isinstance(results, list ):    
        results = [results[k] for k in results.keys()]
        
    if (N > 0):
        results = results[:N]
    img_ids = []
    preds = []
    anns = []
    print (f"computing metrics on {len(results)} samples.")
    
    for r in results:
        img_ids.append (r['id'])
        pred = remove_duplicate_sentences (r['pred'])
        gt = remove_duplicate_sentences (r['gt'])
        
        preds.append(pred)
        anns.append(gt)

    return metric_hf.compute (anns, preds)
    

def compute_metrics (file, N = -1):
    results = json.load (open (file))
    if not isinstance(results, list ):    
        results = [results[k] for k in results.keys()]
    if (N > 0):
        results = results[:N]
        
    img_ids = []
    preds = {}
    anns = {}
    print (f"computing metrics on {len(results)} samples.")
    
    for r in results:
        img_ids.append (r['id'])
        pred = remove_duplicate_sentences (r['pred'])
        gt = remove_duplicate_sentences (r['gt'])
        
        preds[r['id']] = [{'id': r['id'], "image_id": r['id'], 'caption': pred}]
        anns[r['id']] = [{'id': r['id'], "image_id": r['id'], 'caption': gt}]

    coco_eval = COCOEvalCap (anns, preds, img_ids)
    coco_eval.evaluate()

    return coco_eval.eval
    

# ====================================================================================
# ====================================================================================

from pycocotools.coco import COCO
#from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import sys

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice


class COCOEvalCap:
    def __init__(self, coco, cocoRes, imgIds):
        self.coco = coco
        self.cocoRes = cocoRes
        self.eval = {}
        self.imgIds = imgIds

    def evaluate(self):
        imgIds = self.imgIds
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco[imgId]
            res[imgId] = self.cocoRes[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        #print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        #print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            #(Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            #print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    #print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                #print("%s: %0.3f"%(method, score))
                
    def setEval(self, score, method):
        self.eval[method] = round (score, 4) 


# ====================================================================================
# ====================================================================================



import evaluate

class Metrics_Huggingface ():
    def __init__ (self, model_id = "gpt2"):
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load("bleu")
        self.sacrebleu = evaluate.load("sacrebleu")
        self.bertscore = evaluate.load("bertscore")
        self.perplexity =  evaluate.load("perplexity", module_type="metric")
        self.model_id = model_id
    
    def compute (self, labels, preds):
        scores = {}
        #preds = [preds[k][0] for k in preds.keys()]
        #labels = [labels[k] for k in labels.keys()]

        results = self.rouge.compute(predictions=preds, references=labels)
        for k in results.keys():
            scores[k] = round (results[k], 4)
        
        results = self.bleu.compute(predictions=preds, references=labels)
        scores['bleu'] = round (results['bleu'], 4)

        results = self.sacrebleu.compute(predictions=preds, references=labels)
        scores['sacrebleu'] = round (results['score'], 4)

        results = self.perplexity.compute(model_id=self.model_id , add_start_token=True, predictions=preds)
        scores ['ppl_pred'] = results['mean_perplexity']
        results = self.perplexity.compute(model_id=self.model_id , add_start_token=True, predictions=labels)
        scores ['ppl_gt'] = results['mean_perplexity']
        
        #results = self.bertscore.compute(predictions=preds, references=labels, lang="en")
        #scores['bertscore_p'] = round (np.mean(results ['precision']), 4)
        #scores['bertscore_r'] = round (np.mean(results ['recall']), 4)
        #scores['bertscore_f1'] = round (np.mean(results ['f1']), 4)
        
        return scores
    


# ====================================================================================
# ====================================================================================

