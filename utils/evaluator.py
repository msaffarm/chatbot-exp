import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
import pickle

class Evaluator():

    def read_data_from_files(self,target_file,response_file,diagid_file,entity_file):
        self.responses = []
        self.targets = []
        self.diagids = []
        self.entity_info = []
        with open(target_file,'r') as f:
            for line in f:
                self.targets.append(line.strip())
        
        with open(response_file,'r') as f:
            for line in f:
                self.responses.append(line.strip())
        
        with open(diagid_file, 'r') as f:
            for line in f:
                self.diagids.append(line.strip())

        with open(entity_file, 'rb') as f:
            self.entity_info = pickle.load(f)

        assert len(self.targets)==len(self.responses)==len(self.diagids)

    def get_eval_stats(self, mode='soft'):

        # get per turn accuracy
        per_turn_acc = self.get_pet_turn_acc(self.targets,self.responses,mode=mode)

        # get per diag accuracy
        per_diag_acc = self.get_per_dialogue_acc(self.targets, self.responses, self.diagids,mode=mode)

        # get nltk bleu score
        bleu = self.get_bleu_score_corpus(self.targets,self.responses)

        # get entity f score
        entf = self.get_f1_entity_score(self.targets,self.responses)

        report = "Per-turn Acc={}\nPer-Diag Acc={}\nBLEU={}\nf1-score={}\n".format(per_turn_acc*100,per_diag_acc*100,bleu*100,entf*100)

        return report

    def response_target_iter(self):
        for i in range(len(responses)):
            yield self.responses[i],self.targets[i]

    def get_pet_turn_acc(self,target,response,mode='soft'):
        accuracy = 0.0
        if isinstance(target,list) and isinstance(response,list):
            accum_acc = 0.0
            for idx,(t,r) in enumerate(zip(target,response)):
                accum_acc += self._get_turn_acc(t,r,mode=mode)
            accuracy = accum_acc/(idx+1)
        
        elif isinstance(target,str) and isinstance(response,str):
            accuracy = self._get_turn_acc(target,response,mode=mode)
        
        return accuracy
    
    @staticmethod
    def _get_turn_acc(target_sent,response_sent,mode='soft'):
        target_tokens = target_sent.strip().split(" ")
        response_tokens = response_sent.strip().split(" ")

        if mode=='soft':
            # if all(target_token in response_tokens for target_token in target_tokens):
            #     return 1.0
            if all(response_token in target_tokens for response_token in response_tokens):
                return 1.0
            else:
                return 0.0
        elif mode=='hard':
            min_len = min(len(response_tokens), len(target_tokens))
            if all(response_tokens[idx]==target_tokens[idx] for idx in range(min_len)):
                return 1.0
            else:
                return 0.0


    def get_per_dialogue_acc(self,target,response,diagids,mode='soft'):
        
        from collections import defaultdict
        did2idx = defaultdict(list)
        for idx,did in enumerate(diagids):
            did2idx[did].append(idx)

        num_of_diags = len(set(diagids))
        diag_accs = 0.0
        for did, idxlist in did2idx.items():
            diag_acc = 0.0
            for idx in idxlist:
                diag_acc += self._get_turn_acc(target[idx],response[idx],mode=mode)
            if int(diag_acc)==len(idxlist):
                diag_accs += 1.0
        
        return diag_accs / num_of_diags


    def get_f1_entity_score(self,target,response):
        # get response and target entity info
        target_entity_info = self.get_entity_info(target)
        response_entity_info = self.get_entity_info(response)

        assert len(target_entity_info)==len(response_entity_info)
        
        # compute micro-averaged f1 score assuming the target is the gold data
        tpsum,fpsum,fnsum = 0.0,0.0,0.0
        for idx in range(len(target_entity_info)):
            gold_entities = target_entity_info[idx]['entities']
            response_entities = response_entity_info[idx]['entities']
            if not gold_entities:
                continue
            # compute True Positive (TP)
            tp,fp,fn = self.calculate_entity_stats(gold_entities,response_entities)
            tpsum += tp
            fpsum += fp
            fnsum += fn
        
        # calculate micoroaveraged f1 score
        mic_precision = (tpsum)/(tpsum+fpsum)
        mic_recall = (tpsum)/(tpsum+fnsum)

        return (2.0 * (mic_precision * mic_recall))/(mic_precision + mic_recall)

    
    @staticmethod
    def calculate_entity_stats(gold,response):
        gold_set = set([(ent_val_pair['entity'],ent_val_pair['value']) for ent_val_pair in gold])
        response_set = set([(ent_val_pair['entity'],ent_val_pair['value']) for ent_val_pair in response])
        # calculate TP
        tp = len(gold_set.intersection(response_set))
        fp = len(response_set) - tp
        fn = len(gold_set) - tp

        return tp,fp,fn        


    def get_entity_info(self, turns):

        turn_entity_info = []
        for idx, turn in enumerate(turns):
            info = {
                'text': turn,
                'entities':[]
            }
            for ent, possible_vals in self.entity_info['entity_vals'].items():
                for val in possible_vals:
                    if val in turn:
                        info['entities'].append({
                            'entity': ent,
                            'value':val
                        })
            turn_entity_info.append(info)

        return turn_entity_info



    def get_bleu_score_sentence(self,target,response):
        reference = [target.strip().split()]
        candidate = response.strip().split()
        score = sentence_bleu(reference, candidate)

        return score

    def get_bleu_score_corpus(self,target,response):   
        
        references = [[t.split()] for t in target]
        candidates = [r.split() for r in response]
        # print(references)
        # print(candidates)
        score = corpus_bleu(references, candidates)
        
        return score


def main():
    eval = Evaluator()
    target =["what type of food and price range should i look for ?","hi there , i"]
    response = ["what type of food and price range should i look for ?","hi there , i"]
    acc = eval.get_bleu_score_corpus(target,response)
    print(acc)


if __name__ == '__main__':
    main()