import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu


class Evaluator():

    def read_data_from_files(self,target_file,response_file):
        self.responses = []
        self.targets = []
        
        with open(target_file,'r') as f:
            for line in f:
                self.targets.append(line.strip())
        
        with open(response_file,'r') as f:
            for line in f:
                self.responses.append(line.strip())

        assert len(self.targets)==len(self.responses)

    def get_eval_stats(self):
        # get per turn accuracy
        per_turn_acc = self.get_pet_turn_acc(self.targets,self.responses)

        # get nltk bleu score
        bleu = self.get_bleu_score_corpus(self.targets,self.responses)

        report = "Per-turn Acc={}\nBLEU={}\n".format(per_turn_acc*100,bleu*100)

        return report

    def response_target_iter(self):
        for i in range(len(responses)):
            yield self.responses[i],self.targets[i]

    def get_pet_turn_acc(self,target,response):
        accuracy = 0.0
        if isinstance(target,list) and isinstance(response,list):
            accum_acc = 0.0
            for idx,(t,r) in enumerate(zip(target,response)):
                accum_acc += self._get_turn_acc(t,r)
            accuracy = accum_acc/(idx+1)
        
        elif isinstance(target,str) and isinstance(response,str):
            accuracy = self._get_turn_acc(target,response)
        
        return accuracy
    
    @staticmethod
    def _get_turn_acc(target_sent,response_sent):
        target_tokens = target_sent.strip().split(" ")
        response_tokens = response_sent.strip().split(" ")

        if all(target_token in response_tokens for target_token in target_tokens):
            return 1.0
        else:
            return 0.0
    


    def get_per_dialogue_acc(self,target_diag,response_diag):
        pass
    
    def get_f1_entity_score(self,target,response):
        pass

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