import numpy as np

class Evaluator():

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



def main():
    eval = Evaluator()
    target =["what type of food and price range should i look for ?"]
    response = ["what type of food and price range should i look for"]
    acc = eval.get_pet_turn_acc(target,response)
    print(acc)


if __name__ == '__main__':
    main()