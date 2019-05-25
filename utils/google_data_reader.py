import json
import os
from collections import Counter, OrderedDict,defaultdict

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CURRENT_DIR,'../data')
import spacy
nlp = spacy.load("en_core_web_lg")

class StoryFlag():
    utter = "User utterance (annotation)"
    sys_utter = "System utterance (actions taken after user utterance)"
    both = "Both User and System uttered"
    action = "Action by system"
    annotation = "Annotation"

STORY_FLAGS = StoryFlag()

class SysActionMapper():

    def __init__(self):
        self._slots = [""]

    action_map = {
        "REQUEST":1
    }

class GoogleDataReader(object):

    def __init__(self,path_list=None):
        self._data = []
        if path_list:
            for path in path_list:
                self._data += self.read_json_file(path)
        
        self.meta = {}
        self.meta["user_inents"] = Counter()
        self.meta["user_inents"] = Counter()
        self._user_intents = Counter()
        self._sys_intents = Counter()
        self._solo_intent = 0

    @staticmethod
    def read_json_file(path):
        path = os.path.join(DATA_DIR,path)
        return json.load(open(path))
        
    @property
    def data(self):
        return self._data
    
    def read_data_from(self,file_path):
        self._data = self.read_json_file(file_path)

    def create_rasa_nlu_dict(self):
        exmaples = []
        for turn in self.turn_iter():
            if "user_acts" in turn:
                if self._is_rasa_nlu_compatible(turn,"user"):
                    exmaples.append(self._create_rasa_nlu_example(turn,sytem_or_user="user"))
            # if "system_acts" in turn:
            #     if self._is_rasa_nlu_compatible(turn,"system"):
            #         exmaples.append(self._create_rasa_nlu_example(turn,sytem_or_user="system"))
                
        return exmaples


    def _create_rasa_nlu_example(self,turn,sytem_or_user="user"):
        example = OrderedDict()
        if sytem_or_user=="user":
            utter,act = turn["user_utterance"],turn["user_acts"]
        elif sytem_or_user=="system":
            utter,act = turn["system_utterance"],turn["system_acts"]
        example["text"] = utter["text"]
        example["intent"] = act[0]["type"]
        # adding entities
        entities = []
        text = utter["text"]
        tokens = utter["tokens"]
        for slot in utter["slots"]:
            entity = OrderedDict()
            
            # convert indices!!!
            slot_tokens = tokens[slot["start"]:slot["exclusive_end"]]
            start = text.find(slot_tokens[0])
            end = text.find(slot_tokens[-1]) + len(slot_tokens[-1])
            entity["start"] = start
            entity["end"] = end
            entity["value"] = text[start:end]
            entity["entity"] = slot["slot"]
            entities.append(entity)
        
        example["entities"] = entities
        return example


    def create_stories(self):
        stories = []
        for diag in self.diag_iter():
            stories.append(self._create_rasa_story(diag))         
        return stories

    def _create_rasa_story(self,diag):
        story = []
        for turn in diag["turns"]:
            # user_or_sys = self._user_or_sys(turn)
            self._add_annotations(story,turn)            

        return story


    def _add_annotations(self,story,turn):
        
        if "system_acts" in turn:
            sys_utter,sys_act = turn["system_utterance"],turn["system_acts"]
            
        
        if "acts" in turn:
            pass
        

    def _user_or_sys(self,turn):
        if "system_acts" in turn and "acts" in turn:
            return STORY_FLAGS.both
        elif "acts" in turn:
            return STORY_FLAGS.utter
        elif "system_acts" in turn:
            return STORY_FLAGS.sys_utter

    def get_intents(self):
        for turn in self.turn_iter():
            if "user_acts" in turn:
                for ua in turn["user_acts"]:
                    self._user_intents.update({ua["type"]:1})
                if len(ua)==1:
                    self._solo_intent += 1
            if "system_acts" in turn:
                for ua in turn["system_acts"]:
                    self._sys_intents.update({ua["type"]:1})
        return "User acts:\n {}\n\n System acts:\n {} with {} solo intents".\
        format(self._user_intents,self._sys_intents,self._solo_intent)

    def _get_slot_value_map(self,utter):
        svmap = OrderedDict()
        tokens = utter["tokens"]
        for slot in utter["slots"]:
            val = " ".join(tokens[slot["start"]:slot["exclusive_end"]])
            svmap[slot["slot"]] = val
        return svmap

    def diag_iter(self):
        for dialogue in self.data:
            yield dialogue

    def turn_iter(self):
        for dialogue in self.data:
            for turn in dialogue["turns"]:
                yield turn

    def _is_rasa_nlu_compatible(self,turn,speaker):
        is_compatible=True
        if speaker=="user":
            type_set = set([t["type"] for t in turn["user_acts"]])
            if len(type_set)!= 1:
                is_compatible = False
        
        else:
            type_set = set([t["type"] for t in turn["system_acts"]])
            if type_set!=set(["REQUEST"]):
                is_compatible = False

        return is_compatible

    def stats(self):
        nlu_turns,num_turns = 0,0
        for turn in self.turn_iter():
            num_turns +=1
            if self._is_rasa_nlu_compatible(turn,speaker="user"):
                nlu_turns +=1         
        result = "Totol turns={} with {} nlu compatible turns".format(num_turns,nlu_turns)
        return result

    def __str__(self):
        return self.stats()

    def nlu_to_json(self,nlu_data,path):
        rasa_nlu_data = OrderedDict()
        rasa_nlu_data["rasa_nlu_data"] = dict(common_examples=nlu_data,entity_examples=[],intent_examples=[])
        res = json.dumps(rasa_nlu_data,indent=2)
        with open(path,'w') as f:
            f.write(res)

    def write_stories(self,path):
        pass

    def get_token_dict(self):
        token_dict = {}
        token_dict["regular_tokens"] = set()
        token_dict["entity_tokens"] = defaultdict(set)
        counter = 0
        for turn in self.turn_iter():
            counter += 1
            # diagloue_state = {slot:val for slot,val in l.items() for l in turn["dialogue_token"]}
            if "user_utterance" in turn:
                all_tokens = turn["user_utterance"]["tokens"]
                entity_token_indices = set()
                # extract entity tokens
                for slot in turn["user_utterance"]["slots"]:
                    start,end = slot["start"],slot["exclusive_end"]
                    token_dict["entity_tokens"][slot["slot"]].update([" ".join(all_tokens[start:end])])
                    for i in range(start,end):
                        entity_token_indices.add(i)
                
                for idx,token in enumerate(all_tokens):
                    if idx not in entity_token_indices:
                       token_dict["regular_tokens"].add(token) 

            if "system_utterance" in turn:
                all_tokens = turn["system_utterance"]["tokens"]
                entity_token_indices = set()
                # extract entity tokens
                for slot in turn["system_utterance"]["slots"]:
                    start,end = slot["start"],slot["exclusive_end"]
                    token_dict["entity_tokens"][slot["slot"]].update([" ".join(all_tokens[start:end])])
                    for i in range(start,end):
                        entity_token_indices.add(i)
                
                for idx,token in enumerate(all_tokens):
                    if idx not in entity_token_indices:
                       token_dict["regular_tokens"].add(token) 
        print("Processed {} turns".format(counter))
        return token_dict

    def get_dialogues_turn_tuples(self):
        dialgoues = []
        for diag in self.diag_iter():
            info = OrderedDict()
            info["id"] = diag["dialogue_id"]
            info["turn_tuples"] = []
            
            all_turns = []
            for turn in diag["turns"]:
                if "user_utterance" in turn and "system_utterance" in turn:
                    all_turns.append(turn["system_utterance"])
                    all_turns.append(turn["user_utterance"])
                elif "user_utterance" in turn:
                    all_turns.append(turn["user_utterance"])
                elif "system_utterance" in turn:
                    all_turns.append(turn["system_utterance"])
            
            single_turn_list = []
            # print(all_turns)
            for idx, turn in enumerate(all_turns):
                # assumign user starts the conversation then the index of user utterances is even
                if idx%2==0:
                    single_turn_list.append(turn["text"])
                else:
                    single_turn_list.append(turn["text"])
                    if len(single_turn_list)!=2:
                            raise Exception("STH IS WORONG!")
                    info["turn_tuples"].append(tuple(single_turn_list))
                    single_turn_list = []

            
            dialgoues.append(info)
        
        return dialgoues




class DSTCDataReader(object):

    def __init__(self):
        self.train = []
        self.test = []
        self.dev = []
        self.read_data()

    def _create_dialogue(self,dialogue_list):
        from collections import OrderedDict
        diags_list = []
        for i,diag in enumerate(dialogue_list):
            print(i)
            info = OrderedDict()
            info["id"] = str(i+1)
            info["turns"] = []
            for turn in diag:
                text = (turn[0],turn[2])
                tokens = (self.tokenize_sentence(turn[0]),self.tokenize_sentence(turn[2]))
                info["turns"].append(dict(text=text,tokens=tokens))

            diags_list.append(info)
        
        return diags_list

    
    def read_data(self):
        if os.path.exists(os.path.join(CURRENT_DIR,"DSTC2.pkl")):
            import pickle as pk
            with open(os.path.join(CURRENT_DIR,"DSTC2.pkl"),'rb') as f:
                data = pk.load(f)
            self.dev,self.train,self.test = data["dev"],data["train"],data["test"]
            print("READ data from pickled file")
            return
        
        print("Reading data files")
        path = os.path.join(DATA_DIR,"dstc2/data.dstc2.dev.json")
        self.dev = self._create_dialogue(json.load(open(path)))  
        print("created dev set")

        path = os.path.join(DATA_DIR,"dstc2/data.dstc2.test.json")
        self.test = self._create_dialogue(json.load(open(path)))  
        print("created test set")

        path = os.path.join(DATA_DIR,"dstc2/data.dstc2.train.json")
        self.train = self._create_dialogue(json.load(open(path)))  
        print("created train set")

        self.dump_data()
        print("Data dumped!")

    def get_dataset(self,mode="train"):
        data = []
        if mode=="train":
            data = self.train
        elif mode=="test":
            data = self.test
        elif mode=="dev":
            data = self.dev
        
        return data


    def tokenize_sentence(self,sentence):
        return [token.lower_ for token in nlp(sentence)]


    def dump_data(self):
        data = dict(train=self.train,test=self.test,dev=self.dev)
        import pickle as pk
        with open(os.path.join(CURRENT_DIR,"DSTC2.pkl"),'wb') as f:
            pk.dump(data,f)
    

    def get_data_tokens(self,mode="train"):
        tokens = []
        iterator = self.get_data_iterator(mode)
        for turn in iterator:
            s,u = turn["tokens"]
            tokens += list(s)
            tokens +=list(u)
        # add silence token
        tokens.append("sil")

        return list(set(tokens))
    

    def get_data_iterator(self,mode="train"):
        data = self.get_dataset(mode)
        for diag in data:
            for turn in diag["turns"]:
                yield turn
    

    def create_test_data_files(self,mode='test'):
        prependix = "DSTC2"
        diaglogues = self.get_dataset(mode)
        info = {}
        info["input"] = []
        info["target"] = []
        info["diag-id"] = []
        for diag in diaglogues:
            diagId = diag["id"]
            turns_text = []
            for turn in diag["turns"]:
                s,u = turn["tokens"]
                if not s:
                    s = ["sil"]
                if not u:
                    u = ["sil"]
                turns_text.append(' '.join(s))
                turns_text.append(' '.join(u))
            # drop the welcome message
            turns_text = list(turns_text[1:-1])
            # print(turns_text)
            # new shape [u,s,u,s]

            for i in range(0,len(turns_text),2):
                info["input"].append(" ".join(turns_text[0:i+1]))
                info["target"].append(turns_text[i+1])
                info["diag-id"].append(diagId)

        # dump 
        with open(prependix + "input-dev.txt",'w') as f:
            for i in info["input"]:
                f.write(i+'\n')
        with open(prependix+"target-dev.txt",'w') as f:
            for i in info["target"]:
                f.write(i+'\n')
        with open(prependix + "diagid-dev.txt",'w') as f:
            for i in info["diag-id"]:
                f.write(i+'\n')
        
        return info


def main():
    dr = DSTCDataReader()
    dr.read_data()
    t = dr.get_data_tokens("dev")
    print(t)
    dr.create_test_data_files()


def get_m2m_data():

    # file_names = ["sim-R/dev.json"]
    # file_names = ["sim-R/train.json","sim-R/dev.json","sim-R/test.json"]
    file_names = ["sim-M/train.json","sim-M/dev.json","sim-M/test.json"]
    dr = GoogleDataReader(file_names)

    token_dict = dr.get_token_dict()
    print(token_dict["entity_tokens"])


def create_test_files():
    
    file_names = ["sim-M/test.json"]
    prependix = "Movie"

    dr = GoogleDataReader(file_names)
    # create test file to be decoded!
    dialgoues = dr.get_dialogues_turn_tuples()
    info = {}
    info["input"] = []
    info["target"] = []
    info["diag-id"] = []
    for diag in dialgoues:
        diagId = diag["id"]
        turns_text = []
        for (u,s) in diag["turn_tuples"]:
            turns_text.append(u)
            turns_text.append(s)
        for i in range(0,len(turns_text),2):
            info["input"].append(" ".join(turns_text[0:i+1]))
            info["target"].append(turns_text[i+1])
            info["diag-id"].append(diagId)

    # dump 
    with open(prependix + "input-test.txt",'w') as f:
        for i in info["input"]:
            f.write(i+'\n')
    with open(prependix+"target-test.txt",'w') as f:
        for i in info["target"]:
            f.write(i+'\n')
    with open(prependix + "diagid-test.txt",'w') as f:
        for i in info["diag-id"]:
            f.write(i+'\n')
    

if __name__ == '__main__':
    main()
