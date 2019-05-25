import json
import os
from tqdm import tqdm
import pickle

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CURRENT_DIR,'../data')
from google_data_reader import DSTCDataReader

def read_dstc_enities():
    DSTC_ENTITIES = ['addr','area', 'food', 'phone', 'pricerange', 'postcode', 'signature', 'name']

    # get ontologies
    path = os.path.join(DATA_DIR,"dstc2/data.dstc2.ontology.json")
    ontoloy = json.load(open(path))

    # get db values
    path = os.path.join(DATA_DIR,"dstc2/data.dstc2.db.json")
    db = json.load(open(path))

    # get possible values from db
    from collections import defaultdict
    possible_entitie_val = defaultdict(set)

    # update enitity values from ontology
    for ent in DSTC_ENTITIES:
        if ent in ontoloy['informable']:
            possible_entitie_val[ent].update(ontoloy['informable'][ent])
    # update entitiy values from db
    for db_entry in db:
        for ent,val in db_entry.items():
            # fix postal code vals
            if ent=='postcode':
                l = val.strip().split()
                if len(l)==6:
                    val = '{}.{} {}, {} {}.{}'.format(l[0],l[1],l[2],l[3],l[4],l[5]).upper()
            possible_entitie_val[ent].update([val])

    return {
        'db':db,
        'ontology':ontoloy,
        'entity_vals': possible_entitie_val
    }

def get_entity_per_turn(entity_data,mode='test'):
    dr = DSTCDataReader()
    dr.read_data()

    system_test_data = dr.create_test_data_files(mode=mode)['target']
    system_test_data_entity_info = []
    for idx, turn in enumerate(system_test_data):
        info = {
            'text': turn,
            'entities':[]
        }
        for ent, possible_vals in entity_data['entity_vals'].items():
            for val in possible_vals:
                if val in turn:
                    info['entities'].append({
                        'entity': ent,
                        'value':val
                    })
        system_test_data_entity_info.append(info)

    return system_test_data_entity_info

def main():
    entity_data = read_dstc_enities()
    mode='test'
    # system_test_data_entity_info = get_entity_per_turn(entity_data,mode=mode)
    # save results
    # res = json.dumps(entity_data['entity_vals'],indent=4)
    # with open('dstc_entitiy_data.json'.format(mode),'w') as f:
    #     f.write(res)
    # pickelize entity_data
    with open('dstc_entitiy_data.pkl', 'wb') as handle:
        pickle.dump(entity_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()