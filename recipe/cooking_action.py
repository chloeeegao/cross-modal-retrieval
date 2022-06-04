from collections import defaultdict
from typing import Dict
from nltk.corpus import stopwords
import spacy 
from spacy.matcher import Matcher 
from copy import deepcopy
import string

nlp= spacy.load('en_core_web_sm')

# cut_list = ['slice', 'smash', 'mince', 'dice', 'cut', 'shred']
# cook_list = ['bake', 'braise', 'brew', 'boil', 'dress', 'stew', 'simmer',
#             'casserole', 'fry', 'stirfry', 'panfry', 'grill', 'juice',
#             'microwave', 'season', 'roast', ' smoke',' pickle', 'scald',
#             'steam', 'candy', 'jelly']
# invisible_ingrs = ['salt', 'sugar']

# POS and Dependency Parser

def extract_action(ingrs, instrs):
    # ingrs = ingredients[id]
    # instrs = instructions[id]
    ingr_dict = defaultdict(list)

    doc = nlp(' '.join(ingr for ingr in ingrs))
    ingrs_tmp = [token.lemma_ for token in doc if token.pos_=='NOUN']
    ingrs_copy = deepcopy(ingrs)
    ingrs_copy.extend(ingrs_tmp)
    ingrs_list = list(set(ingrs_copy))

    for ingr in ingrs_list:
        sent = ['I ' + instr for instr in instrs if ingr in instr]
        ingr_dict[ingr] = sent

    matcher = Matcher(nlp.vocab)
    pattern = [[{'POS':'VERB'},{'POS':'CCONJ'},{'POS':'VERB'},{'POS':'DET',"OP": "?"},{"POS": "ADJ", "OP": "?"},{'POS': 'NOUN','OP':'+'}],
    [{'POS':'VERB'},{'POS':'DET',"OP": "?"},{"POS": "ADJ", "OP": "?"},{'POS': 'NOUN','OP':'+'},{'POS': 'NOUN','OP':'*'}],
    [{'POS':'VERB'},{'POS':'ADP',"OP": "*"},{'POS':'PROPN','OP':'*'},{'POS':'DET',"OP": "*"},{"POS": "ADJ", "OP": "*"},{'POS': 'NOUN','OP':'+'}],
    [{'POS':'VERB'},{'POS':'ADP',"OP": "?"},{'POS':'CCONJ'},{'POS':'VERB'},{"POS": "ADJ", "OP": "*"},{'POS': 'NOUN','OP':'+'}],
    [{'POS':'VERB'},{'POS':'ADP',"OP": "?"},{"POS": "ADJ", "OP": "?"},{'POS':'NOUN','OP':'+'},{'POS':'CCONJ'},{"POS": "ADJ", "OP": "?"},{'POS': 'NOUN','OP':'+'}],
    [{'POS':'VERB'},{'POS':'ADV','OP':'?'},{"POS": "ADJ", "OP": "?"},{'POS': 'NOUN','OP':'+'}],
    [{'POS':'VERB'},{'POS':'DET',"OP": "?"},{'POS':'NOUN','OP':'+'},{"POS": "ADP", "OP": "?"},{'POS':'DET',"OP": "?"},{'POS': 'NOUN','OP':'+'}]]
    matcher.add('action+ingrs', pattern)

    action_ingrs = []
    for ingr in ingrs_list:
        senq = ' '.join(sent for sent in ingr_dict[ingr])
        doc = nlp(senq)
        matches = matcher(doc)
        for _, start, end in matches:
            # string_id = nlp.vocab.strings[match_id]  # Get string representation
            span = doc[start:end]  # The matched span
            if ingr in span.text:
                action_ingrs.append(span.text)

    action_dict = defaultdict(list)
    if len(action_ingrs) > 0:
        action_ingrs = list(set(action_ingrs))

        for i in range(len(action_ingrs)):
            filtered = [ w for w in action_ingrs[i].split(' ') if not w in stopwords.words('english')] 
            # print(filtered)
            doc = nlp(filtered[0])
            lemma = [token.lemma_ for token in doc]
            # print(lemma,filtered[1:len(filtered)])
            if lemma[0] not in action_dict.keys():
                action_dict[lemma[0]] = filtered[1:len(filtered)]
            else:
                action_dict[lemma[0]].extend(filtered[1:len(filtered)])
        
        for k, v in action_dict.items():
            action_dict[k] = list(set(v))
    
    else:
        lowers = ' '.join(text.lower() for text in ingrs)
        remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
        no_punctuation = lowers.translate(remove_punctuation_map)
        action_dict['mix'] = [ w for w in no_punctuation.split(' ') if not w in stopwords.words('english')]

    return action_dict







    


