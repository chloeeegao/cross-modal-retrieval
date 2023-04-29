import os
import pickle
import spacy
from spacy.pipeline import EntityRuler
import re

nlp= spacy.load("/data/s2478846/cross_modal_retrieval/preprocessing/food_entity_extractor")
# ner = nlp.get_pipe("ner")
if 'entity_ruler' in nlp.pipe_names:
    nlp.remove_pipe("entity_ruler") 
# ruler = nlp.add_pipe(nlp.create_pipe("entity_ruler"))
ruler = EntityRuler(nlp)
nlp.add_pipe(ruler)
ruler.add_patterns([{"label": "ING", "pattern": "rice"}])
ruler.add_patterns([{"label": "ING", "pattern": "mixture"}])
ruler.add_patterns([{"label": "ING", "pattern": "meat"}])
ruler.add_patterns([{"label": "ING", "pattern": "chicken"}])
ruler.add_patterns([{"label": "ING", "pattern": "pork"}])
ruler.add_patterns([{"label": "ING", "pattern": "beef"}])
ruler.add_patterns([{"label": "ING", "pattern": "ingredients"}])
valid_ingredient = pickle.load(open('/data/s2478846/data/valid_ingredients.pkl','rb'))

cooking_verbs = ['add','bake', 'barbecue', 'baste', 'beat', 'blanch', 'blend', 'bring',
                 'boil', 'braise', 'bread', 'break','broil', 'brown','brush', 'candy', 'can', 
                 'carve', 'char', 'check', 'chill', 'chop', 'clean', 'coat', 
                 'combine', 'cook', 'cool', 'core', 'cover','cream', 'crisp', 'crush', 
                 'cube', 'cut', 'debone', 'decorate' 'deep-fry', 'dehydrate', 'dice', 'dip' 'dissolve', 
                 'drain', 'dress', 'drizzle', 'dry', 'drop','dust', 'emulsify', 'ferment', 'filet', 
                 'flame', 'flambé', 'flip', 'fold', 'freeze', 'fry', 'garnish', 'glaze', 'grate', 
                 'grill', 'grind', 'gut', 'heat', 'hull', 'infuse', 'julienne', 'knead', 'layer', 'level', 
                 'liquefy', 'light' 'marinate', 'mash', 'measure','melt', 'mince', 'mix', 'mold', 'move',
                 'microwave','oil', 'pack','pan-fry', 'parboil', 'pare', 'peel', 'place', 'pickle', 'pierce', 'pinch', 'pit', 'poach', 'pop', 
                 'pour', 'preheat', 'preserve', 'pressure-cook', 'prick', 'puree', 'push', 'put', 'reduce', 'remove', 'rinse',
                 'refrigerate', 'roast', 'roll', 'sauté', 'saute','serve', 'scald', 'scramble','scallop', 'score', 'sear', 
                 'season', 'shred', 'simmer', 'sip','sift', 'skewer', 'slice', 'smoke', 'smooth', 'soak', 'soften', 'sprinkle', 
                 'sous-vide', 'spatchcock', 'spice', 'spread', 'squeeze','steam', 'steep', 'stir', 'strain', 'stick',
                 'stuff', 'submerge', 'sweeten', 'swirl', 'taste', 'take','temper', 'tenderize', 'thicken', 'toast', 'top',
                 'toss', 'truss', 'thread', 'turn on', 'turn off', 'wash','weight','whip', 'whisk' 'wilt']

def info_extract(recipe_id, recipe):
    
    ingredients = recipe['ingredients']
    instructions = recipe['instructions']

    entity_list = valid_ingredient[recipe_id]
    for sent in ingredients:
        sent = re.sub(r'[^a-zA-Z ]', ' ', sent)
        doc = nlp(sent)
        for token in doc:
            if token.dep_ =='ROOT' and token.pos_ == 'NOUN':
                entity_list.append(token.text)
            elif token.dep_ in ['dobj','nsubj','pobj'] and token.pos_ =='NOUN':
                entity_list.append(token.text)
    entity_list = list(set(entity_list))      
    patterns = []
    for ent in entity_list:
        patterns.append({"label": "ING", "pattern": ent})
    ruler.add_patterns(patterns)
        
    action_ing = []
    for sent in instructions:
        doc = nlp(sent)
        tmp = []
        entities = [ent.text for ent in doc.ents if ent.label_ in ['ING','FOOD']]
        if 'ingredients' in entities:
            entities = entity_list
        for token in doc:
            if token.dep_ == 'ROOT' and token.lemma_ in cooking_verbs:
                if len(tmp) == 0:
                    tmp.append(token.text)
                    tmp.extend(entities)
            elif token.dep_ in ['compound', 'nmod', 'xcomp', 'amod'] and token.lemma_ in cooking_verbs:
                if len(tmp) == 0:
                    tmp.append(token.text)
                    tmp.extend(entities)
        if len(tmp) != 0:
            action_ing.append(tmp)

    return action_ing


