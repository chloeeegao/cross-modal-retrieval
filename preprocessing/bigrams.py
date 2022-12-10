import json
import os
import re
import copy
import pickle
import argparse
import logging
from logging import FileHandler
import word2vec



w2v_file = '/data/s2478846/data/vocab.bin'
if not os.path.exists(os.path.join(os.path.dirname(w2v_file),'vocab.txt')):
    model = word2vec.load(w2v_file)
    vocab = model.vocab # len(vocab) 30167
    print("Writing to %s..." % os.path.join(os.path.dirname(w2v_file),'vocab.txt'))
    f = open(os.path.join(os.path.dirname(w2v_file),'vocab.txt'),'w')
    f.write("\n".join(vocab))
    f.close()


parser = argparse.ArgumentParser(description='preprocessing parameters')
parser.add_argument('--suffix', type=str, default='1M')
parser.add_argument('--maxlen', type=int, default=20)
parser.add_argument('--vocab', type=str, default='/data/s2478846/data/vocab.txt')
parser.add_argument('--tsamples', type=int, default=20)
parser.add_argument('--f101_cats', type=str, default='../data/food-101/meta/classes.txt')

parser.add_argument('--no_create', dest='create', action ='store_false')
parser.add_argument('--create', dest='create', action='store_true')

# true to compute and store bigrams to disk
# false to go through top N bigrams and create annotations
parser.set_defaults(create=False)

params = parser.parse_args()

class Layer(object):
    L1 = 'layer1'
    L2 = 'layer2'
    L3 = 'layer3'
    INGRS = 'det_ingrs'
    GOODIES = 'goodies'

    @staticmethod
    def load(name, ROOT):
        with open(os.path.join(ROOT, name + '.json'))as f_layer:
            return json.load(f_layer)

    @staticmethod
    def merge(layers, ROOT,copy_base=False):
        layers = [l if isinstance(l, list) else Layer.load(l, ROOT) for l in layers]
        base = copy.deepcopy(layers[0]) if copy_base else layers[0]
        entries_by_id = {entry['id']: entry for entry in base}
        for layer in layers[1:]:
            for entry in layer:
                base_entry = entries_by_id.get(entry['id'])
                if not base_entry:
                    continue
                base_entry.update(entry)
        return base


def detect_ingrs(recipe, vocab):
    try:
        ingr_names = [ingr['text'] for ingr in recipe['ingredients'] if ingr['text']]
    except:
        ingr_names = []
        print("Could not load ingredients! Moving on...")

    detected = set()
    for name in ingr_names:
        name = name.replace(' ','_')
        name_ind = vocab.get(name)
        if name_ind:
            detected.add(name_ind)
        '''
        name_words = name.lower().split(' ')
        for i in xrange(len(name_words)):
            name_ind = vocab.get('_'.join(name_words[i:]))
            if name_ind:
                detected.add(name_ind)
                break
        '''

    return list(detected) + [vocab['</i>']]



print('Loading dataset.')
DATASET = '/data/s2478846/data'
dataset = Layer.merge([Layer.L1, Layer.L2, Layer.INGRS],DATASET)

if params.create:
    print("Creating bigrams...")
    titles = []
    for i in range(len(dataset)):
        title = dataset[i]['title']

        if dataset[i]['partition'] == 'train':
            titles.append(title)
    fileinst = open('../data/titles' + params.suffix + '.txt','w')
    for t in titles:
        fileinst.write( t + " ");

    fileinst.close()

    import nltk
    from nltk.corpus import stopwords
    f = open('../data/titles' +params.suffix+'.txt')
    raw = f.read()
    tokens = nltk.word_tokenize(raw)
    tokens = [i.lower() for i in tokens]
    tokens = [i for i in tokens if i not in stopwords.words('english')]
    #Create your bigrams
    bgs = nltk.bigrams(tokens)
    #compute frequency distribution for all the bigrams in the text
    fdist = nltk.FreqDist(bgs)

    pickle.dump(fdist,open('../data/bigrams'+params.suffix+'.pkl','wb'))

else:
    
    logger = logging.getLogger('bigrams')
    logger.setLevel(logging.DEBUG)
    # logging.basicConfig(filename='./class.log', level=logging.DEBUG)
    filename = 'log.txt'
    fh = FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    
    N = 2000
    MAX_CLASSES = 1000
    MIN_SAMPLES = params.tsamples
    n_class = 1
    ind2class = {}
    class_dict = {}

    fbd_chars = ["," , "&" , "(" , ")" , "'", "'s", "!","?","%","*",".",
                 "free","slow","low","old","easy","super","best","-","fresh",
                 "ever","fast","quick","fat","ww","n'","'n","n","make","con",
                 "e","minute","minutes","portabella","de","of","chef","lo",
                 "rachael","poor","man","ii","i","year","new","style"]

    print('Loading ingr vocab.')
    with open(params.vocab) as f_vocab:
        ingr_vocab = {w.rstrip(): i+2 for i, w in enumerate(f_vocab)} # +1 for lua
        ingr_vocab['</i>'] = 1

    # store number of ingredients (compute only once)
    ningrs_list = []
    for i,entry in enumerate(dataset):

        ingr_detections = detect_ingrs(entry, ingr_vocab)
        ningrs = len(ingr_detections)
        ningrs_list.append(ningrs)

    # load bigrams
    fdist = pickle.load(open('../data/bigrams'+params.suffix+'.pkl','rb'))
    Nmost = fdist.most_common(N)

    # check bigrams
    queries = []
    for oc in Nmost:

        counts = {'train': 0, 'val': 0,'test':0}

        if oc[0][0] in fbd_chars or oc[0][1] in fbd_chars:
            continue

        query = oc[0][0] + ' ' + oc[0][1]
        queries.append(query)
        matching_ids = []
        for i,entry in enumerate(dataset):

            ninstrs = len(entry['instructions'])
            imgs = entry.get('images')
            ningrs =ningrs_list[i]
            title = entry['title'].lower()
            id = entry['id']

            if query in title and ninstrs < params.maxlen and imgs and ningrs<params.maxlen and ningrs is not 0: # if match, add class to id
                # we only add if previous class was background
                # or if there is no class for the id
                if id in class_dict:
                    if class_dict[id] == 0:
                        class_dict[id] = n_class
                        counts[dataset[i]['partition']] +=1
                        matching_ids.append(id)
                else:
                    class_dict[id] = n_class
                    counts[dataset[i]['partition']] +=1
                    matching_ids.append(id)

            else: # if there's no match
                if not id in class_dict: # add background class unless not empty
                    class_dict[id] = 0 # background class


        if counts['train'] > MIN_SAMPLES and counts['val'] > 0 and counts['test'] > 0:
            ind2class[n_class] = query
            logger.info("class_id:{}, class_name:{}, counts:{}".format(n_class, query, counts))
            n_class+=1
        else:
            for id in matching_ids: # reset classes to background
                class_dict[id] = 0

        if n_class > MAX_CLASSES:
            break

    # get food101 categories (if not present)
    food101 = []
    with open(params.f101_cats,'r') as f_classes:
        for l in f_classes:
            cls = l.lower().rstrip().replace('_', ' ')
            # cls = l.lower().rstrip()
            if cls not in queries:
                food101.append(cls)

    for query in food101:
        counts = {'train': 0, 'val': 0,'test':0}
        matching_ids = []
        for i,entry in enumerate(dataset):

            ninstrs = len(entry['instructions'])
            imgs = entry.get('images')
            ningrs =ningrs_list[i]
            title = entry['title'].lower()
            id = entry['id']

            if query in title and ninstrs < params.maxlen and imgs and ningrs<params.maxlen and ningrs is not 0: # if match, add class to id
                # we only add if previous class was background
                # or if there is no class for the id
                if id in class_dict:
                    if class_dict[id] == 0:
                        class_dict[id] = n_class
                        counts[dataset[i]['partition']] +=1
                        matching_ids.append(id)
                else:
                    class_dict[id] = n_class
                    counts[dataset[i]['partition']] +=1
                    matching_ids.append(id)

            else: # if there's no match
                if not id in class_dict: # add background class unless not empty
                    class_dict[id] = 0 # background class

        if counts['train'] > MIN_SAMPLES and counts['val'] > 0 and counts['test'] > 0:
            ind2class[n_class] = query
            print(n_class, query, counts)
            n_class+=1
        else:
            for id in matching_ids: # reset classes to background
                class_dict[id] = 0


    ind2class[0] = 'background'
    print(len(ind2class))
    with open('../data/classes'+params.suffix+'.pkl','wb') as f:
        pickle.dump(class_dict,f)
        pickle.dump(ind2class,f)