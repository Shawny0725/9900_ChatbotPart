import torch
import os
from io import open
import unicodedata
import re
from nltk.tokenize import word_tokenize
import spacy
import pickle

from Voc import Voc

"""
Usage: Generating and storing Voc object and sentence pairs to decrease running time, because they are constant 
if parameters keep the same. Training or evaluating just need to read those two pickle files before launching.

"""

corpus_name = "cornell movie-dialogs corpus"
device = torch.device("cpu")
datafile = os.path.join(corpus_name,'formatted_movie_lines.txt')
nlp = spacy.load("en_core_web_sm")
MIN_COUNT = 3
MAX_LENGTH = 15

""" replacing All non-alphabetic characters except basic punctuation to Space """
def normalizeString(s):
    s = re.sub("[^a-z\\'-.,!?]",' ',s.lower())
    s = word_tokenize(s)
    return ' '.join(s)

""" transform sentence format """
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

# 初始化Voc对象 和 格式化pairs对话存放到list中
""" normalize sentence pairs """
def readVocs(datafile, corpus_name):
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

""" Remove the pairs if two sentences are longer than MAX_LENGTH """
def filterPair(pair):
    return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def loadPrepareData(corpus_name, datafile):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

voc, pairs = loadPrepareData(corpus_name, datafile)

""" remove those pairs containing the words whose frequence are less than MIN_COUNT """
def trimRareWords(voc, pairs, MIN_COUNT):
    voc.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


""" Storing Voc and Pairs in local directory"""
pairs = trimRareWords(voc, pairs, MIN_COUNT)
for pair in pairs[:10]:
    print(pair,len(pair[0].split()),len(pair[1].split()))

output_voc = open("model/voc.pkl", 'wb')
str = pickle.dumps(voc)
output_voc.write(str)
output_voc.close()

output_pairs = open("model/pairs.pkl", 'wb')
str = pickle.dumps(pairs)
output_pairs.write(str)
output_pairs.close()