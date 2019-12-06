import torch
import os
from io import open
import re
import torch.nn as nn

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import pickle
import spacy

from Voc import Voc
from NetWork import EncoderRNN,AttnDecoderRNN

corpus_name = "cornell movie-dialogs corpus"
device = torch.device("cpu")
datafile = os.path.join(corpus_name,'formatted_movie_lines.txt')
nlp = spacy.load("en_core_web_sm")

MAX_LENGTH = 15

PAD_token = 0  # 填充
SOS_token = 1  # 句子开头
EOS_token = 2  # 句子结尾

def normalizeString(s):
    s = re.sub('[^a-z\\-.,!?]',' ',s.lower())
    s = word_tokenize(s)
    return ' '.join(s)

voc = Voc(None)
with open("model/voc.pkl",'rb') as file:
    voc = pickle.loads(file.read())

with open("model/pairs.pkl",'rb') as file:
    pairs = pickle.loads(file.read())

""" transform to numerical sentence  """
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def evaluate(searcher, voc, sentence, max_length=MAX_LENGTH):
    indexes_batch = [indexesFromSentence(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput( searcher, voc, input_sentence):
    try:
        input_sentence = normalizeString(input_sentence)
        output_words = evaluate(searcher, voc, input_sentence)
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        print('Bot:', ' '.join(output_words))
    except KeyError:
        print("无法匹配，请重新输入。")

""" Using greedy search to find the most likely words from start to end of sentence. """
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token

        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)

        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores

""" For task-Oriented Chatbot: Distinguishing meanning of input sentence through synonyms """
class Synonym:
    def __init__(self, sentence):
        self.content = sentence

    def preprocess(self):
        NounPhrase = self.findNounPhrase()
        self.content = re.sub('[^a-zA-Z]', ' ', self.content)
        self.content = self.content.split()
        self.content += self.mergeWords(NounPhrase)
        self.removeStop()

    def findNounPhrase(self):
        l = []
        for np in nlp(self.content).noun_chunks:
            a = [w for w in np.text.split() if not w in stopwords.words("english")]
            l.append('_'.join(a))
        return l

    def mergeWords(self, phrases_list):
        l = []
        for i in phrases_list:
            temp = i.split()
            if len(temp) > 1:
                l.append('_'.join(temp))
            else:
                l.append(i)
        return l

    def removeStop(self):
        self.content = [w for w in self.content if not w in stopwords.words("english")]

    def findSynonym(self):
        for i in self.content:
            for synset in wn.synsets(i):
                for eachword in synset.lemma_names():
                    key = self.iterDatabase(eachword)
                    if key:
                        return key
        return False

    def iterDatabase(self, word):
        for key in synonyms:
            for value in synonyms[key]:
                if word.lower() == value:
                    return key
        return False


model_name = 'model'
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
checkpoint_iter = 8000
loadFilename = os.path.join(model_name,'{}_checkpoint.tar'.format(checkpoint_iter))

checkpoint = torch.load(loadFilename)
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']
embedding_sd = checkpoint['embedding']
voc.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
embedding = nn.Embedding(voc.num_words, hidden_size)
embedding.load_state_dict(embedding_sd)

encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = AttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)
print('Models built and ready to go!')


encoder.eval()
decoder.eval()

searcher = GreedySearchDecoder(encoder, decoder)
sentence = "How are you?"
synonyms = {'ultrasound': ['ultrasonic', 'ultrasonography,sonography', 'echography', 'ultrasound'],
            'mri': ['magnetic resonance imaging', 'mri']
            }
sys = Synonym(sentence)

sys.preprocess()

syn_result = sys.findSynonym()
if syn_result:
    print("Flag is : " + syn_result)
else:
    print("No Flag.")
    evaluateInput(encoder, decoder, searcher, voc, sentence)
print('Finished')


