import torch
import os
from io import open
import itertools
import random
import torch.nn as nn
from torch import optim

import pickle

from Voc import Voc
from NetWork import EncoderRNN,AttnDecoderRNN

"""
Usage: Running this file to train model(general dialogue chatbot) and save model in local directory.
"""

corpus_name = "cornell movie-dialogs corpus"
device = torch.device("cpu")
datafile = os.path.join(corpus_name,'formatted_movie_lines.txt')
MIN_COUNT = 3
MAX_LENGTH = 15

PAD_token = 0  # padding
SOS_token = 1  # start tag of sentence
EOS_token = 2  # end tag of sentace

"""
Loading voc object and pairs variable from local directory.
"""
voc = Voc(None)
with open("model/voc.pkl",'rb') as file:
    voc = pickle.loads(file.read())

with open("model/pairs.pkl",'rb') as file:
    pairs = pickle.loads(file.read())

""" transform to numerical sentence  """
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

""" padding """
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

""" transform sentences to Mask"""
def binaryMatrix(l):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


""" return length of sentences before padding and paded sentences(tensors) """
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

""" return max length befor padding and reutnr paded tensor and mask """
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

""" apply above functions on batchs """
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

""" Using binary mask to caculate NLLLoss"""
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    return loss, nTotal.item()

"""
In training process, 
1) we apply teacher_frocing randomly to improve training performance and decrease probability
of instability.
2) gradient clipping is helpful to avoid gradient exploding problem.

"""
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
          encoder_optimizer, decoder_optimizer, batch_size, clip):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            """ Teacher forcing: next input is the current real target output """
            decoder_input = target_variable[t].view(1, -1)

            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            """" No teacher forcing: next input is the decoder’s guess """
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])

            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    loss.backward()

    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding
               , n_iteration, batch_size, print_every, save_every, clip):

    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    print('Initializing ...')
    start_iteration = 1
    print_loss = 0

    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, batch_size, clip)

        print_loss += loss

        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration,
                                                                                          iteration / n_iteration * 100,
                                                                                          print_loss_avg))
            print_loss = 0

        if (iteration % save_every == 0):
            directory = os.path.join(model_name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


""" Initial parameters and start training """
model_name = 'model'
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

loadFilename = None

print('Generating encoder and decoder!')

embedding = nn.Embedding(voc.num_words, hidden_size)
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = AttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
print('Model ready to go!')

clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 8000
print_every = 100
save_every = 8000

encoder.train()
decoder.train()

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding
               , n_iteration, batch_size, print_every, save_every, clip)

