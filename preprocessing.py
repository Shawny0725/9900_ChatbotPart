import torch
import os
from io import open
import codecs
import csv

"""
Usage: Doing preprocessing work to sentences pairs data, and write them into a new file.
"""
corpus = "cornell movie-dialogs corpus"
device = torch.device("cpu")
def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)
printLines(os.path.join(corpus, "movie_lines.txt"))
printLines(os.path.join(corpus, "movie_conversations.txt"))

"""read movie_lines.txt，return a dictionary"""
def loadLines(fileName,fields):
    lines = {}
    with open(fileName,'r',encoding = 'iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines

"""read movie_conversations.txt，return a dictionary"""
def loadConversations(fileName,lines,fields):
    conversations = []
    with open(fileName,'r',encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            lineIds = eval(convObj["utteranceIDs"])
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations

"""extract pairs of sentences"""
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        for i in range(len(conversation["lines"]) - 1):
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs

"""****************************************************************"""

datafile = os.path.join(corpus,'formatted_movie_lines.txt')
delimiter = '\t'
delimiter = str(codecs.decode(delimiter, "unicode_escape"))
lines = {}
conversations = []
MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),lines, MOVIE_CONVERSATIONS_FIELDS)

with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)
