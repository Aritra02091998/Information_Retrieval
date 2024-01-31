'''
This script is created for:

1. Get the count of max-similarity scores of OMCS_knowledge facts with CRIC_supplied_knowledge_facts.
2. Retrieval of the facts using BiEncoder, Ranking of the facts with Cross_Encoder_Ranker.
3. The the final top ranked fact is checked with the golden fact using cosine similarity.
'''

import os
import re
import torch
import spacy
import json
import math
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from PIL import Image as img
from collections import Counter as Count

import warnings
warnings.filterwarnings('ignore')


from sentence_transformers import SentenceTransformer, CrossEncoder, util

# sentence Transformer for encoding the sentences
sbertModel = SentenceTransformer('all-MiniLM-L6-v2')

# initialise bi_encoder retriever
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens

# cross_encoder_reranker for reranking the retreived sentence facts trained on MS-MARCO passage dataset
cross_encoder_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


from datasets import load_dataset

def sigmoid(x):
    z = 1/(1 + np.exp(-x)) 
    return z

# Load the OMCS Json data

f = open('omcs_retrieval_full.json')
data = json.load(f)
f.close()

# Store all 1.5M facts into Py List

facts = [] 
for i in range(len(data)):
    facts.append(data[i]['fact'])

# Encode the corpus using BiEncoder

corpus_embeddings = bi_encoder.encode(facts, convert_to_tensor=True, show_progress_bar=True)


# Semantic Search

# Encode the query using the bi-encoder and find potentially relevant passages

def retreiveFacts(query, k):

    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k = k)
    hits_dict = hits[0]
            
    '''
    for element in sorted_list_of_dicts_hits:
        print(f'{facts[element["corpus_id"]]} -> {element["score"]}')
    '''
    return hits_dict

# This function takes in 2 sentence and returns the cosine_sim score

def cosine_similarity(sentence1, sentence2):
    
    embeddings1 = sbertModel.encode(sentence1, convert_to_tensor=True)
    embeddings2 = sbertModel.encode(sentence2, convert_to_tensor=True)
    
    cosine_score = util.cos_sim(embeddings1, embeddings2)
    
    return cosine_score.item()

# The purpose of the function is, it will retrieve the facts from omcs based on the query we supply to it with BiEncoder
# Retreiver. Then it will rerank the facts using the Cross_encoder_reranker. The top ranked facts will be compared
# against the golden_fact/groundtruthKnowledge and return the cos_simialrity of the two.

def maxCosSimilarity(query, groundTruthKnowledgeSupplied, k):
    
    hits_dictionary = retreiveFacts(query, k)
    
    # Here we will not use the default ranking and scores of the biEncoder rather we will use the CrossEncoder for ranking
    # the retreived facts. Bienocoder is only for retrieval.
    
    # Store the facts Only in the below list
    relevantFactsRetrieved = [ facts[element['corpus_id']] for element in hits_dictionary]
    
    # reranking using sentence-transformers cross-encoder model
    cross_encoder_inputs = []

    for fact in relevantFactsRetrieved:
        cross_encoder_inputs.append( (query, fact) )
        
    cross_encoder_scores = cross_encoder_reranker.predict(cross_encoder_inputs)
    cross_encoder_sigmoid_scores = list(map(sigmoid, cross_encoder_scores ))
    
    '''
    for idx in range(len(cross_encoder_sigmoid_scores)):
        print(f'{relevantFactsRetrieved[idx]} -> {cross_encoder_sigmoid_scores[idx]}')
    ''' 
    factAndCrossEncoderZipped = list(zip(relevantFactsRetrieved, cross_encoder_sigmoid_scores))

    # Here we will get the highest rank fact ranked by the cross-encoder.
    highestRankedFact = (max(factAndCrossEncoderZipped, key = lambda x:x[1]))[0]
    
    #print('\n', highestRankedFact, '<->', groundTruthKnowledgeSupplied)
    
    # Now check the cosine_similarity of this fact with the groundTruthKnowldegeSupplied
    cos_score = cosine_similarity(highestRankedFact, groundTruthKnowledgeSupplied)
    
    return cos_score
    


# Fetching CRIC

train_file_path = '/home/aritra/cric/train_questions.json'
val_file_path = '/home/aritra/cric/val_questions.json'
test_file_path = '/home/aritra/cric/test_v1_questions.json'

# Training Set

with open(train_file_path, "r") as file:
     train_json = json.load(file)
        
# Validation Set

with open(val_file_path, "r") as file:
     val_json = json.load(file)
        
# Test Set

with open(test_file_path, "r") as file:
     test_json = json.load(file)


questionList, answerList, imgList, k_triplet = [],[],[],[]

# verifying
indexToExclude = []

with open('../text_files/error1.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        indexToExclude.append(number)
        
with open('../text_files/error2.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        indexToExclude.append(number)
        
with open('../text_files/error3.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        indexToExclude.append(number)

for i in tqdm(range(len(train_json))):
    
    if i in indexToExclude:
        continue
        
    pointer = train_json[i]
    
    questionList.append(pointer['question'])
    answerList.append(pointer['answer'])
    imgList.append(pointer['image_id'])
    k_triplet.append( ' '.join(pointer['sub_graph']['knowledge_items'][0]['triplet']) + '. ' )

# Subsetting dataset

questionList = questionList[0:12000]
answerList = answerList[0:12000]
imgList = imgList[0:12000]
k_triplet = k_triplet[0:12000]


# Table Data Preparation

k_values = (20,40,60,80,100,120)

count01to04_20, count05to07_20, count08to10_20  = 0,0,0
count01to04_40, count05to07_40, count08to10_40  = 0,0,0
count01to04_60, count05to07_60, count08to10_60  = 0,0,0
count01to04_80, count05to07_80, count08to10_80  = 0,0,0
count01to04_100, count05to07_100, count08to10_100  = 0,0,0
count01to04_120, count05to07_120, count08to10_120  = 0,0,0


for i in tqdm(range(len(questionList))):
    
    currentQuestion = questionList[i]
    groundKnowledgeSupplied = k_triplet[i]

    cos_val = maxCosSimilarity(currentQuestion, groundKnowledgeSupplied, 20)

    if cos_val <= 0.4:
        count01to04_20 += 1

    if cos_val > 0.4 and cos_val <= 0.7:
        count05to07_20 += 1

    if cos_val > 0.7 and cos_val <= 1.0:
        count08to10_20 += 1


for i in tqdm(range(len(questionList))):
    
    currentQuestion = questionList[i]
    groundKnowledgeSupplied = k_triplet[i]

    cos_val = maxCosSimilarity(currentQuestion, groundKnowledgeSupplied, 40)

    if cos_val <= 0.4:
        count01to04_40 += 1

    if cos_val > 0.4 and cos_val <= 0.7:
        count05to07_40 += 1

    if cos_val > 0.7 and cos_val <= 1.0:
        count08to10_40 += 1


for i in tqdm(range(len(questionList))):
    
    currentQuestion = questionList[i]
    groundKnowledgeSupplied = k_triplet[i]

    cos_val = maxCosSimilarity(currentQuestion, groundKnowledgeSupplied, 60)

    if cos_val <= 0.4:
        count01to04_60 += 1

    if cos_val > 0.4 and cos_val <= 0.7:
        count05to07_60 += 1

    if cos_val > 0.7 and cos_val <= 1.0:
        count08to10_60 += 1


for i in tqdm(range(len(questionList))):
    
    currentQuestion = questionList[i]
    groundKnowledgeSupplied = k_triplet[i]

    cos_val = maxCosSimilarity(currentQuestion, groundKnowledgeSupplied, 80)

    if cos_val <= 0.4:
        count01to04_80 += 1

    if cos_val > 0.4 and cos_val <= 0.7:
        count05to07_80 += 1

    if cos_val > 0.7 and cos_val <= 1.0:
        count08to10_80 += 1


for i in tqdm(range(len(questionList))):
    
    currentQuestion = questionList[i]
    groundKnowledgeSupplied = k_triplet[i]

    cos_val = maxCosSimilarity(currentQuestion, groundKnowledgeSupplied, 100)

    if cos_val <= 0.4:
        count01to04_100 += 1

    if cos_val > 0.4 and cos_val <= 0.7:
        count05to07_100 += 1

    if cos_val > 0.7 and cos_val <= 1.0:
        count08to10_100 += 1


for i in tqdm(range(len(questionList))):
    
    currentQuestion = questionList[i]
    groundKnowledgeSupplied = k_triplet[i]

    cos_val = maxCosSimilarity(currentQuestion, groundKnowledgeSupplied, 120)

    if cos_val <= 0.4:
        count01to04_120 += 1

    if cos_val > 0.4 and cos_val <= 0.7:
        count05to07_120 += 1

    if cos_val > 0.7 and cos_val <= 1.0:
        count08to10_120 += 1


# Print the results into a .txt file

with open('results_biencoder_retreival_then_cross_encoder.txt', 'w') as file:
    string = f"k_val:\t0.1 to 0.4\t0.5 to 0.7\t0.8 to 1.0"
    string1 = f"k_20:\t\t{count01to04_20}\t\t{count05to07_20}\t\t{count08to10_20}"
    string2 = f"k_40:\t\t{count01to04_40}\t\t{count05to07_40}\t\t{count08to10_40}"
    string3 = f"k_60:\t\t{count01to04_60}\t\t{count05to07_60}\t\t{count08to10_60}"
    string4 = f"k_80:\t\t{count01to04_80}\t\t{count05to07_80}\t\t{count08to10_80}"
    string5 = f"k_100:\t\t{count01to04_100}\t\t{count05to07_100}\t\t{count08to10_100}"
    string6 = f"k_120:\t\t{count01to04_120}\t\t{count05to07_120}\t\t{count08to10_120}"

    file.write(string + '\n')
    file.write(string1 + '\n')
    file.write(string2 + '\n')
    file.write(string3 + '\n')
    file.write(string4 + '\n')
    file.write(string5 + '\n')
    file.write(string6 + '\n')



