'''
This script is created for:

1. Graphically visualisation of the max-similarity scores of OMCS_knowledge facts with CRIC_supplied_knowledge_facts
2. Ranking of the facts with simple cosine-similarity calcualted between the retreived_facts and the 
   groundtruth_knowledge supplied in the dataset.
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

# cross_encoder_reranker for reranking the retreived sentence facts trained on MS-MARCO passage dataset
cross_encoder_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


from datasets import load_dataset

def sigmoid(x):
    z = 1/(1 + np.exp(-x)) 
    return z

# Pull from Huggingface directly

omcs_full_with_embeddings = load_dataset("dutta18/omcs_dataset_full_with_embeds", split='train')

# RAG

# rename the column

omcs_full_with_embeddings = omcs_full_with_embeddings.rename_column('fact', 'text')

# delete the count column

omcs_full_with_embeddings = omcs_full_with_embeddings.remove_columns('count')

# Have to copy the text(facts) column to anothe title column

title_list = omcs_full_with_embeddings['text']

omcs_full_with_embeddings = omcs_full_with_embeddings.add_column('title', title_list)

omcs_full_with_embeddings.add_faiss_index(column='embeddings')

from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, DPRContextEncoderTokenizerFast

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")

retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", indexed_dataset = omcs_full_with_embeddings)
retriever.return_tokenized_docs = True

ctx_encoder_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
retriever.ctx_encoder_tokenizer = ctx_encoder_tokenizer

model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

# function to retrieve the relevant facts from captions

def retrieveRelevantFacts(caption, k):

    captionPassed = caption

    input_dict = tokenizer.prepare_seq2seq_batch( captionPassed, return_tensors="pt") 
    
    question_encoder = model.question_encoder
    question_enc_outputs = question_encoder(input_dict['input_ids'], return_dict=True)
    question_encoder_last_hidden_state = question_enc_outputs[0]  
    
    retriever_outputs = retriever(
                    input_dict['input_ids'],
                    question_encoder_last_hidden_state.cpu().detach().to(torch.float32).numpy(),
                    prefix=model.generator.config.prefix,
                    n_docs=k,
                    return_tensors="pt",
    )
    
    context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_ids = (
                        retriever_outputs["context_input_ids"],
                        retriever_outputs["context_attention_mask"],
                        retriever_outputs["retrieved_doc_embeds"],
                        retriever_outputs["doc_ids"],
    )
    
    # set to correct device

    retrieved_doc_embeds = retrieved_doc_embeds.to(question_encoder_last_hidden_state)
    context_input_ids = context_input_ids.to(input_dict['input_ids'])
    context_attention_mask = context_attention_mask.to(input_dict['input_ids'])
    
    
    # compute doc_scores

    doc_scores = torch.bmm(
        question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
    ).squeeze(1)
    
    results = ctx_encoder_tokenizer.batch_decode(retriever_outputs['tokenized_doc_ids'], skip_special_tokens=True)
    results = [fact.split('.')[0] for fact in results]
    
    return results

# This function takes in 2 sentence and returns the cosine_sim score

def cosine_similarity(sentence1, sentence2):
    
    embeddings1 = sbertModel.encode(sentence1, convert_to_tensor=True)
    embeddings2 = sbertModel.encode(sentence2, convert_to_tensor=True)
    
    cosine_score = util.cos_sim(embeddings1, embeddings2)
    
    return cosine_score.item()

# This function will find and return the highest_similarity_score of the retrieved facts
# For details print the factScoreTupleList

def findMaxSimilarityForHist(question, groundTruthKnowledgeSupplied, k):
    
    relevantFactsRetrieved = retrieveRelevantFacts(question, k)    
    
    # reranking using sentence-transformers cross-encoder model
    crosss_encoder_inputs = []

    for fact in relevantFactsRetrieved:
        crosss_encoder_inputs.append( (question, fact) )

    cross_encoder_scores = cross_encoder_reranker.predict(crosss_encoder_inputs)
    cross_encoder_sigmoid_scores = list(map(sigmoid, cross_encoder_scores ))
    
    '''
    for idx in range(len(cross_encoder_sigmoid_scores)):
        print(f'{relevantFactsRetrieved[idx]} -> {cross_encoder_sigmoid_scores[idx]}')
    '''
    
    factAndCrossEncoderZipped = list(zip(relevantFactsRetrieved, cross_encoder_sigmoid_scores))

    # Here we will get the highest rank fact ranked by the cross-encoder.
    highestRankedFact = (max(factAndCrossEncoderZipped, key = lambda x:x[1]))[0]
    
    # print(highestRankedFact)
    
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

    cos_val = findMaxSimilarityForHist(currentQuestion, groundKnowledgeSupplied, 20)

    if cos_val <= 0.4:
        count01to04_20 += 1

    if cos_val > 0.4 and cos_val <= 0.7:
        count05to07_20 += 1

    if cos_val > 0.7 and cos_val <= 1.0:
        count08to10_20 += 1


for i in tqdm(range(len(questionList))):
    
    currentQuestion = questionList[i]
    groundKnowledgeSupplied = k_triplet[i]

    cos_val = findMaxSimilarityForHist(currentQuestion, groundKnowledgeSupplied, 40)

    if cos_val <= 0.4:
        count01to04_40 += 1

    if cos_val > 0.4 and cos_val <= 0.7:
        count05to07_40 += 1

    if cos_val > 0.7 and cos_val <= 1.0:
        count08to10_40 += 1


for i in tqdm(range(len(questionList))):
    
    currentQuestion = questionList[i]
    groundKnowledgeSupplied = k_triplet[i]

    cos_val = findMaxSimilarityForHist(currentQuestion, groundKnowledgeSupplied, 60)

    if cos_val <= 0.4:
        count01to04_60 += 1

    if cos_val > 0.4 and cos_val <= 0.7:
        count05to07_60 += 1

    if cos_val > 0.7 and cos_val <= 1.0:
        count08to10_60 += 1


for i in tqdm(range(len(questionList))):
    
    currentQuestion = questionList[i]
    groundKnowledgeSupplied = k_triplet[i]

    cos_val = findMaxSimilarityForHist(currentQuestion, groundKnowledgeSupplied, 80)

    if cos_val <= 0.4:
        count01to04_80 += 1

    if cos_val > 0.4 and cos_val <= 0.7:
        count05to07_80 += 1

    if cos_val > 0.7 and cos_val <= 1.0:
        count08to10_80 += 1


for i in tqdm(range(len(questionList))):
    
    currentQuestion = questionList[i]
    groundKnowledgeSupplied = k_triplet[i]

    cos_val = findMaxSimilarityForHist(currentQuestion, groundKnowledgeSupplied, 100)

    if cos_val <= 0.4:
        count01to04_100 += 1

    if cos_val > 0.4 and cos_val <= 0.7:
        count05to07_100 += 1

    if cos_val > 0.7 and cos_val <= 1.0:
        count08to10_100 += 1


for i in tqdm(range(len(questionList))):
    
    currentQuestion = questionList[i]
    groundKnowledgeSupplied = k_triplet[i]

    cos_val = findMaxSimilarityForHist(currentQuestion, groundKnowledgeSupplied, 120)

    if cos_val <= 0.4:
        count01to04_120 += 1

    if cos_val > 0.4 and cos_val <= 0.7:
        count05to07_120 += 1

    if cos_val > 0.7 and cos_val <= 1.0:
        count08to10_120 += 1


# Print the results into a .txt file

with open('results_rag_retrieve_then_cross_encoder_ranking.txt', 'w') as file:
    
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



