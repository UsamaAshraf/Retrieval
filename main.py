__author__ = 'Usama Ashraf'

import os
import string
from stemming.porter2 import stem
import collections
import math

def calculate_cosine_similarity(v1, v2):
    euclidean = lambda x: math.sqrt(sum(i**2 for i in x))
    return sum(list(map(lambda x,y: x * y, v1, v2))) / euclidean(v1) * euclidean(v2)


DEFAULT_DIR = '20_newsgroup'
EXCLUDED = set(string.punctuation)         # Punctuation set.
STOPWORD_LIST = []                         # Terms that are too common to be searchable.
MIN_REQUIRED_FREQUENCY = 3                 # Minimum overall count required for any term to be searchable.
ALL_DOCS = set()
NUM_OF_DOCS = 0

search_dir = os.getcwd() + '/' + DEFAULT_DIR
if DEFAULT_DIR not in os.listdir(os.getcwd()) or not os.path.isdir(search_dir):
    print('Sorry, directory ' + DEFAULT_DIR + ' not found.')
    exit(1)

groups = os.listdir(search_dir)
print('Inverted index being created. Will take approximately 8 minutes. Progress displayed:\n')
parent_dictionary = {}
document_word_counts = collections.defaultdict(lambda :0)

for group in groups:
    group_dir = search_dir + '/' + group
    files = os.listdir(group_dir)
    for file in files:
        file_path = group_dir + '/' + file
        if not os.path.isfile(file_path):
            continue
        text = open(file_path, 'r').read().lower().replace('.', ' ')  # Down-casing, removing full-stops.
        text = ''.join(ch for ch in text if ch not in EXCLUDED)       # Filtering out punctuation.
        terms = ' '.join(filter(lambda x: x.lower() not in STOPWORD_LIST, text.split())).split() # Removing the stop-list words.
        NUM_OF_DOCS += 1
        filename = group + '/' + file
        document_word_counts[filename] += len(terms)

        for term in terms:
            stemmed = stem(term)
            if stemmed not in parent_dictionary.keys():
                parent_dictionary[stemmed] = {}
            child_dictionary = parent_dictionary[stemmed]
            if filename in child_dictionary.keys():
                child_dictionary[filename] += 1
            else:
                child_dictionary[filename] = 1
            parent_dictionary[stemmed] = child_dictionary

    print(group + ' done !')

# Clearing the terms that have a low word collection count.
for k in parent_dictionary.keys():
    if sum(parent_dictionary[k].values()) < MIN_REQUIRED_FREQUENCY:
        parent_dictionary[k].clear()

print('Done ! \n')
query = ['learning', 'life']
query_vector = []
idf_dict = {}
considerable_documents = []

for term in query:
    idf = 0.0
    if term in parent_dictionary.keys():
        idf = 1 + math.log10(NUM_OF_DOCS / float(len(parent_dictionary[term])))
        considerable_documents += parent_dictionary[term].keys()
    idf_dict[term] = idf
    tf = query.count(term) / float(len(query))
    query_vector.append(tf * idf)

cosine_similarities = {}
document_vectors = collections.defaultdict(list)

for term in query:
    idf = idf_dict[term]
    for doc in considerable_documents:
        tf = 0.0
        if term in parent_dictionary.keys() and doc in parent_dictionary[term].keys():
            tf = parent_dictionary[term][doc] / float(document_word_counts[doc])
        document_vectors[doc].append(tf * idf)

for doc in document_vectors.keys():
    cosine_similarities[doc] = calculate_cosine_similarity(query_vector, document_vectors[doc])

top_sims = sorted(cosine_similarities.values(), reverse=True)
for cs in cosine_similarities.keys():
    if cosine_similarities[cs] in top_sims[:5]:
        print(cs)

