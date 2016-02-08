__author__ = 'Usama Ashraf'

import os
import string
from stemming.porter2 import stem
import collections
import math
import re

def calculate_cosine_similarity(v1, v2):
    euclidean = lambda x: math.sqrt(sum(i**2 for i in x))
    return sum(list(map(lambda x,y: x * y, v1, v2))) / euclidean(v1) * euclidean(v2)

def create_term_doc_matrix(relative_dir, exclude_set, stop_words, min_frequency):
    """ Returns the matrix, the corpus document count and a dictionary holding document word counts. """
    search_dir = os.getcwd() + '/' + relative_dir
    if relative_dir not in os.listdir(os.getcwd()) or not os.path.isdir(search_dir):
        print('Sorry, directory ' + relative_dir + ' not found.')
        return -1
    print('Term-document matrix being created. Please wait...\n')
    _parent_dictionary = {}
    _document_word_counts = collections.defaultdict(lambda :0)
    num_of_docs = 0

    for (_dir, _, files) in os.walk(relative_dir):
        for filename in files:

            file_with_path = os.path.join(_dir, filename)
            text = open(file_with_path, 'r').read().lower().replace('.', ' ')  # Down-casing, removing full-stops.
            text = ''.join(ch for ch in text if ch not in exclude_set)         # Filtering out punctuation.
            terms = ' '.join(filter(lambda x: x.lower() not in stop_words, text.split())).split() # Removing the stop-list words.
            num_of_docs += 1
            _document_word_counts[file_with_path] += len(terms)

            for _term_ in terms:
                stemmed = stem(_term_)
                if stemmed not in _parent_dictionary.keys():
                    _parent_dictionary[stemmed] = {}
                child_dictionary = _parent_dictionary[stemmed]
                if file_with_path in child_dictionary.keys():
                    child_dictionary[file_with_path] += 1
                else:
                    child_dictionary[file_with_path] = 1
                _parent_dictionary[stemmed] = child_dictionary

    # Clearing the terms that have a low overall count.
    for k in _parent_dictionary.keys():
        if sum(_parent_dictionary[k].values()) < min_frequency:
            _parent_dictionary[k].clear()

    return _parent_dictionary, num_of_docs, _document_word_counts


DEFAULT_DIR = '20_newsgroup'
DIR_RELEVANT_STORAGE = {}                  # A dictionary for keeping a directory's metadata once indexed.
EXCLUDED = set(string.punctuation)         # Punctuation set.
STOPWORD_LIST = []                         # Terms that are too common to be searchable.
MIN_REQUIRED_FREQUENCY = 3                 # Minimum overall count required for any term to be searchable.

print('Please enter queries in the exact specified format and with double quotes.\n'
      'If the given relative directory has not been indexed, it will first, preceding the search.')

while True:
    _input = input('\nEnter a query e.g "Learn life" 20_newsgroup : ')
    query = re.findall(r'"([^"]*)"', _input)[0]
    specified_dir = _input.split('"' + query + '"').pop().split()[0]
    query = query.split()

    if specified_dir not in DIR_RELEVANT_STORAGE.keys():
        result = create_term_doc_matrix(specified_dir, EXCLUDED, STOPWORD_LIST, MIN_REQUIRED_FREQUENCY)
        if result == -1:
            continue
        DIR_RELEVANT_STORAGE[specified_dir] = result

    parent_dictionary, NUM_OF_DOCS, document_word_counts = DIR_RELEVANT_STORAGE[specified_dir]
    print('Searching... ! \n')
    query_vector = []
    idf_dict = {}
    considerable_documents = []
    for _term in query:
        term = stem(_term)
        idf = 0.0
        if term in parent_dictionary.keys():
            idf = 1 + math.log10(NUM_OF_DOCS / float(len(parent_dictionary[term])))
            considerable_documents += parent_dictionary[term].keys()
        idf_dict[term] = idf
        tf = query.count(term) / float(len(query))
        query_vector.append(tf * idf)

    cosine_similarities = {}
    document_vectors = collections.defaultdict(list)

    for _term in query:
        term = stem(_term)
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
