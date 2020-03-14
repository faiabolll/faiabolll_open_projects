import zipfile
import numpy as np 
import pandas as pd 
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize.repp import ReppTokenizer
from nltk.tokenize.stanford import StanfordTokenizer
# simple tokenizer is split
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from itertools import product
import pprint
import gc
import time
import json
from scipy.sparse import save_npz, load_npz
import os
import shutil

texts_path = 'C:\\Users\\EGOR\\Projects\\NLP competition\\text_matrixes'

def create_datasets():
	# creating parameters generator
	params_generator = product_params()

	# extracting train dataset from zip file
	with zipfile.ZipFile('nlp-getting-started.zip') as zfile:
		with zfile.open('train.csv') as dd:
			df = pd.read_csv(dd, usecols=['text'])

	for param_id, param_set in enumerate(params_generator):
		process_text(df, param_set, param_id)

def product_params():
	token_params = token_params_maker()
	lemma_params = [{'type':LancasterStemmer}, {'type':PorterStemmer}, {'type':EnglishStemmer}]
	vector_params = vector_params_maker()
	res = product(token_params, lemma_params, vector_params)
	del token_params
	gc.collect()
	return res

def token_params_maker():
	# TweetTokenizer
	tweet_params = list(product([True, False], repeat=3))
	def make_tweet_dict(elem):
		param_names = ['preserve_case', 'reduce_len', 'strip_handles']
		res = {name:elem[i] for i, name in enumerate(param_names)}
		res.update({'type':TweetTokenizer})
		return res
	tweet_params = list(map(make_tweet_dict, tweet_params))

	# ReppTokenizer
	repp_params = {'type':ReppTokenizer}

	# simple tokenizer
	simple_params = {'type':SimpleTokenizer}

	# StanfordTokenizer
	stanford_params = {'type':StanfordTokenizer}

	# concat params
	token_params = tweet_params + [repp_params] + [simple_params] + [stanford_params]
	return token_params

def vector_params_maker():
	# CountVectorizer
	min_df = [1,3,5,10]
	max_df = [100,250,500,750]
	ngram_range = [(1,1), (1,2), (1,3)]
	lowercase = [True, False]
	max_features = [500, 1000, 5000, 10000, 50000]
	# TfidfVectorizer
	norm = ['l1', 'l2']
	# HashingVectorizer
	n_features = [500, 1000, 5000, 10000, 50000]

	# creating **kwargs for CountVectorizer
	count_params = list(product(min_df, max_df, ngram_range, lowercase, max_features))
	# replacing single value by pair argument_name:argument_value
	def make_count_dict(elem):
		param_names = ['min_df', 'max_df', 'ngram_range', 'lowercase', 'max_features']
		res = {name:elem[i] for i, name in enumerate(param_names)}
		res.update({'type':CountVectorizer})
		return res
	count_params = list(map(make_count_dict, count_params))

	# creating **kwargs for TfidfVectorizer
	tfidf_params = list(product(min_df, max_df, ngram_range, lowercase, max_features, norm))
	def make_tfidf_dict(elem):
		param_names = ['min_df', 'max_df', 'ngram_range', 'lowercase', 'max_features']
		res = {name:elem[i] for i, name in enumerate(param_names)}
		res.update({'type':TfidfVectorizer})
		return res
	tfidf_params = list(map(make_tfidf_dict, tfidf_params))

	# creating **kwargs for HashingVectorizer
	hashing_params = list(product(lowercase, n_features, norm))
	def make_hashing_dict(elem):
		param_names = ['lowercase', 'n_features', 'norm']
		res = {name:elem[i] for i, name in enumerate(param_names)}
		res.update({'type':HashingVectorizer})
		return res
	hashing_params = list(map(make_hashing_dict, hashing_params))

	# concat params
	vector_params = count_params + tfidf_params + hashing_params

	return vector_params

def process_text(text, params, param_id):
	# troubles with params
	if param_id <= 1448:
		return 0
	# saving parameters and their id with global variable
	def classname_to_string(d):
		res = dict(d)
		res['type'] = res['type'].__name__
		return res
	params_to_write = list(map(lambda x: classname_to_string(x), params))

	# clear list of params and clear folder with processd texts
	if param_id == 0:
		if os.path.exists('params_list.txt'):
			os.remove('params_list.txt')
			f = open('params_list.txt', 'w'); f.close()	
		if os.path.exists('text_matrixes.txt'):
			shutil.rmtree('text_matrixes.txt')
			os.mkdir('text_matrixes')

	with open('params_list.txt', 'a') as f:
		to_write = str(param_id)+'#'+json.dumps(params_to_write)+'\n'
		f.write(to_write)

	# debugging
	print(param_id, params[0]['type'].__name__, params[1]['type'].__name__, params[2]['type'].__name__)

	def collect_non_type_params(d):
		res = {}
		for k, v in d.items():
			res.update({k:v}) if k != 'type' else res.update({})
		return res

	# tokenizing
	tokenizer = params[0]['type']
	tokenizer = tokenizer(**collect_non_type_params(params[0]))
	text = text['text'].apply(tokenizer.tokenize)

	# lemmatizing
	lemmatizer = params[1]['type']
	lemmatizer = lemmatizer(**collect_non_type_params(params[1]))
	text = text.apply(lambda row: ' '.join([lemmatizer.stem(word) for word in row]))

	# vectorizing
	vectorizer = params[2]['type']
	vectorizer = vectorizer(**collect_non_type_params(params[2]))
	text = vectorizer.fit_transform(text)

	# saving former text as sparse matrix in .npz format
	save_npz(os.path.join(texts_path, str(param_id)), text)

def iter_datasets():
	for text in os.listdir(texts_path):
		yield load_npz(text)

def fit_opt_model(df):
	pass

class SimpleTokenizer():
	def __init__(self, split_char):
		self.split_char = split_char

	def tokenize(self, sent):
		return sent.split(self.split_char)


if __name__ == '__main__':
	# create_datasets() # check
	
	for dataset in iter_datasets:
		print('dora dura')
		fit_opt_model(dataset)
	