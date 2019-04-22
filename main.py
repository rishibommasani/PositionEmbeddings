import torch 
import logging
from pytorch_pretrained_bert import BertModel, OpenAIGPTModel, GPT2Model
from pytorch_pretrained_bert import BertTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer
import nltk
from tqdm import tqdm
import pickle
nltk.download('punkt')
import time
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import datasets
digits = datasets.load_digits()


def fetch_objects():
	bert = BertModel.from_pretrained('bert-base-uncased').embeddings.position_embeddings.weight.data
	gpt = OpenAIGPTModel.from_pretrained('openai-gpt').positions_embed.weight.data
	gpt2 = GPT2Model.from_pretrained('gpt2').wpe.weight.data
	bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	gpt_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
	gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	return {'bert' : bert, 'gpt' : gpt, 'gpt2' : gpt2}, {'bert' : bert_tokenizer, 'gpt' : gpt_tokenizer, 'gpt2' : gpt2_tokenizer}


def visualize(matrix):
	matrix = matrix[:32,]
	print(matrix.shape)
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	X_2d = tsne.fit_transform(matrix)
	plt.scatter([x[0] for x in X_2d], [x[1] for x in X_2d], c = list(range(32)), cmap=plt.cm.get_cmap("jet", 32))
	plt.colorbar(ticks=range(32))
	plt.title('GPT')
	plt.show()

def fetch_sentences():
	pickled = True
	data = []
	if pickled:
		print("Loading data")
		data = pickle.load(open('dump.txt', 'rb'))
		data = data[:1000000]
		return data
	else:
		with open('wikipedia_utf8_filtered_20pageviews.csv', mode = 'r') as f:
			for i, row in tqdm(enumerate(f)):
				index, text = row.split(',', 1)
				sent_text = nltk.sent_tokenize(text)
				data.extend([sent for sent in sent_text])
				if len(data) > 1000000:
					break 
		pickle.dump(data, open('dump.txt', 'wb'))
		print("Filed was pickled in {}".format('dump.txt'))
		exit()


def clean(seq):
	return [token.strip().lower() for token in seq]


def alignment(sents, tokenizers):
	n = len(sents)
	bert, gpt, gpt2 = tokenizers['bert'], tokenizers['gpt'], tokenizers['gpt2']
	stats = {'bert': {'human' : {'correct' : 0, 'total' : 0}, 'gpt' : {'correct' : 0, 'total' : 0}}, 
			 'human' : {'bert' : {'correct' : 0, 'total' : 0}, 'gpt' : {'correct' : 0, 'total' : 0}}, 
			 'gpt': {'human' : {'correct' : 0, 'total' : 0}, 'bert' : {'correct' : 0, 'total' : 0}}}
	for sent in tqdm(sents):
		sbert = clean(bert.tokenize(sent))
		sgpt = clean(gpt.tokenize(sent))
		sgpt = [token.replace('</w>', '') for token in sgpt]
		shuman = clean(sent.split())

		for i, ref in enumerate(sbert):
			if i < 32:
				stats['bert']['human']['total'] += 1
				stats['bert']['gpt']['total'] += 1
				if i < len(shuman):
					stats['bert']['human']['correct'] += int(ref == shuman[i])
				if i < len(sgpt):
					stats['bert']['gpt']['correct'] += int(ref == sgpt[i])

		for i, ref in enumerate(sgpt):
			if i < 32:
				stats['gpt']['human']['total'] += 1
				stats['gpt']['bert']['total'] += 1
				if i < len(shuman):
					stats['gpt']['human']['correct'] += int(ref == shuman[i])
				if i < len(sbert):
					stats['gpt']['bert']['correct'] += int(ref == sbert[i])

		for i, ref in enumerate(shuman):
			if i < 32:
				stats['human']['bert']['total'] += 1
				stats['human']['gpt']['total'] += 1
				if i < len(sbert):
					stats['human']['bert']['correct'] += int(ref == sbert[i])
				if i < len(sgpt):
					stats['human']['gpt']['correct'] += int(ref == sgpt[i])
	print(stats)
	exit()


def main():
	print("Fetching Embeddings")
	position_matrices, tokenizers = fetch_objects()
	# print("Fetching Sentences")
	# sentences = fetch_sentences()
	# print("Computing tokenizer alignment")
	# alignment_stats = alignment(sentences, tokenizers)
	visualize(position_matrices['gpt'])
	exit()


if __name__ == '__main__':
	main()
