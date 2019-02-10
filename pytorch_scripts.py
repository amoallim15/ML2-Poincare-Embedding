import nltk
from random import randint
from nltk.corpus import wordnet as wn
from collections import defaultdict as ddict
from itertools import count
import torch as th
from torch import nn
from torch.autograd import Function, Variable
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def downloadNLTK():
	nltk.download('wordnet')

def get_target(noun):
	return wn.synsets(noun, pos= 'n')

def generate_synsets(targets, fname):
	edges = set()
	for synset in wn.all_synsets(pos = 'n'):
		for path in synset.hypernym_paths():
			for hyper in path:
				if hyper in targets:
					pos1 = targets.index(hyper)
					pos2 = path.index(targets[pos1])
					for i in range(pos2, len(path) - 1):
						edges.add((synset.name(), path[i].name()))
	with open(fname, 'w') as fout:
		for i, j in edges:
			fout.write('{0}\t{1}\n'.format(i, j))
	pass

def iter_line(fname):
	with open(fname, 'r') as fin:
		for line in fin:
			d = line.strip().split('\t')
			yield tuple(d)

def get_data(fname):
	ecount = count()
	enames = ddict(ecount.__next__)
	rows = []
	relations = ddict(dict)
	for i, j in iter_line(fname):
		rows.append((enames[i], enames[j]))
		relations[enames[i]][enames[j]] = True
	ids = th.LongTensor(rows)
	objects = dict(enames)
	print('data: objects={0}, edges={1}'.format(len(objects), len(ids)))
	return ids, objects, relations

class PoincareDataset(Dataset):
	def __init__(self, ids, objects, relations, negs, unigram_size=1e8):
		self.ids = ids
		self.objects = objects
		self.negs = negs
		self.max_tries = self.negs * 10
		self.relations = relations
		pass
	def __len__(self):
		return len(self.ids)

	def __getitem__(self, index):
		t, h = self.ids[index]
		negids = set()
		for i in range(self.max_tries):
			if len(negids) >= self.negs:
				break
			idx = randint(0, len(self.objects) - 1)
			if idx not in self.relations[t.item()]:
				negids.add(idx)
		indexes = [t, h] + list(negids)
		if len(negids) == 0:
			negids.add(t)
		while len(indexes) < self.negs + 2:
			indexes.append(indexes[randint(2, len(negids))])
		return th.tensor(indexes).long(), th.zeros(1) #.long()

class PoincareModule(nn.Module):

	def __init__(self, size, dim, scale, lr, eps):
		super(PoincareModule, self).__init__()
		self.lossfn = nn.CrossEntropyLoss(reduction='mean', weight=None)
		self.embeds = nn.Embedding(size, dim, max_norm = 1, scale_grad_by_freq = False)
		self.embeds.weight.data.uniform_(-scale, scale)
		self.lr = lr
		self.eps = eps
		pass

	def forward(self, inputs):
		# e = relation vectors x batch_size
		e = self.embeds(inputs)
		v = e.narrow(1, 1, e.size(1) - 1)
		u = e.narrow(1, 0, 1).expand_as(v)
		dists = self.distance(u, v)
		return dists

	def loss(self, preds, targets):
		dist_uv = preds.narrow(1, 0, 1)
		negs_dist = preds.narrow(1, 1, preds.size(1) - 1)
		loss = -1 * th.log(th.exp(-1 * dist_uv).squeeze()/th.exp(-1 * negs_dist).sum(1)).unsqueeze(1).mean()
		#print(loss)
		#loss2 = self.lossfn(preds, targets.squeeze(1).long())
		#print(loss2)
		return loss

	def optimize(self):
		for e in self.parameters():
			ee = th.sum(e * e, dim = -1, keepdim=True)
			alpha = 1 - ee
			beta = -self.lr * e.grad.data * (alpha ** 2 / 4)
			en = th.norm(e.data)
			if en >= 1:
				e.data.add_(beta/en + self.eps) #.expand_as(edx)
			else:
				e.data.add_(beta)

	def arcosh(self, x):
		return th.log(th.clamp(x + th.sqrt(th.clamp(th.pow(x, 2) - 1, min = self.eps)), min = self.eps))

	def distance(self, u, v):
		uu = th.sum(u * u, dim=-1)
		vv = th.sum(v * v, dim=-1)
		u_v = th.sum(th.pow(u - v, 2), dim=-1)
		alpha = 1 - uu
		beta = 1 - vv
		gamma = 1 + 2 * u_v / th.clamp(alpha * beta, min = self.eps)
		return self.arcosh(gamma)
