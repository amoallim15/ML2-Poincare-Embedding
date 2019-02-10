import torch as th
from torch import nn
import torch.optim as optim
import pytorch_scripts as ps
import timeit
from torch.autograd import Variable

use_cuda = th.cuda.is_available()
device = th.device("cuda:0" if use_cuda else "cpu")
device = 'cpu'

targets = ps.get_target('brown')
path = 'data/data.tsv'
max_epochs = 100
num_workers = 6
dim = 5
batch_size = 5
eps = 1e-5
neg = 10
scale = 0.001
lr = 0.01
#ps.downloadNLTK()
#ps.generate_synsets(targets, path)

params = {
	'batch_size': batch_size, 
	'shuffle': True, 
	'num_workers': num_workers
}

ids, objects, relations = ps.get_data(path)
data = ps.PoincareDataset(ids, objects, relations, neg)
model = ps.PoincareModule(len(objects), dim, scale, lr, eps)
loader = th.utils.data.DataLoader(data, **params)

for epoch in range(max_epochs):
	epoch_loss = []
	loss = None
	# inputs = [v, u, negs ....] ids x batch_size
	# targets = [0] tensor x batch_size
	for inputs, targets in loader:
		preds = model(inputs)
		loss = model.loss(preds, targets)
		loss.backward()
		model.optimize()
		epoch_loss.append(loss.data.item())

	print('epoch loss: {0}'.format(th.tensor(epoch_loss).mean()))

print('embedding vectors:\n{0}'.format(model.embeds.weight))


