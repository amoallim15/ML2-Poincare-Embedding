import torch as th
from torch import nn
import torch.optim as optim
import pytorch_scripts as ps
import timeit
import torch.optim as optim
from torch.autograd import Variable

use_cuda = th.cuda.is_available()
device = th.device("cuda:0" if use_cuda else "cpu")
device = 'cpu'

targets = ps.get_target('brown')
path = 'data/data.tsv'
max_epochs = 10
num_workers = 6
dim = 2
batch_size = 25
start_lr = 0.1
final_lr = 0.001
neg = 10
scale = 0.001

#ps.downloadNLTK()
#ps.generate_synsets(targets, path)

params = {
	'batch_size': batch_size, 
	'shuffle': True, 
	'num_workers': num_workers,
	#'collate_fn': ps.collate_fn
}

ids, objects, relations = ps.get_data(path)
data = ps.PoincareDataset(ids, objects, relations, neg)
model = ps.PoincareModule(len(objects), dim, scale)
loader = th.utils.data.DataLoader(data, **params)

print(ids, objects)

for epoch in range(max_epochs):
	epoch_loss = []
	loss = None
	for inputs, targets in loader:
		inputs, targets = inputs.to(device), targets.to(device)
		preds = model(inputs)
		loss = model.loss(preds, targets)
		#loss.backward()
		print(loss)#, th.ones(len(loss)))




