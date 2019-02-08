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
max_epochs = 1
num_workers = 6
dim = 3
batch_size = 2
start_lr = 0.1
final_lr = 0.001
neg = 10
scale = 0.001
lr = 0.01
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

#embeds = nn.Embedding(len(objects), dim, max_norm = 1, sparse = True, scale_grad_by_freq = False)
#embeds.weight.data.uniform_(-scale, scale)
#print(ids, objects)

for epoch in range(max_epochs):
	epoch_loss = []
	loss = None
	# inputs = [v, u, negs ....] ids x batch_size
	# targets = [0] tensor x batch_size
	for inputs, targets in loader:


		preds = model(inputs)
		#print(preds, targets)
		loss = model.loss(preds, targets)
		loss.backward()

		for p in model.parameters():
			d_p = p.grad.data
			ps.poincare_grad(p, d_p)

			print(p)
			#print(lr, '\n', p.data, '\n', d_p)
			p.data.add_(-lr, d_p)
			print('now')
			print(p)

		#print(list(model.parameters())[0].grad)
		#print(loss)








		#inputs, targets = inputs.to(device), targets.to(device)
		#preds = model(inputs)
		#loss = model.loss(preds, targets)
		#loss.backward()
		#optimizer.step(lr = lr)
		#epoch_loss.append(loss.item())



		#e = embeds(inputs)
		#v = Variable(e.narrow(1, 1, e.size(1) - 1), requires_grad=True)
		#u = Variable(e.narrow(1, 0, 1).expand_as(v), requires_grad=True)
		#dists = ps.poincare_distance(u, v)

		#uv_dist = dists.narrow(1, 0, 1)
		#negs_dist = dists.narrow(1, 1, dists.size(1) - 1)
		#loss = th.log(th.exp(-uv_dist).squeeze()/th.exp(-negs_dist).sum(1)).unsqueeze(1)
		#loss.backward(targets)
		#epoch_loss
		#print(v, v.data.grad)
		#print(e)
		
		#print(e)
		#for i in model.state_dict():
		#	print(model.state_dict()[i])
		#print(loss)#, th.ones(len(loss)))
	#print(epoch_loss)

#for i in model.state_dict():
#	print(model.state_dict()[i])


