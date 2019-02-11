import torch as th
import pytorch_scripts as ps
import timeit, os.path

#use_cuda = th.cuda.is_available()
#device = th.device("cuda:0" if use_cuda else "cpu")

word = 'fruit'
targets = ps.get_target(word)
path = 'data/{0}.tsv'.format(word)
max_epochs = 100
num_workers = 6
dim = 2
batch_size = 5
eps = 1e-5
neg = 10
scale = 0.01
lr = 0.001

params = {
	'batch_size': batch_size, 
	'shuffle': True, 
	'num_workers': num_workers
}

if os.path.isfile(path) == False:
	ps.downloadNLTK()
	ps.generate_synsets(targets, path)

ids, objects, relations = ps.get_data(path)

print('data: objects={0}, edges={1}'.format(len(objects), len(ids)))

data = ps.PoincareDataset(ids, objects, relations, neg)
model = ps.PoincareModule(len(objects), dim, scale, lr, eps)
loader = th.utils.data.DataLoader(data, **params)

for epoch in range(max_epochs):

	epoch_loss = []

	for inputs, targets in loader:
		preds = model(inputs)
		loss = model.loss(preds, targets)
		loss.backward()
		model.optimize()
		epoch_loss.append(loss.data.item())

	print('epoch loss: {0}'.format(th.tensor(epoch_loss).mean()))

print('\nobjects:\n{0}'.format(objects))
print('\nids:\n{0}'.format(ids))
print('\nembedding vectors:\n{0}'.format(model.embeds.weight))

ps.plot_graph(objects, model.embeds, eps)


