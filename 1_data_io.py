import torch
from torchvision import datasets,transforms,utils
from torch.utils import data
import numpy as np
import helpers
from sklearn.model_selection import train_test_split
torch.multiprocessing.set_sharing_strategy('file_system')
import tqdm


class Data_IO():
	def __init__(self,samples_per_class,batch_size,imgnet_normalize=False,dataset='mnist',unlab_samples_per_class=1000):

		assert dataset in ('mnist','cifar10')
		self.root = 'data/%s/'%dataset
		self.dataset = dataset
		self.img_sz = 32
		self.samples_per_class = samples_per_class
		self.batch_size = batch_size
		self.imgnet_normalize = imgnet_normalize
		self.seed = 42
		self.unlab_samples_per_class = unlab_samples_per_class

	def get_dataset(self,split,verbose=0):
		if self.dataset == 'mnist':
			normalize = transforms.Normalize(mean=[.5,],std=[.5])
		else:
			normalize = transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
		transform = transforms.Compose([
				transforms.Resize(self.img_sz),
				transforms.CenterCrop(self.img_sz),
				transforms.ToTensor(),
				normalize,
			])
		if split == 'train' or split == 'valid':
			if self.dataset == 'mnist':
				dataset = datasets.MNIST(root=self.root, train=True, transform=transform, target_transform=None, download=True)
				y = dataset.train_labels.numpy()
			else:
				dataset = datasets.CIFAR10(root=self.root, train=True, transform=transform, target_transform=None, download=True)
				y = dataset.train_labels
			X = np.arange(len(y)) ; 
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=self.seed)
			if split == 'train':
				dataset = data.Subset(dataset, X_train)
			else:
				dataset = data.Subset(dataset, X_test)

		elif split == 'test':
			if self.dataset == 'mnist':
				dataset = datasets.MNIST(root=self.root, train=False, transform=transform, target_transform=None, download=True)
			else:
				dataset = datasets.CIFAR10(root=self.root, train=False, transform=transform, target_transform=None, download=True)
		if verbose > 0:
			labels = torch.tensor([y for x,y in dataset])
			labels = labels.numpy().astype(np.uint8)
			print(split,len(dataset))
			print(np.unique(labels,return_counts=True))
		return dataset

	def get_dataloader(self,split,verbose=0):
		if split == 'all_train':
			dataset = self.get_dataset(split='train') ; shuffle = True ;
			if self.unlab_samples_per_class != -1:
				labels = torch.tensor([y for x,y in dataset])
				indices = torch.arange(len(labels))
				indices = torch.cat([indices[labels==x][:self.unlab_samples_per_class] for x in torch.unique(labels)])
				dataset = data.Subset(dataset, indices)

		elif split == 'lab_train':
			dataset = self.get_dataset(split='train') ; shuffle = True ;
			if self.samples_per_class != -1:
				labels = torch.tensor([y for x,y in dataset])
				indices = torch.arange(len(labels))
				indices = torch.cat([indices[labels==x][:self.samples_per_class] for x in torch.unique(labels)])
				dataset = data.Subset(dataset, indices)

		elif split == 'test':
			dataset = self.get_dataset(split='test') ; shuffle = False ;

		elif split == 'valid':
			dataset = self.get_dataset(split='valid') ; shuffle = False ;

		dataloader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, sampler=None, batch_sampler=None, num_workers=16)
		
		if verbose > 0:
			print(split,len(dataloader))
			labels = torch.cat([y for x,y in dataloader])
			labels = labels.numpy()
			print(np.unique(labels,return_counts=True))
		return dataloader

	def write_imgs(self,imgs,path):
		imgs = imgs[:25]
		make_grid = utils.save_image(imgs, path, nrow=5, padding=1, normalize=True, range=None, scale_each=False, pad_value=0)

	def create_infinite_dataloader(self,dataloader):
		data_iter = iter(dataloader)
		while True:
			try:
				yield next(data_iter)
			except StopIteration:
				data_iter = iter(dataloader)

if __name__ == '__main__':

	dataset_name = 'mnist'
	# dataset_name = 'cifar10'
	io = Data_IO(samples_per_class=100,batch_size=50,dataset=dataset_name,unlab_samples_per_class=1000)

	dataset = io.get_dataset(split='train',verbose=1)
	dataset = io.get_dataset(split='valid',verbose=1)
	dataset = io.get_dataset(split='test',verbose=1)

	dataloader = io.get_dataloader(split='lab_train',verbose=1)
	dataloader = io.get_dataloader(split='valid',verbose=1)
	dataloader = io.get_dataloader(split='test',verbose=1)
	dataloader = io.get_dataloader(split='all_train',verbose=1)

	disp_dir = 'tmp/%s/'%(dataset_name)
	helpers.clear_folder(disp_dir)
	for i,(x,y) in enumerate(dataloader):
		io.write_imgs(x,path=disp_dir+str(i).zfill(4)+'.jpg')
		if i<1:
			print(i,x.shape,y)