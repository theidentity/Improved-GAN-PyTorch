import torch.nn as nn
import torch
import torchvision
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
from sklearn import metrics
import argparse

data_io = __import__('1_data_io')
import networks
import helpers


class SupervisedClassfier():
	def __init__(self,samples_per_class,seed,gpu,dataset):

		self.num_classes = 10
		self.batch_size = 100
		self.samples_per_class = samples_per_class
		self.io = data_io.Data_IO(self.samples_per_class,self.batch_size,dataset=dataset)
		self.lr = 1e-3
		self.early_stopping_patience = 15

		self.dataset = dataset
		self.name = 'sup_lab_%s_%d_seed%d'%(dataset,samples_per_class,seed)
		self.best_save_path = 'models/%s/best/'%(self.name)
		self.last_save_path = 'models/%s/last/'%(self.name)
		self.device = 'cuda:%d'%(gpu)
		self.seed = seed
		torch.manual_seed(self.seed)
		
		self.writer = SummaryWriter('logs/%s/'%(self.name))

	def get_model(self):

		if self.dataset == 'mnist':
			__,D = networks.get_mnist_gan_networks(latent_dim=100,num_classes=self.num_classes)
		elif self.dataset == 'cifar10':
			__,D = networks.get_cifar_gan_networks(latent_dim=100,num_classes=self.num_classes)
		D = D.cuda()
		return D

	def get_dataloader(self,split):
		assert split in ('all_train','lab_train','test','valid')
		return self.io.get_dataloader(split=split)

	def train(self,num_epochs):
		model = self.get_model().cuda()
		train_loader = self.get_dataloader(split='lab_train')
		valid_loader = self.get_dataloader(split='valid')
		helpers.clear_folder(self.best_save_path)
		helpers.clear_folder(self.last_save_path)
		
		# criterion = nn.NLLLoss().cuda()
		criterion = nn.CrossEntropyLoss().cuda()
		opt = torch.optim.Adam(model.parameters(), lr=self.lr)
		# opt = torch.optim.SGD(model.parameters(), lr=self.lr, nesterov=True, momentum=.9,weight_decay=1e-6)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=8, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
		max_val_loss = None
		no_improvement = 0

		global_train_step = 0
		global_test_step = 0

		for epoch_idx in range(num_epochs):
			
			train_loss = 0.0
			model.train()
			for x,y in tqdm(train_loader):
				x = x.cuda(); y = y.cuda() ;
				opt.zero_grad()
				__,logits = model(x)
				loss = criterion(logits,y)
				self.writer.add_scalar('train_loss',loss,global_train_step)
				global_train_step += 1
				loss.backward() ; opt.step() ; 
				train_loss += loss.item()
			train_loss /= len(train_loader)

			val_loss = num_correct = total_samples = 0.0
			with torch.no_grad():
				model.eval()
				for x,y in tqdm(valid_loader):
					x = x.cuda(); y = y.cuda();
					__,logits = model(x)
					loss = criterion(logits,y)
					self.writer.add_scalar('val_loss',loss,global_test_step)
					global_test_step += 1
					val_loss += loss.item()
					pred = torch.argmax(logits,dim=1)
					num_correct += torch.sum(pred==y)
					total_samples += len(y)
	
				val_loss /= len(valid_loader)
				acc = num_correct.item() / total_samples

			print('Epoch %d train_loss %.3f val_loss %.3f acc %.3f'%(epoch_idx,train_loss,val_loss,acc))
			scheduler.step(val_loss)

			if max_val_loss is None:
				max_val_loss = val_loss + 1
			
			no_improvement += 1
			if val_loss < max_val_loss:
				no_improvement = 0
				max_val_loss = val_loss
				print('Best model updated - Loss reduced to :',max_val_loss)
				torch.save(model.state_dict(), self.best_save_path+'disc.pth')

			torch.save(model.state_dict(), self.last_save_path+'disc.pth')

			if no_improvement > self.early_stopping_patience:
				print('Early Stopping')
				break

		self.writer.close()

	def get_pred(self,use_saved):
		if not use_saved:
			model = self.get_model().cuda()
			model.load_state_dict(torch.load(self.best_save_path+'disc.pth'))
			model.eval()

			test_loader = self.get_dataloader(split='test')
			y_scores = torch.empty((len(test_loader)*self.batch_size,self.num_classes)).cuda()
			y_true = torch.empty((len(test_loader)*self.batch_size,)).cuda()
			
			first_idx = 0
			with torch.no_grad():
				for x,y in tqdm(test_loader):
					x = x.cuda();y = y.cuda();
					__,logits = model(x)
					y_scores[first_idx:first_idx+len(y)] = logits
					y_true[first_idx:first_idx+len(y)] = y
					first_idx += len(y)

			y_scores = y_scores[:first_idx].cpu().numpy()
			y_true = y_true[:first_idx].cpu().numpy()
			np.savez_compressed('tmp/%s.npz'%(self.name),y_true=y_true,y_scores=y_scores)
			return y_true,y_scores
		else:
			data = np.load('tmp/%s.npz'%(self.name))
			return data['y_true'],data['y_scores']

	def evaluate(self,use_saved=False):
		y_true,y_scores = self.get_pred(use_saved)
		# y_scores = np.exp(y_scores)
		y_pred = np.argmax(y_scores,axis=1)
		acc = metrics.accuracy_score(y_true,y_pred)
		# cm = metrics.confusion_matrix(y_true,y_pred)
		# print(cm)
		print('Model : %s Acc %.3f'%(self.name,acc))
		log_file = open('metrics/%s.txt'%(self.name),'w')
		print('Model : %s Acc %.3f'%(self.name,acc),file=log_file)
		log_file.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu',default=0)
	parser.add_argument('--seed',default=42)
	parser.add_argument('--labels',default=100)
	parser.add_argument('--dataset',default='mnist')
	args = parser.parse_args()

	seed = int(args.seed)
	gpu = int(args.gpu)
	labels = int(args.labels)
	dataset = args.dataset

	sup = SupervisedClassfier(samples_per_class=labels,gpu=gpu,seed=seed,dataset=dataset)
	sup.train(num_epochs=200)
	sup.evaluate(use_saved=False)