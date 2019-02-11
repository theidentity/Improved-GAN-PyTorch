import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
from sklearn import metrics
import argparse

data_io = __import__('1_data_io')
import helpers
import networks


class SSL_GAN():
	def __init__(self,samples_per_class,seed,gpu):

		self.num_classes = 10
		self.latent_dim = 100
		self.batch_size = 250
		self.samples_per_class = samples_per_class
		self.io = data_io.Data_IO(self.samples_per_class,self.batch_size)
		self.lr = 1e-3
		self.early_stopping_patience = 15

		self.name = 'ssl_lab%d_seed%d'%(samples_per_class,seed)
		self.best_save_path = 'models/%s/best/'%(self.name)
		self.last_save_path = 'models/%s/last/'%(self.name)
		self.device = 'cuda:%d'%(gpu)
		self.seed = 42
		torch.manual_seed(self.seed)
		
		self.log_dir = 'logs/%s/'%(self.name)
		helpers.clear_folder(self.log_dir)
		self.writer = SummaryWriter(self.log_dir)

	def get_model(self,verbose=0):
		G,D = networks.get_mnist_gan_networks(latent_dim=self.latent_dim,num_classes=self.num_classes)
		G = G.cuda(); D = D.cuda() ;
		if verbose > 0:
			print(G);print(D); 
		return G,D


	def get_dataloader(self,split):
		assert split in ('all_train','lab_train','test','valid')
		return self.io.get_dataloader(split=split)

	def train(self,num_epochs):
		G,D = self.get_model()
		all_train_loader = self.get_dataloader(split='all_train')
		train_loader = self.get_dataloader(split='lab_train')
		lab_train_loader = self.io.create_infinite_dataloader(train_loader)
		valid_loader = self.get_dataloader(split='valid')
		helpers.clear_folder(self.best_save_path)
		helpers.clear_folder(self.last_save_path)

		XE = nn.CrossEntropyLoss().cuda()
		MSE = nn.MSELoss().cuda()

		opt_gen = torch.optim.Adam(G.parameters(), lr=self.lr)
		opt_disc = torch.optim.Adam(D.parameters(), lr=self.lr)
		scheduler_disc = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_disc, mode='min', factor=0.1, patience=8, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
		scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_gen, mode='min', factor=0.1, patience=8, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
		max_val_loss = None
		no_improvement = global_train_step = global_test_step = 0
		fixed_noise = torch.randn(self.batch_size,self.latent_dim)

		for epoch_idx in range(num_epochs):
			
			avg_gen_loss = avg_disc_loss = 0
			G.train(); D.train()
			for unlab_train_x,__ in tqdm(all_train_loader):
				lab_train_x,lab_train_y = next(lab_train_loader)
				
				unl = unlab_train_x.cuda()
				inp = lab_train_x.cuda()
				lbl = lab_train_y.cuda()
				z = torch.randn(self.batch_size,self.latent_dim).cuda()

				# Train Discriminator
				opt_disc.zero_grad()
				gen_inp = G(z)
				__, logits_lab = D(inp)
				layer_fake, logits_gen = D(gen_inp)
				layer_real, logits_unl = D(unl)
				l_unl = torch.logsumexp(logits_unl,dim=1)
				l_gen = torch.logsumexp(logits_gen,dim=1)
				loss_unl = .5 * torch.mean(F.softplus(l_unl)) - .5* torch.mean(l_unl) +.5 * torch.mean(F.softplus(l_gen))
				loss_lab = torch.mean(XE(logits_lab, lbl))
				loss_disc = .5 * loss_lab + .5 * loss_unl
				loss_disc.backward()
				opt_disc.step()
				avg_disc_loss += loss_disc

				# Train Generator
				opt_gen.zero_grad()
				gen_inp = G(z)
				layer_fake, __ = D(gen_inp)
				layer_real, __ = D(unl)
				m1 = torch.mean(layer_real,dim=0)
				m2 = torch.mean(layer_fake,dim=0)
				loss_gen = torch.mean((m1-m2)**2)
				loss_gen.backward()
				opt_gen.step()
				avg_gen_loss += loss_gen

				self.writer.add_scalar('gen_loss',loss_gen,global_train_step)
				self.writer.add_scalar('disc_loss',loss_disc,global_train_step)
				global_train_step += 1
				# print('Loss Gen %.4f Loss Disc %.4f'%(loss_gen,loss_disc))

			avg_gen_loss /= len(all_train_loader)
			avg_disc_loss /= len(all_train_loader)

			val_loss = num_correct = total_samples = 0.0
			with torch.no_grad():
				D.eval()
				for x,y in tqdm(valid_loader):
					x = x.cuda(); y = y.cuda();
					__,logits = D(x)
					loss = XE(logits,y)
					self.writer.add_scalar('val_loss',loss,global_test_step)
					global_test_step += 1
					val_loss += loss.item()
					pred = torch.argmax(logits,dim=1)
					num_correct += torch.sum(pred==y)
					total_samples += len(y)
	
				val_loss /= len(valid_loader)
				acc = num_correct.item() / total_samples

			print('Epoch %d disc_loss %.3f gen_loss %.3f val_loss %.3f acc %.3f'%(epoch_idx,avg_disc_loss,avg_gen_loss,val_loss,acc))
			# print(acc)
			scheduler_gen.step(val_loss)
			scheduler_disc.step(val_loss)

			if max_val_loss is None:
				max_val_loss = val_loss + 1
			
			no_improvement += 1
			if val_loss < max_val_loss:
				no_improvement = 0
				max_val_loss = val_loss
				print('Best model updated - Loss reduced to :',max_val_loss)
				torch.save(D.state_dict(), self.best_save_path+'disc.pth')
				torch.save(G.state_dict(), self.best_save_path+'gen.pth')

				torch.save(D.state_dict(), self.last_save_path+'disc.pth')
				torch.save(G.state_dict(), self.last_save_path+'gen.pth')

			if no_improvement > self.early_stopping_patience:
				print('Early Stopping')
				break

		self.writer.close()

	def get_pred(self,use_saved):
		if not use_saved:
			__,model = self.get_model()
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
	args = parser.parse_args()

	seed = int(args.seed)
	gpu = int(args.gpu)
	labels = int(args.labels)

	ssl = SSL_GAN(samples_per_class=labels,gpu=gpu,seed=seed)
	ssl.train(num_epochs=100)
	ssl.evaluate(use_saved=False)