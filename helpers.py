import os
import shutil
import torch.nn as nn
import torch


def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

class Flatten(nn.Module):
	def forward(self,x):
		return x.view(x.shape[0],-1)

class Reshape(nn.Module):
	def __init__(self, target_shape):
		super(Reshape,self).__init__()
		self.target_shape = (-1,) + target_shape

	def forward(self,x):
		return x.view(self.target_shape)

if __name__ == '__main__':
	img = torch.randn((25,1,32,32))
	print(img.shape)

	model = nn.Sequential(
			# Flatten(),
			Reshape(target_shape=(1,16,64)),
		)
	print(model(img).shape)