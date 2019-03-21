import os
from joblib import Parallel, delayed
import time
import numpy as np


def get_freer_gpu():
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp/gpu_free')
	memory_available = [int(x.split()[2]) for x in open('tmp/gpu_free', 'r').readlines()]
	freer_fpu = np.argmax(memory_available)
	return freer_fpu 

def run_cmd_sup(gpu,labels,seed,dataset):
	cmd = 'CUDA_VISIBLE_DEVICES=%d python 2_sup_baseline.py --labels=%d --seed=%d --dataset=%s'%(gpu,labels,seed,dataset)
	print(cmd)
	# os.system(cmd)

def run_cmd_ssl(gpu,labels,seed,dataset):
	cmd = 'CUDA_VISIBLE_DEVICES=%d python 3_ssl_gan.py --labels=%d --seed=%d --dataset=%s'%(gpu,labels,seed,dataset)
	print(cmd)
	os.system(cmd)

def run_sup_baselines():
	# for labels in [10,50,100,250,500,750,1000]:
	for labels in [100,250,500,750,1000]:
		Parallel(n_jobs=2)(delayed(run_cmd_sup)(gpu=gpu,labels=labels,seed=seed,dataset='cifar10') for gpu,seed in zip([1,2],[2019,2019]))

def run_ssl():
	# for labels in [10,50,100,250,500,750,1000]:
	# for labels in [10,50]:
	for labels in [1000]:
		Parallel(n_jobs=4)(delayed(run_cmd_ssl)(gpu=gpu,labels=labels,seed=seed,dataset='cifar10') for gpu,seed in zip([0,1,2,3],[10,42,1337,2019]))

if __name__ == '__main__':
	# get_freer_gpu()
	run_sup_baselines()
	# run_ssl()
	# pass
