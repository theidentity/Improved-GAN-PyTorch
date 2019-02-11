import os
from joblib import Parallel, delayed
import time


def run_cmd_sup(gpu,labels,seed):
	cmd = 'CUDA_VISIBLE_DEVICES=%d python 2_sup_baseline.py --labels=%d --seed=%d'%(gpu,labels,seed)
	print(cmd)
	os.system(cmd)

def run_cmd_ssl(gpu,labels,seed):
	cmd = 'CUDA_VISIBLE_DEVICES=%d python 3_ssl_gan.py --labels=%d --seed=%d'%(gpu,labels,seed)
	print(cmd)
	os.system(cmd)

def run_sup_baselines():
	for labels in [10,50,100,250,500,750,1000]:
		Parallel(n_jobs=4)(delayed(run_cmd_sup)(gpu=gpu,labels=labels,seed=seed) for gpu,seed in zip([0,1,2,3],[10,42,1337,2019]))

def run_ssl():
	for labels in [10,50,100,250,500,750,1000]:
		Parallel(n_jobs=4)(delayed(run_cmd_ssl)(gpu=gpu,labels=labels,seed=seed) for gpu,seed in zip([0,1,2,3],[10,42,1337,2019]))

if __name__ == '__main__':
	# run_sup_baselines()
	# run_ssl()
	pass