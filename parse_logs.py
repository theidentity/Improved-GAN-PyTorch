import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def parse_txt(path):
	lines = open(path,'r').read().split('\n')
	acc = lines[0].split(' ')[-1]
	acc = float(acc)
	return acc

def parse_logs(which):
	print(which)
	res = []
	labels = [10,50,100,250,500,750,1000]
	seeds = [10,42,1337,2019]
	df = pd.DataFrame(index=labels,columns=seeds)
	df.index.names = ['labels']
	for lab in labels:
		for seed in seeds:
			try:
				acc = parse_txt('metrics/%s_lab_cifar10_%d_seed%d.txt'%(which,lab,seed))
			except:
				acc = -1
			df[seed][lab] = acc
	# res = np.array(res).reshape(-1,4)
	# print(res)
	df['mean'] = df.mean(axis=1)
	df['std'] = df.std(axis=1)
	print(df)
	df.to_csv('metrics/%s_results.csv'%(which))


def plot_graphs():
	which = 'ssl'
	df = pd.read_csv('metrics/%s_results.csv'%(which),header=0, index_col='labels')
	mean_ssl = df['mean'].values
	std_ssl = df['std'].values

	which = 'sup'
	df = pd.read_csv('metrics/%s_results.csv'%(which),header=0, index_col='labels')
	mean_sup = df['mean'].values
	std_sup = df['std'].values
	x = df.index.values

	plt.plot(x,mean_ssl,label='semi_supervised_gan',color='red')
	plt.plot(x,mean_sup,label='supervised',color='blue')
	# plt.errorbar(x, mean_ssl, yerr=std_ssl, fmt='-', label='semi_supervised_gan',color='red')
	# plt.errorbar(x, mean_sup, yerr=std_sup, fmt='-', label='supervised',color='blue')
	plt.legend()
	plt.grid()
	plt.xlabel('number of labelled samples')
	plt.ylabel('accuracy')
	plt.savefig('graphs/ssl_sup_compare.png')


if __name__ == '__main__':
	# parse_logs('sup')
	parse_logs('ssl')
	# plot_graphs()