r'''
calculation of pairwise distance, and return condensed result, i.e. we omit the diagonal and duplicate entries and store everything in a one-dimensional array
'''
import torch

def pairwise_distance(data1, data2=None, device=-1):
	r'''
	using broadcast mechanism to calculate pairwise ecludian distance of data
	the input data is N*M matrix, where M is the dimension
	we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
	then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
	'''
	if data2 is None:
		data2 = data1 

	if device!=-1:
		data1, data2 = data1.cuda(device), data2.cuda(device)

	#N*1*M
	A = data1.unsqueeze(dim=1)

	#1*N*M
	B = data2.unsqueeze(dim=0)

	dis = (A-B)**2.0
	#return N*N matrix for pairwise distance
	dis = dis.sum(dim=-1).squeeze()
	return dis

def group_pairwise(X, groups, device=0, fun=lambda r,c: pairwise_distance(r, c).cpu()):
	group_dict = {}
	for group_index_r, group_r in enumerate(groups):
		for group_index_c, group_c in enumerate(groups):
			R, C = X[group_r], X[group_c]
			if device!=-1:
				R = R.cuda(device)
				C = C.cuda(device)
			group_dict[(group_index_r, group_index_c)] = fun(R, C)
	return group_dict

