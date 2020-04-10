import torch
import numpy as np
from kmeans_pytorch.pairwise import pairwise_distance

def forgy(X, n_clusters):
	X=X.unique(dim=0)
	_len = len(X)

	indices = np.random.choice(_len, n_clusters,replace=False)
	initial_state = X[indices]
	return initial_state


def lloyd(X, n_clusters, device=0, tol=1e-4):
	X = torch.from_numpy(X).float().cuda(device)

	initial_state = forgy(X, n_clusters)


	while True:
		dis = pairwise_distance(X, initial_state)

		choice_cluster = torch.argmin(dis, dim=1)

		initial_state_pre = initial_state.clone()

		for index in range(n_clusters):
			selected = torch.nonzero(choice_cluster==index).squeeze()

			selected = torch.index_select(X, 0, selected)
			initial_state[index] = selected.mean(dim=0)
		

		center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))

		if torch.isnan(center_shift):
			return False,None,None

		if center_shift ** 2 < tol:
			break

	return True,choice_cluster.cpu().numpy(), initial_state.cpu().numpy()