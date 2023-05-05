# pagerank.py
# Implement the pagerank algorithm in python.
# Source: https://en.wikipedia.org/wiki/PageRank
# Source: https://www.geeksforgeeks.org/page-rank-algorithm-
#	implementation/
# Python 3.7
# Windows/MacOS/Linux


import json
import networkx as nx
import numpy as np


def pagerank(G, alpha=0.85, personalization=None, max_iter=100, 
		tol=1.0e-6, nstart=None, weight='weight', dangling=None):
	# Return the PageRank of the nodes in the graph.
	# PageRank computes a ranking of the nodes in the graph G based on
	# the structure of the incoming links. It was originally designed
	# as an algorithm to rank web pages.
	# @param: G (networkx graph), a graph. Undirected graphs will be
	#	converted to a directed graph with two directed edges for each
	#	undirected edge.
	# @param: alpha (optional float), dampening parameter for PageRank.
	#	Defualt is 0.85.
	# @param: personalization (optional dict), the "personalization 
	#	vector" consisting of a dictionary with a key for every graph
	#	node and nonzero personalization value for each node. By 
	#	default, a uniform distribution is used.
	# @param: max_iter (int optional), maximum number of iterations in
	#	power method eigenvalue solver.
	# @param: tol (optional float), error tolerance used to check
	#	convergence in power method solver.
	# @param: nstart (optional dict), starting value of PageRank 
	#	iteration for each node.
	# @param: weight (optional key), edge data key to use as weight. If
	#	None weights are set to 1.
	# @param: dangling (optional dict), the outedges to be assigned to 
	#	any "dangling" nodes, i.e., nodes without any outedges. The
	#	dict key is the node the outedge points to and the dict value
	#	is the weight of that outedge. By default, dangling nodes are
	#	given outedges according to the personalization vector (uniform
	#	if not specified). This must be selected to result in an 
	#	irreducible transition matrix (see notes under google_matrix).
	#	It may be common to have the dangling dict to be the same as
	#	the personalization dict.
	# @retrun: returns pagerank (dict), dictionary of nodes with 
	#	PageRank as value.

	# Notes
	# -----
	# The eigenvector calculation is done by the power iteration method
	# and has no guarantee of convergence. The iteration will stop
	# after max_iter iterations or an error tolerance of
	# number_of_nodes(G)*tol has been reached.
	# The PageRank algorithm was designed for directed graphs but this
	# algorithm does not check if the input graph is directed and will
	# execute on undirected graphs by converting each edge in the
	# directed graph to two edges.

	# Return empty dictionary if the graph is empty.
	if len(G) == 0:
		return {}

	# Convert graph to directed graph.
	if not G.is_directed():
		D = G.to_directed()
	else:
		D = G

	# Create a copy in (right) stochastic form.
	W = nx.stochastic_graph(D, weight=weight)
	N = W.number_of_nodes()

	# Choose fixed starting vector if not given.
	if nstart is None:
		x = dict.fromkeys(W, 1.0 / N)
	else:
		# Normalized nstart vector.
		s = float(sum(nstart.values()))
		x = dict((k, v / s) for k, v in nstart.items())

	if personalization is None:
		# Assign uniform personalization vector if not given.
		p = dict.fromkeys(W, 1.0 / N)
	else:
		missing = set(G) - set(personalization)
		if missing:
			raise nx.NetworkXError(
				'Personalization dictionary '
				'must have a value for every node. '
				'Missing nodes %s' % missing
			)
		s = float(sum(personalization.values()))
		p = dict((k, v / s) for k, v in personalization.items())

	if dangling is None:
		# Use personalization vector if dangling vector not specified.
		dangling_weights = p
	else:
		missing = set(G) - set(dangling)
		if missing:
			raise nx.NetworkXError(
				'Dangling node dictionary '
				'must have a value for every node. '
				'Missing nodes %s' % missing
			)
		s = float(sum(dangling.values()))
		dangling_weights = dict((k, v/s) for k, v in dangling.items())
	dangling_nodes = [
		n for n in W if W.out_degree(n, weight=weight) == 0.0
	]

	# power iteration: make up to max_iter iterations.
	for _ in range(max_iter):
		xlast = x
		x = dict.fromkeys(xlast.keys(), 0)
		danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
		for n in x:

			# this matrix multiply looks odd because it is
			# doing a left multiply x^T=xlast^T*W.
			for nbr in W[n]:
				x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
			x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]

		# check convergence, l1 norm.
		err = sum([abs(x[n] - xlast[n]) for n in x])
		if err < N*tol:
			return x

	raise nx.NetworkXError(
		'pagerank: power iteration failed to converge '
		'in %d iterations.' % max_iter
	)


def pagerank_numpy(M, num_iterations: int = 100, d: float = 0.85):
	# PageRank algorithm with explicit number of iterations. Returns 
	# ranking of nodes (pages) in the adjacency matrix.
	# @param: M (numpy array), adjacency matrix where M_i,j represents
	#	the link from 'j' to 'i', such that for all 'j' sum(i, M_i,j) 
	#	= 1.
	# @param: num_iterations (optional int), number of iterations, by
	#	default.
	# @param: d (optional float), damping factor, by default 0.85.
	# @return: retruns numpy array which is a vector of ranks such that
	#	v_i is the i-th rank from [0, 1], v sums to 1.
	N = M.shape[1]
	v = np.ones(N) / N
	M_hat = (d * M + (1 - d) / N)
	for i in range(num_iterations):
		v = M_hat @ v
	return v


def main():
	# Initialize a random graph (using Barabasi-Albert
	# preferential attachment).
	G = nx.barabasi_albert_graph(60,41)

	# Get the pagerank of each node in the graph.
	pr = nx.pagerank(G,0.4)
	print(json.dumps(pr, indent=4))

	# Initialize a new graph through an adjacency matrix.
	M = np.array(
		[[0, 0, 0, 0, 1],
		[0.5, 0, 0, 0, 0],
		[0.5, 0, 0, 0, 0],
		[0, 1, 0.5, 0, 0],
		[0, 0, 0.5, 1, 0]]
	)

	# Get the pagerank of each node in the graph.
	v = pagerank_numpy(M, 100, 0.85)
	print(v)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()