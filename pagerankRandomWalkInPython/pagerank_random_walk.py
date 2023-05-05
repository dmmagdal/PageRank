# pagerank_random_walk.py
# Implement pagerank using random walk.
# Source: https://www.geeksforgeeks.org/random-walk-implementation-
#	python/
# Source: https://www.geeksforgeeks.org/implementation-of-page-rank-
#	using-random-walk-method-in-python/
# Python 3.7
# Windows/MacOS/Linux


import random
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import networkx as nx
import numpy as np


# Add directed edges in graph.
def add_edges(g, pr):
	for each in g.nodes():
		for each1 in g.nodes():
			if (each != each1):
				ra = random.random()
				if (ra < pr):
					g.add_edge(each, each1)
				else:
					continue
	return g

# Sort the nodes.
def nodes_sorted(g, points):
	t = np.array(points)
	t = np.argsort(-t)
	return t


# Distribute points randomly in a graph.
def random_Walk(g):
	rwp = [0 for i in range(g.number_of_nodes())]
	nodes = list(g.nodes())
	r = random.choice(nodes)
	rwp[r] += 1
	neigh = list(g.out_edges(r))
	z = 0
	
	while (z != 10000):
		if (len(neigh) == 0):
			focus = random.choice(nodes)
		else:
			r1 = random.choice(neigh)
			focus = r1[1]
		rwp[focus] += 1
		neigh = list(g.out_edges(focus))
		z += 1
	return rwp


def main():
	# -----------------------------------------------------------------
	# 1D random walk: An elementary example of a random walk is the 
	# random walk on the integer number line, which starts at 0 and at 
	# each step moves +1 or ? 1 with equal probability. 
	# -----------------------------------------------------------------
	# Probability to move up or down.
	prob = [0.05, 0.95]

	# statically defining the starting position.
	start = 2
	positions = [start]

	# creating the random points.
	rr = np.random.random(1000)
	downp = rr < prob[0]
	upp = rr > prob[1]

	for idownp, iupp in zip(downp, upp):
		down = idownp and positions[-1] > 1
		up = iupp and positions[-1] < 4
		positions.append(positions[-1] - down + up)

	# plotting down the graph of the random walk in 1D.
	plt.plot(positions)
	plt.savefig('1D-random-walk.png')
	# plt.show()


	# -----------------------------------------------------------------
	# Higer dimension random walk: In higher dimensions, the set of 
	# randomly walked points has interesting geometric properties. In 
	# fact, one gets a discrete fractal, that is, a set that exhibits 
	# stochastic self-similarity on large scales. On small scales, one 
	# can observe “jaggedness” resulting from the grid on which the 
	# walk is performed. Two books of Lawler are good sources on this 
	# topic. The trajectory of a random walk is the collection of 
	# points visited, considered as a set with disregard to when the 
	# walk arrived at the point. In one dimension, the trajectory is 
	# simply all points between the minimum height and the maximum 
	# height the walk achieved (both are, on average, on the order of ?
	# n). 
	# Below is a 2D random walk.
	# -----------------------------------------------------------------
	# defining the number of steps.
	n = 100000
	
	# creating two array for containing x and y coordinate of size
	# equals to the number of size and filled up with 0's.
	x = np.zeros(n)
	y = np.zeros(n)
	
	# filling the coordinates with random variables.
	for i in range(1, n):
		val = random.randint(1, 4)
		if val == 1:
			x[i] = x[i - 1] + 1
			y[i] = y[i - 1]
		elif val == 2:
			x[i] = x[i - 1] - 1
			y[i] = y[i - 1]
		elif val == 3:
			x[i] = x[i - 1]
			y[i] = y[i - 1] + 1
		else:
			x[i] = x[i - 1]
			y[i] = y[i - 1] - 1
		
	# plotting stuff:
	pylab.title("Random Walk ($n = " + str(n) + "$ steps)")
	pylab.plot(x, y)
	pylab.savefig(
		"rand_walk" + str(n) + ".png", bbox_inches="tight", dpi=600
	)
	# pylab.show()

	# Applications:
	# - In computer networks, random walks can model the number of 
	#	transmission packets buffered at a server.
	# - In population genetics, a random walk describes the 
	#	statistical properties of genetic drift.
	# - In image segmentation, random walks are used to determine the 
	#	labels (i.e., “object” or “background”) to associate with each 
	#	pixel.
	# - In brain research, random walks and reinforced random walks are 
	#	used to model cascades of neuron firing in the brain.
	# Random walks have also been used to sample massive online graphs 
	# such as online social networks.

	# -----------------------------------------------------------------
	# Implementation of pagerank using random walk method.
	# Random Walk Method: 
	# 	In the random walk method we will choose 1 node from the graph 
	# 	uniformly at random. After choosing the node we will look at 
	#	its neighbors and choose a neighbor uniformly at random and 
	#	continue these iterations until convergence is reached. After N
	#	iterations a point will come after which there will be no 
	#	change In points of every node. This situation is called 
	#	convergence.
	# Algorithm (steps for implementing the Random Walk method):
	#	1. Create a directed graph with N nodes.
	#	2. Now perform a random walk.
	#	3. Now get sorted nodes as per points during random walk.
	#	4. At last, compare it with the inbuilt PageRank method.
	# -----------------------------------------------------------------
	# 1. Create a directed graph with N nodes
	g = nx.DiGraph()
	N = 15
	g.add_nodes_from(range(N))

	# 2. Add directed edges in graph
	g = add_edges(g, 0.4)

	# 3. perform a random walk
	points = random_Walk(g)

	# 4. Get nodes rank according to their random walk points
	sorted_by_points = nodes_sorted(g, points)
	print("PageRank using Random Walk Method")
	print(sorted_by_points)
	
	# p_dict is dictionary of tuples
	p_dict = nx.pagerank(g)
	p_sort = sorted(p_dict.items(), key=lambda x: x[1], reverse=True)

	print("PageRank using inbuilt pagerank method")
	# for i in p_sort:
	# 	print(i[0], end=", ")
	print(", ".join([str(i[0]) for i in p_sort]))

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()