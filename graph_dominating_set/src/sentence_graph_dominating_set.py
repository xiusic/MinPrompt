#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from graph import Graph
# import matplotlib.pyplot as plt
import time
import random
import networkx as nx
import sys
import math

from networkx import dominating_set

G = nx.Graph()
# graph = Graph(104160)
# adj_list = []
IndependentNodes = set()
v = 0
e = 0
with open("~/ptQA/mrqa-few-shot/searchqa/adj_list.txt", 'r') as readGraph:
    i = 0
    for line in readGraph.readlines():
        edge_data = list(map(lambda x: int(x), line.split()))
        if not edge_data:
            IndependentNodes.add(i)
        # if i == 0:
            # tmp = line.split()
            # v, e = int(tmp[0]), int(tmp[1])
            # i += 1

        else:
            for j in edge_data:
                # graph.add_edge(i, j)
                G.add_edge(i, j)
        # adj_list.append(str(i) + ' ' + line)
        i += 1

print(f"# IndependentNodes: {len(IndependentNodes)}")
print(f"# Non-IndependentNodes: {G.number_of_nodes()}")
print(f"G.number_of_edges: {G.number_of_edges()}")
# graph = nx.parse_adjlist(adj_list, nodetype=int)

DS = dominating_set(G)


with open("~/ptQA/mrqa-unsupervised/searchqa-DS-id.txt", "w") as fout:
    fout.write(",".join(map(str, DS)) + "\n")
        

with open("~/ptQA/mrqa-unsupervised/searchqa-IndependentNodes-id.txt", "w") as fout:
    fout.write(",".join(map(str, IndependentNodes)) + "\n")
    




