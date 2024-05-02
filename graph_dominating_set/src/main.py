from graph import Graph
import matplotlib.pyplot as plt
import time
import random
import networkx as nx
import sys
import math

def getRandomVer(n):
    return [(random.randint(1, 9),random.randint(1, 9)) for i in range(n)]

# Create nodes not being coincident nor too close
def getRandomVer_v2(n):
    final_v = []
    while len(final_v) < n:
        new_x, new_y = (random.randint(1, 9), random.randint(1, 9))
        if final_v:
            for x,y in final_v:
                distance = math.sqrt((x - new_x)**2 + (y - new_y)**2)
                if distance > 0.1:
                    final_v.append((new_x, new_y))
        else:
            final_v.append((new_x, new_y))
    return final_v


def main(inst):
    begin = time.time()

    G = nx.Graph()
    graph = Graph(104160)
    # adj_list = []
    IndependentNodes = set()
    v = 0
    e = 0
    with open(inst, 'r') as readGraph:
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
                    graph.add_edge(i, j)
                    G.add_edge(i, j)
            # adj_list.append(str(i) + ' ' + line)
            i += 1
            
    print(f"# IndependentNodes: {len(IndependentNodes)}")
    print(f"# Non-IndependentNodes: {G.number_of_nodes()}")
    print(f"G.number_of_edges: {G.number_of_edges()}")
    # graph = nx.parse_adjlist(adj_list, nodetype=int)

    # nodes = getRandomVer(num_vertices)
    
    # graph = Graph(num_vertices)
    # G = nx.Graph()

    # for i in range(len(nodes)):
    #     G.add_node(i,pos=(nodes[i][0],nodes[i][1]))

    # pos=nx.get_node_attributes(G,'pos')

    # chosen_edges = {}
    # # Generate random edges
    # for i in range(0,num_vertices):
    #         n_iters = random.choice([x for x in range(1,int(num_vertices/2))])
    #         used_vertices = []
    #         for iteration in range(0, n_iters):
    #             j = []
    #             if i in chosen_edges:
    #                 j = list( set([ x for x in range(num_vertices) if x != i]) - set(chosen_edges[i]) - set(used_vertices))
    #             else:
    #                 j = list( set([ x for x in range(num_vertices) if x != i]) - set(used_vertices))
                
    #             j_choice = None
    #             if j:
    #                 j_choice = random.choice(j)
                
    #             if j_choice:
    #                 graph.add_edge(i,j_choice)  # Append to Matrix
    #                 G.add_edge(i,j_choice)
    #                 if j_choice not in chosen_edges.keys():
    #                     chosen_edges[j_choice] = set([i])
    #                 else:
    #                     chosen_edges[j_choice].add(i)

    #                 used_vertices.append(j_choice)
    

    creation_time = time.time() - begin
    print("Graph created with {} vertices and {} edges! Time elapsed: {} (ms)".format(num_vertices, graph.getNumEdges(), round(creation_time*1000, 4)))
            
    #graph.printMatriz()
    graph.writeMatrix()
    graph.getEdgesAdjacency()
    nx.draw(G, pos, with_labels=True)
    
    # Exhaustive Search
    begin = time.time()
    result = graph.findExhaustiveSolution()
    print("Minimum edge dominating set ({} algorithm): {}".format("exhaustive", result))
    print("Time elapsed: {} (ms) | Num basic ops {}".format( round((time.time() - begin)*1000, 4), graph.getBasicOps()))
    graph.cleanBasicOps()   

    # Greedy Search
    begin = time.time()
    result = graph.findGreedySolution()
    print("Minimum edge dominating set ({} algorithm): {}".format("greedy", result))
    print("Time elapsed: {} (ms) | Num basic ops {}".format( round((time.time() - begin)*1000, 4), graph.getBasicOps()))
      
    
    plt.show()

if __name__ == "__main__":
    num_vertices = None
    inst = sys.argv[1]
    # try:
    #     num_vertices = int(sys.argv[1])
    # except Exception as err:
    #     print("Usage: python3 main.py <generate random graph with N vertices (int)>")

    # if not isinstance(num_vertices, int):
    #     print("Vertices not int!")
    #     print("Usage: python3 main.py <generate random graph with N vertices (int)>")
    #     sys.exit(2)
    
    # else:
    main(inst)
