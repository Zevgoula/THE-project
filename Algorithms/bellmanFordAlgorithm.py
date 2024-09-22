import networkx as nx   

def bellman_ford_shortest_path(graph, start, end):
    """
    This function finds the shortest path between two nodes using Bellman-Ford algorithm.
    """
    try:
        # Using NetworkX's built-in Bellman-Ford algorithm
        shortest_path = nx.bellman_ford_path(graph, source=start, target=end, weight='weight')
        return shortest_path
    except nx.NetworkXNoPath:
        print(nx.NetworkXNoPath, "No path found from", start, "to", end)
        return None