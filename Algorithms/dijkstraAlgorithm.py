import networkx as nx

def dijkstra_shortest_path(G, start, end):
    """
    This function finds the shortest path between two nodes using Dijkstra's algorithm.
    """
    try:
        # Using NetworkX's built-in Dijkstra algorithm
        shortest_path = nx.dijkstra_path(G, source=start, target=end, weight='weight')
        return shortest_path, nx.dijkstra_path_length(G, source=start, target=end, weight='weight')
    except nx.NetworkXNoPath:
        print(nx.NetworkXNoPath, "No path found from", start, "to", end)
        return None, None
