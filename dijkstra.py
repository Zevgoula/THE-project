import time

def dijkstra(graph, start):
    # Initialize distances dictionary with infinity for all nodes except start node
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    # Initialize visited set to keep track of visited nodes
    visited = set()

    # Initialize previous dictionary to keep track of previous node for each visited node
    previous = {node: None for node in graph}

    while len(visited) < len(graph):
        # Find the node with the minimum distance from the start node
        min_distance = float('inf')
        min_node = None
        for node in graph:
            if node not in visited and distances[node] < min_distance:
                min_distance = distances[node]
                min_node = node

        # If min_node is None, all remaining nodes are inaccessible from the start node
        if min_node is None:
            break

        # Mark the current node as visited
        visited.add(min_node)

        # Update distances of neighboring nodes
        for neighbor, weight in graph[min_node].items():
            distance = distances[min_node] + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = min_node

    return distances, previous

# Example usage
graph = {
    'A': {'B': 2, 'C': 1},
    'B': {'D': 4, 'E': 6},
    'C': {'E': 6, 'F': 1},
    'D': {'E': 4},
    'E': {},
    'F': {'D': 2}
}

start_node = 'A'
end_node = 'E'

# Measure the execution time
start_time = time.time()

distances, previous = dijkstra(graph, start_node)

end_time = time.time()

# Show the shortest path from the start node to the end node
path = []
current_node = end_node
while current_node is not None:
    path.append(current_node)
    current_node = previous[current_node]
path.reverse()

# Check if a path was found
if distances[end_node] == float('inf'):
    print(f"\nNo path from node {start_node} to node {end_node}")
else:
    print(f"\nShortest path from node {start_node} to node {end_node}: {' -> '.join(path)}")
    # Show the total weight of the shortest path
    total_weight = distances[end_node]
    print(f"Total weight of the shortest path: {total_weight}\n")

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")
