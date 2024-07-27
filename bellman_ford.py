import time

def bellman_ford(graph, start):
    # Initialize distances dictionary with infinity for all nodes except start node
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    # Initialize previous dictionary to keep track of previous node for each visited node
    previous = {node: None for node in graph}

    # Relax edges up to |V| - 1 times (where |V| is the number of vertices)
    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbor, weight in graph[node].items():
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight
                    previous[neighbor] = node

    # Check for negative-weight cycles
    for node in graph:
        for neighbor, weight in graph[node].items():
            if distances[node] + weight < distances[neighbor]:
                raise ValueError("Graph contains a negative-weight cycle")

    return distances, previous

# Example usage
graph = {
    'A': {'B': 2, 'C': 1},
    'B': {'D': 4, 'E': 6},
    'C': {'E': 6, 'F': 1},
    'D': {'E': 2},
    'E': {},
    'F': {'D': 2}
}

start_node = 'A'
end_node = 'E'

# Measure the execution time
start_time = time.time()

try:
    distances, previous = bellman_ford(graph, start_node)
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
        print(f"No path from node {start_node} to node {end_node}")
    else:
        print(f"Shortest path from node {start_node} to node {end_node}: {' -> '.join(path)}")
        # Show the total weight of the shortest path
        total_weight = distances[end_node]
        print(f"Total weight of the shortest path: {total_weight}")

except ValueError as e:
    end_time = time.time()
    print(e)

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")
