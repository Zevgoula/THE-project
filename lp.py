import time
import pulp

def solve_shortest_path(graph, start, end):
    # Create the linear programming problem
    prob = pulp.LpProblem("ShortestPath", pulp.LpMinimize)

    # Create a dictionary to hold the decision variables
    edges = [(u, v) for u in graph for v in graph[u]]
    edge_vars = pulp.LpVariable.dicts("Edge", edges, 0, 1, pulp.LpBinary)

    # Objective function: minimize the total weight of the path
    prob += pulp.lpSum([graph[u][v] * edge_vars[(u, v)] for (u, v) in edges])

    # Constraints
    nodes = set(graph.keys()).union(set(v for neighbors in graph.values() for v in neighbors))
    for node in nodes:
        if node == start:
            prob += pulp.lpSum([edge_vars[(u, v)] for (u, v) in edges if u == node]) - \
                    pulp.lpSum([edge_vars[(u, v)] for (u, v) in edges if v == node]) == 1
        elif node == end:
            prob += pulp.lpSum([edge_vars[(u, v)] for (u, v) in edges if u == node]) - \
                    pulp.lpSum([edge_vars[(u, v)] for (u, v) in edges if v == node]) == -1
        else:
            prob += pulp.lpSum([edge_vars[(u, v)] for (u, v) in edges if u == node]) - \
                    pulp.lpSum([edge_vars[(u, v)] for (u, v) in edges if v == node]) == 0

    # Solve the problem
    prob.solve()

    # Extract the path and total weight
    if pulp.LpStatus[prob.status] == "Optimal":
        path = []
        current_node = start
        total_weight = 0

        while current_node != end:
            for v in graph[current_node]:
                if pulp.value(edge_vars[(current_node, v)]) == 1:
                    path.append((current_node, v))
                    total_weight += graph[current_node][v]
                    current_node = v
                    break

        return path, total_weight
    else:
        return None, None

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

path, total_weight = solve_shortest_path(graph, start_node, end_node)

end_time = time.time()

if path is not None:
    print("Shortest path:", " -> ".join(f"{u}->{v}" for (u, v) in path))
    print("Total weight of the shortest path:", total_weight)
else:
    print("No path found from", start_node, "to", end_node)

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")
