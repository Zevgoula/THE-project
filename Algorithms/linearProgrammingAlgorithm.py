import pulp
import networkx as nx

def lp_shortest_path(graph, start, end):
    # Create the linear programming problem
    prob = pulp.LpProblem("ShortestPath", pulp.LpMinimize)

    # Create a dictionary to hold the decision variables
    edges = list(graph.edges(data=True))  # Include edge attributes (weights)
    edge_vars = pulp.LpVariable.dicts("Edge", [(u, v) for u, v, _ in edges], 0, 1, pulp.LpBinary)

    # Objective function: minimize the total weight of the path
    prob += pulp.lpSum([attr['weight'] * edge_vars[(u, v)] for (u, v, attr) in edges])

    # Constraints
    nodes = list(graph.nodes())
    for node in nodes:
        if node == start:
            prob += pulp.lpSum([edge_vars[(u, v)] for (u, v, _) in edges if u == node]) - \
                    pulp.lpSum([edge_vars[(u, v)] for (u, v, _) in edges if v == node]) == 1
        elif node == end:
            prob += pulp.lpSum([edge_vars[(u, v)] for (u, v, _) in edges if u == node]) - \
                    pulp.lpSum([edge_vars[(u, v)] for (u, v, _) in edges if v == node]) == -1
        else:
            prob += pulp.lpSum([edge_vars[(u, v)] for (u, v, _) in edges if u == node]) - \
                    pulp.lpSum([edge_vars[(u, v)] for (u, v, _) in edges if v == node]) == 0

    # Solve the problem
    prob.solve()

    # Extract the path and total weight
    if pulp.LpStatus[prob.status] == "Optimal":
        path = []
        current_node = start
        total_weight = 0
        visited_nodes = set()  # Keep track of visited nodes to avoid cycles

        # Safety mechanism to avoid infinite loops
        safety_counter = 0
        max_iterations = len(graph.nodes()) + 1  # More than total nodes means something's wrong

        while current_node != end and safety_counter < max_iterations:
            found = False
            visited_nodes.add(current_node)

            for v in graph[current_node]:
                if (current_node, v) in edge_vars and pulp.value(edge_vars[(current_node, v)]) == 1:
                    path.append((current_node, v))
                    total_weight += graph[current_node][v]['weight']
                    current_node = v
                    found = True
                    break
                # If the graph is undirected, check in the opposite direction too
                elif (v, current_node) in edge_vars and pulp.value(edge_vars[(v, current_node)]) == 1:
                    path.append((v, current_node))
                    total_weight += graph[v][current_node]['weight']
                    current_node = v
                    found = True
                    break

            # Safety increment
            safety_counter += 1

            # If no edge is found, exit to prevent infinite loop
            if not found or current_node in visited_nodes:
                print(f"Error: No valid edge found from node {current_node} or cycle detected.")
                return None, None

        if safety_counter >= max_iterations:
            print("Error: Max iterations reached, possible infinite loop.")
            return None, None

        return path, total_weight
    else:
        return None, None
