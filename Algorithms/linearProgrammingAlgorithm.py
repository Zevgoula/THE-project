import pulp
import networkx as nx

def lp_shortest_path(graph, start, end):
    # Create the linear programming problem
    prob = pulp.LpProblem("ShortestPath", pulp.LpMinimize)

    # Create a dictionary to hold the decision variables for each edge
    edges = list(graph.edges(data=True))
    edge_vars = pulp.LpVariable.dicts("Edge", [(u, v) for u, v, _ in edges], 0, 1, pulp.LpBinary)

    # Set the objective function to minimize the total weight of the selected edges
    prob += pulp.lpSum([edge_vars[(u, v)] * data['weight'] for u, v, data in edges])

    # Add constraints for flow conservation at each node
    nodes = graph.nodes()

    for node in nodes:
        if node == start:
            # Start node: outgoing flow should be 1
            prob += pulp.lpSum([edge_vars[(u, v)] for u, v, _ in edges if u == node]) == 1
        elif node == end:
            # End node: incoming flow should be 1
            prob += pulp.lpSum([edge_vars[(u, v)] for u, v, _ in edges if v == node]) == 1
        else:
            # Other nodes: incoming flow should equal outgoing flow
            prob += (pulp.lpSum([edge_vars[(u, v)] for u, v, _ in edges if v == node]) ==
                     pulp.lpSum([edge_vars[(u, v)] for u, v, _ in edges if u == node]))

    # Solve the problem
    prob.solve()

    # Extract the optimal path
    if pulp.LpStatus[prob.status] == "Optimal":
        path = []
        for u, v, _ in edges:
            if pulp.value(edge_vars[(u, v)]) == 1:
                path.append((u, v))

        # Get the path nodes in sequence
        path_nodes = [start]
        current = start
        while current != end:
            for u, v in path:
                if u == current:
                    path_nodes.append(v)
                    current = v
                    break

        # Return the path and its total weight
        total_weight = sum(graph[u][v]['weight'] for u, v in zip(path_nodes[:-1], path_nodes[1:]))
        return path_nodes, total_weight
    else:
        return None, None
