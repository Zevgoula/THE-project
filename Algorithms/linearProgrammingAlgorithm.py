import pulp
import networkx as nx

        
def lp_shortest_path(graph, start, end):
    # Create the linear programming problem
    prob = pulp.LpProblem("ShortestPath", pulp.LpMinimize)
    # Create a dictionary to hold the decision variables for each edge
    edges = list(graph.edges(data=True))
    edge_vars = pulp.LpVariable.dicts("Edge", [(u, v) for u, v, _ in edges] + [(v, u) for u, v, _ in edges], 0, 1, pulp.LpBinary)

   # Modify the objective function to handle both directions
    prob += pulp.lpSum([edge_vars[(u, v)] * data.get('weight', 1) for u, v, data in edges] +
                    [edge_vars[(v, u)] * data.get('weight', 1) for u, v, data in edges]), "Total Weight of Path"


    # Add constraints for flow conservation at each node
    nodes = graph.nodes()

    for node in nodes:


        if node == start:
            prob += pulp.lpSum([edge_vars[(u, v)] for u, v in edge_vars if u == node]) == 1
        elif node == end:
            prob += pulp.lpSum([edge_vars[(u, v)] for u, v in edge_vars if v == node]) == 1
        else:
            prob += pulp.lpSum([edge_vars[(u, v)] for u, v in edge_vars if (v == node and u!=end)]) -\
                    pulp.lpSum([edge_vars[(u, v)] for u, v in edge_vars if (u == node and v!=start)]) == 0


    # Solve the problem
    # print("Problem: ", prob.objective, prob.constraints )
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Debug: Check the status of the solution
    # print("Solver Status: ", pulp.LpStatus[prob.status])

    # Extract the optimal path if it exists
    if pulp.LpStatus[prob.status] == "Optimal":
        path = []
        for u, v in edge_vars:
            if pulp.value(edge_vars[(u, v)]) == 1:
                path.append((u, v))

        # Get the path nodes in sequence
        path_nodes = [start]
        current = start
        while current != end:
            found_next = False
            for u, v in path:
                if u == current:
                    path_nodes.append(v)
                    current = v
                    found_next = True
                    break
            if not found_next:
                print(f"No next node found from current node: {current}. Path: {path_nodes}")
                break  # Break if no next node was found

        # Check if we reached the end
        if current == end:
            total_weight = sum(graph[u][v]['weight'] for u, v in zip(path_nodes[:-1], path_nodes[1:]))
            
            return path_nodes, total_weight
        else:
            print("Path does not reach the end node.")
            return None, None
    else:
        print("No optimal solution found.")
        return None, None