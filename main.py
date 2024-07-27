import pulp

# Define the problem
prob = pulp.LpProblem("Daily_Commute_Optimization", pulp.LpMinimize)

# Define the nodes
nodes = ["H", "I1", "I2", "I3", "I4", "W"]

# Define the edges and distances
edges = {
    ("H", "I1"): 10,
    ("H", "I2"): 15,
    ("I1", "I3"): 10,
    ("I1", "I4"): 20,
    ("I2", "I4"): 10,
    ("I3", "W"): 15,
    ("I4", "W"): 10
}

# Define decision variables
x = pulp.LpVariable.dicts("x", edges, cat="Binary")

# Objective function
prob += pulp.lpSum([x[edge] * distance for edge, distance in edges.items()])

# Flow conservation constraints
for node in nodes:
    if node == "H":
        prob += pulp.lpSum([x[(node, j)] for (node, j) in edges if node == "H"]) - pulp.lpSum([x[(i, node)] for (i, node) in edges if node == "H"]) == 1
    elif node == "W":
        prob += pulp.lpSum([x[(node, j)] for (node, j) in edges if node == "W"]) - pulp.lpSum([x[(i, node)] for (i, node) in edges if node == "W"]) == -1
    else:
        prob += pulp.lpSum([x[(node, j)] for (node, j) in edges if node == node]) - pulp.lpSum([x[(i, node)] for (i, node) in edges if node == node]) == 0

# Binary constraints
for edge in edges:
    prob += x[edge] <= 1

# Solve the problem
prob.solve()

# Print the results
print("Status:", pulp.LpStatus[prob.status])
print("Optimal route:")
for edge in edges:
    if pulp.value(x[edge]) == 1:
        print(f"Edge {edge} with distance {edges[edge]} km")
