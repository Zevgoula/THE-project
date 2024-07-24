import pulp

# Define the problem
prob = pulp.LpProblem("EV_Charging_Stations", pulp.LpMinimize)

# Define the nodes
nodes = ["H", "I1", "C1", "I2", "I3", "C2", "O"]

# Define the edges and distances
edges = {
    ("H", "I1"): 30,
    ("I1", "C1"): 40,
    ("I1", "I2"): 50,
    ("I2", "O"): 30,
    ("I2", "I3"): 30,
    ("I3", "C2"): 20,
    ("C1", "C2"): 60,
    ("C2", "O"): 30
}

# Define the battery capacity and consumption
battery_capacity = 100
battery_consumption = edges

# Define decision variables
x = pulp.LpVariable.dicts("x", edges, cat="Binary")
battery = pulp.LpVariable.dicts("battery", nodes, lowBound=0, cat="Continuous")

# Objective function
prob += pulp.lpSum([x[edge] * distance for edge, distance in edges.items()])

# Flow conservation constraints
for node in nodes:
    if node == "H":
        prob += pulp.lpSum([x[(node, j)] for (node, j) in edges if node == "H"]) - pulp.lpSum([x[(i, node)] for (i, node) in edges if node == "H"]) == 1
    elif node == "O":
        prob += pulp.lpSum([x[(node, j)] for (node, j) in edges if node == "O"]) - pulp.lpSum([x[(i, node)] for (i, node) in edges if node == "O"]) == -1
    else:
        prob += pulp.lpSum([x[(node, j)] for (node, j) in edges if node == node]) - pulp.lpSum([x[(i, node)] for (i, node) in edges if node == node]) == 0

# Battery constraints
prob += battery["H"] == battery_capacity
for (i, j) in edges:
    prob += battery[j] >= battery[i] - battery_consumption[(i, j)] * x[(i, j)]
for node in nodes:
    prob += battery[node] >= 0

# Charging stations (reset battery to max capacity)
charging_stations = ["C1", "C2"]
for cs in charging_stations:
    prob += battery[cs] == battery_capacity

# Solve the problem
prob.solve()

# Print the results
print("Status:", pulp.LpStatus[prob.status])
print("Optimal route and battery levels:")
for (i, j) in edges:
    if pulp.value(x[(i, j)]) == 1:
        print(f"Edge ({i}, {j}) with distance {edges[(i, j)]} km")
for node in nodes:
    print(f"Battery at {node}: {pulp.value(battery[node])} km")
