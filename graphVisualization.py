import matplotlib.pyplot as plt

# Get the positions for each stop (latitude and longitude)
pos = {stop['stop_id']: (stop['stop_lon'], stop['stop_lat']) for _, stop in stops.iterrows()}

# Draw the graph
plt.figure(figsize=(10, 10))
nx.draw(G, pos, node_size=10, with_labels=True, node_color="blue", font_size=8, font_color="black", edge_color="gray")

# Show the plot
plt.show()
