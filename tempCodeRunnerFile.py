import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import Algorithms.linearProgrammingAlgorithm as lp
import Algorithms.dijkstraAlgorithm as dj

# Load data (assuming data has been loaded correctly)
stops = pd.read_csv('stops.txt')
stop_times = pd.read_csv('stop_times.txt')

G = nx.Graph()

# Add stops as nodes
for _, stop in stops.iterrows():
    G.add_node(stop['stop_id'], pos=(stop['stop_lon'], stop['stop_lat']))

# Add edges based on stop_times
for trip_id in stop_times['trip_id'].unique():
    trip_stop_times = stop_times[stop_times['trip_id'] == trip_id].sort_values('stop_sequence')
    for i in range(len(trip_stop_times) - 1):
        stop_start_id = trip_stop_times.iloc[i]['stop_id']
        stop_end_id = trip_stop_times.iloc[i + 1]['stop_id']
        G.add_edge(stop_start_id, stop_end_id, weight=1)  # or use distance if available

# Initialize plot
pos = {stop['stop_id']: (stop['stop_lon'], stop['stop_lat']) for _, stop in stops.iterrows()}
fig, ax = plt.subplots(figsize=(12, 12))

# Draw the nodes and edges manually to make nodes clickable
nx.draw_networkx_edges(G, pos, ax=ax)

# Plot nodes using scatter to make them clickable
node_positions = [pos[node] for node in G.nodes()]
nodes = ax.scatter([x for x, y in node_positions], [y for x, y in node_positions], s=100, c="blue", picker=True)

selected_nodes = []
path_edges = None  # This will store the current path edges plot object

def clear_previous_path():
    """Clear the previously drawn path from the plot."""
    global path_edges
    if path_edges:
        # Remove the current path edges plot object from the axes
        path_edges.remove()
        path_edges = None
        plt.draw()

def on_click(event, algorithm='lp'):
    # Check if a node was clicked based on proximity
    if event.inaxes is not None:
        x, y = event.xdata, event.ydata
        closest_node = min(pos, key=lambda node: (pos[node][0] - x)**2 + (pos[node][1] - y)**2)

        print(f"Selected node: {closest_node}")
        selected_nodes.append(closest_node)
        start = selected_nodes[0]
        end = selected_nodes[-1]
        if len(selected_nodes) == 2:
            # Clear previous path before drawing the new one
            clear_previous_path()
            if algorithm == 'lp':
                path, total_weight = lp.lp_shortest_path(G, start, end)
                if path:
                    print("Shortest path:", path)
                    print("Total weight of the shortest path:", total_weight)
                    # Convert path to edges for drawing
                    path_edges_list = list(zip(path, path[1:]))
                    path_edges = nx.draw_networkx_edges(G, pos, edgelist=path_edges_list, edge_color='red', width=2, ax=ax)
                    plt.title(f"Shortest path from {start} to {end}")
                else:
                    print(f"No path between {start} and {end}")
            elif algorithm == 'dijkstra':
                path = dj.dijkstra_shortest_path(G, start, end)
                if path:
                    print("Shortest path:", path)
                    path_edges_list = list(zip(path, path[1:]))
                    path_edges = nx.draw_networkx_edges(G, pos, edgelist=path_edges_list, edge_color='red', width=2, ax=ax)
                    plt.title(f"Shortest path from {start} to {end}")
                else: 
                    print(f"No path between {start} and {end}")
            elif algorithm == 'basic':
                try:
                    shortest_path = nx.shortest_path(G, source=start, target=end, weight='weight')
                    path_edges_list = list(zip(shortest_path, shortest_path[1:]))
                    path_edges = nx.draw_networkx_edges(G, pos, edgelist=path_edges_list, edge_color='red', width=2, ax=ax)
                    plt.title(f"Shortest path from {start} to {end}")
                except nx.NetworkXNoPath:
                    print(f"No path between {start} and {end}")
            else:
                print(f"Unknown algorithm: {algorithm}")

            plt.draw()

        # If more than 2 nodes are selected, clear previous selections and path
        if len(selected_nodes) > 2:
            selected_nodes.clear()
            clear_previous_path()
            selected_nodes.append(closest_node)  # Add the new node as the first in the new selection

# Connect the event handler
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
