import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys
from datetime import timedelta

# Import algorithms
import Algorithms.linearProgrammingAlgorithm as lp
import Algorithms.dijkstraAlgorithm as dj
import Algorithms.bellmanFordAlgorithm as bl
import Algorithms.astarAlgorithm as astar

def main():
    # Welcome message and location input
    print("--------------------Welcome to Shortest Path Finder--------------------")
    print("\nWould you like to find the shortest paths (1) or get statistics? (2)")
    print("\nChoose the location you want to find the shortest path in:")
    print("1. ATHENS\n2. THESSALONIKI\n3. ITALY, SARDINIA\n4. SPAIN, MADRID\n5. CANADA, VANCOUVER")

    # Get location input
    location_map = {
        "1": "athens",
        "2": "thessaloniki",
        "3": "italy",
        "4": "spain",
        "5": "canada"
    }
    
    algorithm_map = {
        "1": "linear-programming",
        "2": "dijkstra",
        "3": "bellman-ford",
        "4": "astar",
        "5": "basic"
    }
    location = input("Enter the location number: ")

    if location not in location_map:
        print("Invalid location")
        sys.exit(1)
    location = location_map[location]
    
    print(f"Choose an algorithm to find the shortest path in {location}:")
    print("1. Linear Programming\n2. Dijkstra\n3. Bellman-Ford\n4. A*")
    algorithm = input("Enter the algorithm number: ")
    if algorithm not in algorithm_map:
        print("Invalid algorithm")
        sys.exit(1)
    algorithm = algorithm_map[algorithm]
    # Load data
    print("\nLoading data ...")
    try:
        stops = pd.read_csv(f'locations/{location}/stops.txt')  # Replace with actual file path
        stop_times = pd.read_csv(f'locations/{location}/stop_times.txt')  # Replace with actual file path
        print(f"Data loaded successfully for {location}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Handle extended hours (times beyond 24:00:00)
    def handle_extended_hours(time_str):
        hours, minutes, seconds = map(int, time_str.split(':'))
        if hours >= 24:
            return timedelta(days=1, hours=hours-24, minutes=minutes, seconds=seconds)
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)

    # Create a graph using NetworkX
    G = nx.Graph()

    # Add edges based on stop_times data
    for trip_id in stop_times['trip_id'].unique():
        trip_stop_times = stop_times[stop_times['trip_id'] == trip_id].sort_values('stop_sequence')

        for i in range(len(trip_stop_times) - 1):
            start_id = trip_stop_times.iloc[i]['stop_id']
            end_id = trip_stop_times.iloc[i + 1]['stop_id']

            # Calculate the time difference between stops
            departure_time = handle_extended_hours(trip_stop_times.iloc[i]['departure_time'])
            arrival_time = handle_extended_hours(trip_stop_times.iloc[i + 1]['arrival_time'])
            time_diff = (arrival_time - departure_time).total_seconds() / 60.0

            if time_diff <= 0:
                time_diff = 0.5  # Handle edge cases with no time difference

            G.add_edge(start_id, end_id, weight=time_diff)

    # Print graph information
    print("\nNumber of nodes in the graph:", len(G.nodes()))
    print("Number of edges in the graph:", len(G.edges()))
    print("Graph is", "connected" if nx.is_connected(G) else "not connected")
    print("Graph is", "directed" if nx.is_directed(G) else "undirected")

    # Plot setup
    pos = {stop['stop_id']: (stop['stop_lon'], stop['stop_lat']) for _, stop in stops.iterrows()}
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.title(f"{location}\nSelect two nodes to find the shortest path".upper())
    nx.draw_networkx_edges(G, pos, ax=ax)

    # Initial node plot
    node_positions = [pos[node] for node in G.nodes()]
    nodes = ax.scatter([x for x, y in node_positions], [y for x, y in node_positions], s=50, c='blue', picker=True)

    selected_nodes = []
    path_edges = None

    # Functions to handle path drawing and node selection
    def clear_previous_path():
        nonlocal path_edges
        if path_edges:
            path_edges.remove()
            path_edges = None
            plt.draw()
            

    def update_node_colors():
        node_colors = ['green' if node in selected_nodes else 'blue' for node in G.nodes()]
        nodes.set_color(node_colors)
        plt.draw()

    def on_click(event):
        nonlocal path_edges
        if event.inaxes is None:
            return

        x, y = event.xdata, event.ydata
        closest_node = min(pos, key=lambda node: (pos[node][0] - x)**2 + (pos[node][1] - y)**2)

        print(f"Selected node: {closest_node}")
        selected_nodes.append(closest_node)
        update_node_colors()

        if len(selected_nodes) == 2:
            clear_previous_path()

            start, end = selected_nodes[0], selected_nodes[-1]
            path = None
            if algorithm == 'linear-programming':
                path, total_weight = lp.lp_shortest_path(G, start, end)
            elif algorithm == 'dijkstra':
                path, total_weight = dj.dijkstra_shortest_path(G, start, end)
            elif algorithm == 'basic':
                try:
                    path = nx.shortest_path(G, source=start, target=end, weight='weight')
                except nx.NetworkXNoPath:
                    print(f"No path between {start} and {end}")
            elif algorithm == 'bellman-ford':
                path, total_weight = bl.bellman_ford_shortest_path(G, start, end)
            elif algorithm == 'astar':
                path, total_weight = astar.astar_shortest_path(G, start, end)
            if path:
                print("Shortest path:", path)
                path_edges_list = list(zip(path, path[1:]))
                path_edges = nx.draw_networkx_edges(G, pos, edgelist=path_edges_list, edge_color='red', width=2, ax=ax)
                plt.title(f"{location}\nShortest path from {start} to {end} with total weight: {total_weight}".upper())
            else:
                print(f"No path between {start} and {end}")
                
            fig.text(0.5, 0.05, f"Using the {algorithm} algorithm".upper(), ha='center', va='center')
            plt.draw()

        if len(selected_nodes) > 2:
            selected_nodes.clear()
            clear_previous_path()
            selected_nodes.append(closest_node)
            update_node_colors()

    # Connect event handler
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

if __name__ == '__main__':
    main()
