import random
import sys
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from datetime import timedelta
import matplotlib.pyplot as plt

# Import your algorithms
import Algorithms.linearProgrammingAlgorithm as lp
import Algorithms.dijkstraAlgorithm as dj
import Algorithms.bellmanFordAlgorithm as bl
import Algorithms.astarAlgorithm as astar

# Define location categories based on graph size
small_locations = ["alabama", "california", "arizona"]  # less than 100 edges
medium_locations = ["athens", "spain", "thessaloniki", "italy"]  # 100-500 edges
large_locations = ["canada"]  # 500-1000 edges
huge_locations = ["new_york"]  # 1000+ edges
locations = small_locations + medium_locations + large_locations # + huge_locations

def handle_extended_hours(time_str) -> timedelta:
    """Handle time formatting for hours that exceed 24 hours."""
    hours, minutes, seconds = map(int, time_str.split(':'))
    if hours >= 24:
        return timedelta(days=1, hours=hours - 24, minutes=minutes, seconds=seconds)
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)

def create_graph(location) -> nx.Graph:
    """Load stop and stop_times data for a given location."""
    G = nx.Graph()
    try:
        stops = pd.read_csv(f'locations/{location}/stops.txt')  # Replace with actual file path
        stop_times = pd.read_csv(f'locations/{location}/stop_times.txt')  # Replace with actual file path
        print(f"Data loaded successfully for {location}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

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

    return G, G.number_of_edges(), G.number_of_nodes()

def load_grapghs(locations):
    """Load graphs for all locations."""
    graphs = {}
    for location in locations:
        G, num_edges, num_nodes = create_graph(location)
        graphs[location] = G, num_edges, num_nodes
    return graphs

def create_unique_couples(G: nx.graph, num_couples=100):
    """Create unique node pairs for testing."""
    nodes = list(G.nodes())
    unique_couples = random.sample([(node1, node2) for node1 in nodes for node2 in nodes if (node1 != node2 and nx.has_path(G, node1, node2))], num_couples)
    return unique_couples

def calculate_time(sorted_locations):
    results = []
    for graph_name, (graph, num_edges, num_nodes) in sorted_locations:
        for (start, end) in create_unique_couples(graph):
            try:
                paths = {}
                weights = {}
                times = {}

                # Dijkstra Algorithm
                time0 = timeit.default_timer()
                paths['Dijkstra'], weights['Dijkstra'] = dj.dijkstra_shortest_path(graph, start, end)
                time1 = timeit.default_timer()
                times['Dijkstra'] = time1 - time0

                # Linear Programming Algorithm
                time0 = timeit.default_timer()
                paths['LP'], weights['LP'] = lp.lp_shortest_path(graph, start, end)
                time1 = timeit.default_timer()
                times['LP'] = time1 - time0

                # Bellman-Ford Algorithm
                time0 = timeit.default_timer()
                paths['Bellman-Ford'], weights['Bellman-Ford'] = bl.bellman_ford_shortest_path(graph, start, end)
                time1 = timeit.default_timer()
                times['Bellman-Ford'] = time1 - time0

                # A* Algorithm
                time0 = timeit.default_timer()
                paths['A*'], weights['A*'] = astar.astar_shortest_path(graph, start, end)
                time1 = timeit.default_timer()
                times['A*'] = time1 - time0

                # Check weights
                if all(weights['LP'] == weights[algo] for algo in weights) and all(path is not None for path in paths.values()):
                    results.append({
                        'Graph': graph_name,
                        'Start': start,
                        'End': end,
                        'Times': times,
                        'Weights': weights
                    })

            except Exception as e:
                print(f"Error in {graph_name} for {start} to {end}: {e}")

    print("results: ", results) 
    return results

def get_mean_time(results):
    mean_times = {}
    algo_counts = {}

    for result in results:
        graph_name = result['Graph']
        times = result['Times']

        for algo, time in times.items():
            if (graph_name, algo) not in mean_times:
                mean_times[(graph_name, algo)] = 0
                algo_counts[(graph_name, algo)] = 0
            
            mean_times[(graph_name, algo)] += time
            algo_counts[(graph_name, algo)] += 1

    # Calculate the mean for each (graph, algo)
    for key in mean_times:
        mean_times[key] /= algo_counts[key]
    print("mean_times: ", mean_times)
    return mean_times

def get_graph_stuff_from_name(graph_name):
    if graph_name in graphs:
        return graphs[graph_name][1], graphs[graph_name][2] #edges and nodes
    else:
        return None, None
        
def plot_time_vs_edges(mean_times):
    edges = []
    times = {algo: [] for algo in set(algo for _, algo in mean_times.keys())}

    for (graph_name, algo), time in mean_times.items():
        # Get the number of edges using the tuple index
        num_edges = get_graph_stuff_from_name(graph_name)[1]

        if num_edges not in edges:
            edges.append(num_edges)
        times[algo].append(time*1000)

    
    sorted_times = {algo: [time for edge, time in sorted(zip(edges, algo_times), key=lambda x: x[0])]
                    for algo, algo_times in times.items()}
    sorted_edges = sorted(set(edges))  # Sort unique edges


    # Plotting
    plt.figure(figsize=(12, 6))

    for algo, algo_times in sorted_times.items():
        plt.plot(sorted_edges, algo_times, marker='o', label=algo)

    plt.xlabel('Number of Edges')
    plt.ylabel('Execution Time (milliseconds)')
    plt.title('Execution Time of Algorithms vs. Number of Edges')
    plt.xticks(sorted_edges)  # Set x-ticks to unique edge counts
    plt.legend()
    plt.yscale('log')
    plt.grid()
    plt.show()
    
def main():
    #loading all graphs
    global graphs
    graphs = load_grapghs(locations)
    
    # from smallest to largest number of edges
    sorted_locations = sorted(graphs.items(), key=lambda x: x[1][1])
    results = calculate_time(sorted_locations)
    plot_time_vs_edges(get_mean_time(results))
    
    

if __name__ == "__main__":
    main()