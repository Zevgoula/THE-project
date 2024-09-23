import random
import sys
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from datetime import timedelta

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

graphs = {}


# =================== Algorithm Comparison ===================
def compare_algorithms(graph, unique_pairs):
    """Compare the performance of various pathfinding algorithms on the graph."""
    results = []

    for (start, end) in unique_pairs:
        times = {}
        paths = {}
        weights = {}

        # Dijkstra Algorithm
        time0 = timeit.default_timer()
        paths['Dijkstra'], weights['Dijkstra'] = dj.dijkstra_shortest_path(graph, start, end)
        if paths['Dijkstra'] is None:
            continue
        time1 = timeit.default_timer()

        # Linear Programming Algorithm
        paths['LP'], weights['LP'] = lp.lp_shortest_path(graph, start, end)
        time2 = timeit.default_timer()

        # Bellman-Ford Algorithm
        paths['Bellman-Ford'], weights['Bellman-Ford'] = bl.bellman_ford_shortest_path(graph, start, end)
        time3 = timeit.default_timer()

        # A* Algorithm
        paths['A*'], weights['A*'] = astar.astar_shortest_path(graph, start, end)
        time4 = timeit.default_timer()

        # Compare results and record times
        if all(weight == weights['LP'] for weight in weights.values()):
            times['Dijkstra'] = time1 - time0
            times['LP'] = time2 - time1
            times['Bellman-Ford'] = time3 - time2
            times['A*'] = time4 - time3
            results.append(times)
        else:
            print(f"Different paths found for {start} to {end}. Skipping comparison.")

    return results


# =================== Aggregate Results Across Locations ===================
def aggregate_results(all_results):
    """Aggregate the execution times across all locations."""
    aggregated_data = {
        'Dijkstra': [],
        'LP': [],
        'Bellman-Ford': [],
        'A*': []
    }

    # Combine all results into one list
    for result_set in all_results:
        for times in result_set:
            for algorithm in ['Dijkstra', 'LP', 'Bellman-Ford', 'A*']:
                aggregated_data[algorithm].append(times[algorithm])

    return aggregated_data


def analyze_aggregated_results(aggregated_data):
    """Analyze the aggregated results and display one summary graph."""
    df = pd.DataFrame(aggregated_data)

    # Calculate average, min, and max for each algorithm
    summary = df.agg(['mean', 'min', 'max']).T
    summary = summary * 1000  # Convert to milliseconds
    print(summary)

    # Plot a single combined graph
    plt.figure(figsize=(12, 8))
    summary['mean'].plot(kind='bar', color=['blue', 'orange', 'green', 'red'])
    plt.title('Average Execution Time of Pathfinding Algorithms Across All Locations')
    plt.ylabel('Time (ms)')
    plt.xlabel('Algorithm')
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.tight_layout()
    
    plt.show()


# =================== Helper Functions ===================
def handle_extended_hours(time_str):
    """Handle time formatting for hours that exceed 24 hours."""
    hours, minutes, seconds = map(int, time_str.split(':'))
    if hours >= 24:
        return timedelta(days=1, hours=hours - 24, minutes=minutes, seconds=seconds)
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def load_graph_data(location):
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

    return G


# =================== Main Function ===================
def main(locations, n):
    """Main function to run the analysis on the selected locations."""
    all_results = []

    for location in locations:
        G = load_graph_data(location)

        # Store graph
        graphs[location] = G

        # Generate random unique pairs
        all_nodes = list(G.nodes())
        unique_pairs = random.sample([(node1, node2) for node1 in all_nodes for node2 in all_nodes if node1 != node2], n)

        # Compare algorithms and collect results
        results = compare_algorithms(G, unique_pairs)
        all_results.append(results)

    # Aggregate results from all locations
    aggregated_data = aggregate_results(all_results)

    # Analyze and visualize aggregated results
    analyze_aggregated_results(aggregated_data)


# =================== User Interaction ===================
if __name__ == '__main__':
    print("Running stats.py")
    print("Choose the graph size to analyze:")
    print("1. Small (less than 100 edges)\n2. Medium (100-500 edges)\n3. Large (500-1000 edges)\n4. Very Large (1000+ edges)\n5. All")
    
    chosen_size = input("Enter the number of the graph size: ")
    if chosen_size == "1":
        main(small_locations, 40)
    elif chosen_size == "2":
        main(medium_locations, 70)
    elif chosen_size == "3":
        main(large_locations, 100)
    elif chosen_size == "4":
        main(huge_locations, 200)
    elif chosen_size == "5":
        print("How many paths do you want to analyze for each location? (the smaller the number, the faster the analysis)")
        n = int(input("Enter the number of paths: "))
        if n<=10:
            print("Please enter a number greater than 10")
            sys.exit(1)
        main(small_locations + medium_locations + large_locations + huge_locations, n)
    else:
        print("Invalid graph size")
        sys.exit(1)
