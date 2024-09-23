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
huge_locations = ["italy"]  # 1000+ edges
locations = small_locations + medium_locations + large_locations + huge_locations
# huge_locations = ["new_york"]  # 1000+ edges

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
def aggregate_results(all_results, flag=False):
    """Aggregate the execution times across all locations."""
    if flag: 
        aggregated_data = {
            'small': {'Dijkstra': [], 'LP': [], 'Bellman-Ford': [], 'A*': []},
            'medium': {'Dijkstra': [], 'LP': [], 'Bellman-Ford': [], 'A*': []},
            'large': {'Dijkstra': [], 'LP': [], 'Bellman-Ford': [], 'A*': []},
            'huge': {'Dijkstra': [], 'LP': [], 'Bellman-Ford': [], 'A*': []},
        }
    else: 
        aggregated_data = {
            'Dijkstra': [],
            'LP': [],
            'Bellman-Ford': [],
            'A*': []
        }

    # Combine all results into one list
    for result_set in all_results:
        for times in result_set:
            # Determine the size of the graph based on the input locations
            for location in locations:
                if location in small_locations:
                    size = 'small'
                elif location in medium_locations:
                    size = 'medium'
                elif location in large_locations:
                    size = 'large'
                elif location in huge_locations:
                    size = 'huge'
                else:
                    continue  # Skip if the location is not recognized

                if flag:
                    # Append results to the appropriate size list
                    aggregated_data[size]['Dijkstra'].append(times['Dijkstra'])
                    aggregated_data[size]['LP'].append(times['LP'])
                    aggregated_data[size]['Bellman-Ford'].append(times['Bellman-Ford'])
                    aggregated_data[size]['A*'].append(times['A*'])
                else:
                    aggregated_data['Dijkstra'].append(times['Dijkstra'])
                    aggregated_data['LP'].append(times['LP'])
                    aggregated_data['Bellman-Ford'].append(times['Bellman-Ford'])
                    aggregated_data['A*'].append(times['A*'])

    return aggregated_data



def analyze_aggregated_results(aggregated_data, flag=False):
    """Analyze the aggregated results and display summary graphs."""
    if not flag:
        # Single combined graph
        df = pd.DataFrame(aggregated_data)

        # Calculate average, min, and max for each algorithm
        summary = df.agg(['mean', 'min', 'max']).T * 1000  # Convert to milliseconds

        # Plot mean execution times
        plt.figure(figsize=(12, 8))
        bar_plot = summary['mean'].plot(kind='bar', color=['blue', 'orange', 'green', 'red'])

        # Annotate bars with mean values
        for p in bar_plot.patches:
            bar_plot.annotate(f'{p.get_height():.2f} ms', 
                              (p.get_x() + p.get_width() / 2., p.get_height()), 
                              ha='center', va='baseline', 
                              xytext=(0, 5), textcoords='offset points')

        plt.title('Average Execution Time of Pathfinding Algorithms'.upper())
        plt.ylabel('Time (ms)')
        plt.xlabel('Algorithms')
        plt.xticks(rotation=45)
        plt.yscale('log')  # Optional log scale for better readability of large time ranges
        plt.tight_layout()
        plt.show()

    else:
        # Multiple subplots for different graph sizes
        fig, axs = plt.subplots(2, 2, figsize=(18, 10))
        fig.suptitle('Average Execution Time Across All Locations by Graph Size')

        # Define titles for each subplot
        titles = ['Small Graphs', 'Medium Graphs', 'Large Graphs', 'Largest Graphs']

        # Iterate through each graph size and plot on corresponding subplot
        for i, (size, results) in enumerate(aggregated_data.items()):
            ax = axs[i // 2, i % 2]  # Select subplot

            # Convert the results into a DataFrame
            df = pd.DataFrame(results)

            # Ensure all algorithms are present before plotting
            if set(['Dijkstra', 'LP', 'Bellman-Ford', 'A*']).issubset(df.columns):
                # Calculate mean, min, and max execution times
                summary = df.agg(['mean', 'min', 'max']).T * 1000  # Convert to milliseconds

                # Plot mean execution times for each algorithm
                bar_plot = summary['mean'].plot(kind='bar', ax=ax, color=['blue', 'orange', 'green', 'red'])
                ax.set_title(titles[i])
                ax.set_ylabel('Time (ms)')
                ax.set_xlabel('Algorithm')
                ax.set_xticks(range(4))
                ax.set_xticklabels(['Dijkstra', 'LP', 'Bellman-Ford', 'A*'], rotation=45)
                ax.set_yscale('log')  # Log scale to account for large time differences

                # Annotate bars with mean values
                for p in bar_plot.patches:
                    bar_plot.annotate(f'{p.get_height():.2f} ms',
                                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                                      ha='center', va='baseline', 
                                      xytext=(0, 5), textcoords='offset points')
            else:
                ax.set_title(f"{titles[i]} (No complete data)")
                ax.set_ylabel('Time (ms)')
                ax.set_xlabel('Algorithm')
                ax.text(0.5, 0.5, 'Insufficient data', horizontalalignment='center', verticalalignment='center')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the main title
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
def main(locations, n, flag=False):
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
    print(all_results)
    # Aggregate results from all locations
    aggregated_data = aggregate_results(all_results, flag)
    print(aggregated_data)
    # Analyze and visualize aggregated results
    if not flag:
        analyze_aggregated_results(aggregated_data)
    else:
        pass
        



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
        for i in range(1, 5):
            if i == 1:
                main(small_locations, 2, True)
            elif i == 2:
                main(medium_locations, 2, True)
            elif i == 3:
                main(large_locations, 2, True)
            else:
                main(huge_locations, 2, True)
    else:
        print("Invalid graph size")
        sys.exit(1)
