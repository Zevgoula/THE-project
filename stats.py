from datetime import timedelta
import random
import sys
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pulp
# Import your algorithms
import Algorithms.linearProgrammingAlgorithm as lp
import Algorithms.dijkstraAlgorithm as dj
import Algorithms.bellmanFordAlgorithm as bl
import Algorithms.astarAlgorithm as astar

locations = ["athens", "thessaloniki", "italy", "spain", "canada"]
graphs = {}
def compare_algorithms(graph, unique_pairs):
    results = []

    for (start, end) in unique_pairs:
        times = {}
        
        # Track the paths found
        paths = {}
        weights = {}
        
        time0 = timeit.default_timer()
        paths['Dijkstra'], weights['Dijkstra'] = dj.dijkstra_shortest_path(graph, start, end)
        if paths['Dijkstra'] is None:
            print(f"No path found for {start} to {end}. Skipping comparison.")
            continue
        time1 = timeit.default_timer()
        
        paths['LP'], weights['LP']= lp.lp_shortest_path(graph, start, end)
        time2 = timeit.default_timer()
        
        paths['Bellman-Ford'], weights['Bellman-Ford'] = bl.bellman_ford_shortest_path(graph, start, end)
        time3 = timeit.default_timer()
        
        paths['A*'], weights['A*'] = astar.astar_shortest_path(graph, start, end)
        time4 = timeit.default_timer()
        
        # Check if all paths are exactly the same and not None
        if all(weight == weights['LP'] for weight in weights.values()):
            # Only compare if all algorithms found the same exact path
            times['Dijkstra'] = time1 - time0
            times['LP'] = time2 - time1
            times['Bellman-Ford'] = time3 - time2
            times['A*'] = time4 - time3

            results.append((start, end, times))
        else:
            print(f"Different paths found for {start} to {end}. Skipping comparison.")
            print("GAYYYYYYYYYYYY")
            for algorithm, path in paths.items():
                if path is not None:
                    print(f"{algorithm}: {path} weight: {weights[algorithm]}")
                else:
                    print(f"{algorithm}: No path found")

    return results


def analyze_results(results):
    data = {
        'From': [],
        'To': [],
        'Dijkstra': [],
        'LP': [],
        'Bellman-Ford': [],
        'A*': []
    }

    for start, end, times in results:
        data['From'].append(start)
        data['To'].append(end)
        for algorithm in ['Dijkstra', 'LP', 'Bellman-Ford', 'A*']:
            data[algorithm].append(times[algorithm])

    df = pd.DataFrame(data)

    # Calculate average, min, and max for each algorithm
    summary = df[['Dijkstra', 'LP', 'Bellman-Ford', 'A*']].agg(['mean', 'min', 'max']).T
    summary = summary*1000  # Convert to milliseconds
    print(summary)
    
    plt.figure(figsize=(12, 8))

    # Plotting the average times
    summary['mean'].plot(kind='bar', color=['blue', 'orange', 'green', 'red'])
    plt.title('Average Execution Time of Pathfinding Algorithms')
    plt.ylabel('Time (ms)')
    plt.xlabel('Algorithm')
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.show()
    
def handle_extended_hours(time_str):
        hours, minutes, seconds = map(int, time_str.split(':'))
        if hours >= 24:
            return timedelta(days=1, hours=hours-24, minutes=minutes, seconds=seconds)
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    
def main():
    for location in locations:
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
        
        graphs[location] = G

        # Generate 20 random unique pairs
        all_nodes = list(G.nodes())
        unique_pairs = random.sample([(node1, node2) for node1 in all_nodes for node2 in all_nodes if node1 != node2], 50)

        # Compare algorithms on the current graph
        results = compare_algorithms(G, unique_pairs)
        analyze_results(results)

if __name__ == '__main__':
    main()
