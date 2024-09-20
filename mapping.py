import folium
import osmnx as ox
import networkx as nx
import sys

def create_map_with_route(start_point, end_point, location='New York City, New York, USA'):
    try:
        # Download the street network for the specified location
        G = ox.graph_from_place(location, network_type='drive')
        
        # Convert points to the nearest nodes in the graph
        start_node = ox.distance.nearest_nodes(G, X=start_point[1], Y=start_point[0])
        end_node = ox.distance.nearest_nodes(G, X=end_point[1], Y=end_point[0])
        
        # Use Dijkstra's algorithm to find the shortest path
        shortest_path = nx.shortest_path(G, source=start_node, target=end_node, weight='length')
        
        # Create a Folium map centered around the start point
        mymap = folium.Map(location=start_point, zoom_start=14)
        
        # Add markers for start and end points
        folium.Marker(start_point, popup='Start Point').add_to(mymap)
        folium.Marker(end_point, popup='End Point').add_to(mymap)
        
        # Add the route
        route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in shortest_path]
        folium.PolyLine(route_coords, color='blue', weight=5, opacity=0.8).add_to(mymap)
        
        # Save the map
        mymap.save("route_map_with_markers.html")
        print("Map saved as route_map_with_markers.html")
    
    except ImportError as e:
        print(f"ImportError: {e}. Please install the required dependencies.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        import scikit_learn
    except ImportError:
        print("scikit-learn is not installed. Installing now...")
        import subprocess
          subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
        print("scikit-learn installed successfully. Please re-run the script.")
        sys.exit(1)

    # Define start and end points (latitude, longitude)
    start_point = (40.748817, -73.985428)  # Example: Empire State Building
    end_point = (40.730610, -73.935242)  # Exampl
