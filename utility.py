import pandas as pd

def reduce_trips(input_stops_file, input_stop_times_file, output_stop_times_file, fraction=0.5):
    """
    Reduces the size of the stop_times file by selecting a fraction of trips.

    Parameters:
        input_stops_file (str): Path to the stops.txt file.
        input_stop_times_file (str): Path to the stop_times.txt file.
        output_stop_times_file (str): Path to save the reduced stop_times.txt file.
        fraction (float): The fraction of trips to retain (default is 0.5, or 50%).

    Returns:
        None: The filtered stop_times is saved to output_stop_times_file.
    """
    # Load the stops and stop_times data
    stops = pd.read_csv(input_stops_file)
    stop_times = pd.read_csv(input_stop_times_file)

    # Get unique trip_ids
    unique_trip_ids = stop_times['trip_id'].unique()

    # Calculate how many trip_ids to keep based on the fraction
    num_to_keep = int(len(unique_trip_ids) * fraction)

    # Take the specified fraction of the trip_ids
    selected_trip_ids = unique_trip_ids[:num_to_keep]

    # Filter the stop_times to keep only those trips
    filtered_stop_times = stop_times[stop_times['trip_id'].isin(selected_trip_ids)]

    # Save the reduced stop_times to a new file
    filtered_stop_times.to_csv(output_stop_times_file, index=False)

    print(f"Filtered data saved to '{output_stop_times_file}' with {fraction*100}% of trips.")


def reduce_stops(input_stops_file, input_stop_times_file, output_stops_file):
    """
    Reduces the stops file to include only those referenced in the stop_times file.

    Parameters:
        input_stops_file (str): Path to the stops.txt file.
        input_stop_times_file (str): Path to the stop_times.txt file.
        output_stops_file (str): Path to save the reduced stops.txt file.

    Returns:
        None: The filtered stops is saved to output_stops_file.
    """
    # Load the stops and stop_times data
    stops = pd.read_csv(input_stops_file)
    stop_times = pd.read_csv(input_stop_times_file)

    # Get unique stop_ids from the stop_times file
    unique_stop_ids = stop_times['stop_id'].unique()

    # Filter the stops to keep only those referenced in stop_times
    filtered_stops = stops[stops['stop_id'].isin(unique_stop_ids)]

    # Save the reduced stops to a new file
    filtered_stops.to_csv(output_stops_file, index=False)

    print(f"Filtered stops saved to '{output_stops_file}' with {len(filtered_stops)} stops.")
