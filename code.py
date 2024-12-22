# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque

## IMPORTANT
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    
    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    # Create trip_id to route_id mapping
    ttr = dict(zip(df_trips['trip_id'], df_trips['route_id']))

    trip_to_route=ttr
    # Map route_id to a list of stops in order of their sequence
    rts = (df_stop_times.merge(df_trips[['trip_id', 'route_id']], on='trip_id').sort_values(by=['route_id', 'stop_sequence']).drop_duplicates(subset=['route_id', 'stop_id'])  .groupby('route_id')['stop_id'].apply(list).to_dict()) 
      
    route_to_stops=rts 
    # Ensure each route only has unique stops
    
    # Count trips per stop
    stc = df_stop_times['stop_id'].value_counts().to_dict()

    stop_trip_count=stc
    # Create fare rules for routes

    # Merge fare rules and attributes into a single DataFrame

    fr = df_fare_rules[['route_id', 'origin_id', 'destination_id']]
    fare_rules=fr
    mfd = df_fare_rules.merge(df_fare_attributes, on='fare_id', how='left') 
    merged_fare_df=mfd


def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """
    ans = defaultdict(int)

    values_iterator = iter(trip_to_route.values())

    while True:
        try:
            i = next(values_iterator)
            ans[i] += 1
        except StopIteration:
            break
        
    t=ans.items()

    return sorted(t, key=lambda x: x[1], reverse=True)[:5]

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    t=stop_trip_count.items()
    return sorted(t, key=lambda x: x[1], reverse=True)[:5]


def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    ans = defaultdict(set)

    route_to_stops_iterator = iter(route_to_stops.items())
    while True:
        try:
            i, j = next(route_to_stops_iterator)
            j_iterator = iter(j)
            while True:
                try:
                    k = next(j_iterator)
                    ans[k].add(i)
                except StopIteration:
                    break
        except StopIteration:
            break

    f=ans.items()
    return sorted(((k, len(j)) for k, j in f), key=lambda x: x[1], reverse=True)[:5]

def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route. 
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    lis = df_stop_times.merge(df_trips[['trip_id', 'route_id']], on='trip_id')

    lis = lis.sort_values(['trip_id', 'stop_sequence'])

    stop_pairs = ( lis.assign(next_stop_id=lis.groupby('trip_id')['stop_id'].shift(-1)) .dropna(subset=['next_stop_id'])  )

    unique_route_pairs = ( stop_pairs.groupby(['stop_id', 'next_stop_id'])['route_id'] .nunique() .reset_index() .query('route_id == 1')   .drop(columns='route_id')  )

    unique_route_pairs = unique_route_pairs.merge(  stop_pairs[['stop_id', 'next_stop_id', 'route_id']], on=['stop_id', 'next_stop_id']  ).drop_duplicates()

    unique_route_pairs['combined_trip_count'] = (unique_route_pairs['stop_id'].map(stop_trip_count) + unique_route_pairs['next_stop_id'].map(stop_trip_count) )

    ans = unique_route_pairs.nlargest(5, 'combined_trip_count')

    result = []

    rows_iterator = iter(ans.iterrows())

    while True:
        try:
            _, row = next(rows_iterator)
            result.append((int(row['stop_id']), int(row['next_stop_id']), int(row['route_id'])))
        except StopIteration:
            break
        
    return result

# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.
    
    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """

    data_dict = route_to_stops

    x_vals = list(data_dict.keys())
    
    y_vals = [data_dict[key] for key in x_vals]

    traces = []
    for i, y_list in enumerate(y_vals):
        trace = go.Scatter(x=[x_vals[i]] * len(y_list), y=y_list,mode='markers+lines',name=f"y-values for x={x_vals[i]}")
        traces.append(trace)

    fig = go.Figure(data=traces)

    fig.update_layout(title="Interactive Plot for route_to_stops",xaxis_title="X-axis" ,yaxis_title="Y-axis ",showlegend=True)

    fig.show()

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    ans = []
    
    route_to_stops_iterator = iter(route_to_stops.items())
    while True:
        try:
            i, j = next(route_to_stops_iterator)
            if start_stop in j and end_stop in j:
                start_index = j.index(start_stop)
                end_index = j.index(end_stop)
                
                if start_index < end_index:
                    ans.append(i)
        except StopIteration:
            break
                
    return ans
                                                                                                                                                                    

# Initialize Datalog predicates for reasoning

pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')

def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  

    # Define Datalog predicates 
    DirectRoute(X, Y, R) <= RouteHasStop(R, X) & RouteHasStop(R, Y) & (X != Y)

    OptimalRoute(R1, R2, X, Y, Z) <= (DirectRoute(X, Z, R1) & DirectRoute(Z, Y, R2) & (R1 != R2))

    create_kb()  
    add_route_data(route_to_stops)  


# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    route_to_stops_iterator = iter(route_to_stops.items())
    while True:
        try:
            route_id, stops = next(route_to_stops_iterator)
            stops_iterator = iter(stops)
            while True:
                try:
                    stop = next(stops_iterator)
                    + RouteHasStop(route_id, stop)
                except StopIteration:
                    break
            
            i = 0
            while i < len(stops) - 1:
                + DirectRoute(stops[i], stops[i + 1], route_id)
                i += 1
        except StopIteration:
            break

# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.
    pass

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """
    query = DirectRoute(start, end, R)
    g=query.data
    result = sorted({row[0] if isinstance(row, tuple) else row for row in g})
    
    return result

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    result = OptimalRoute(R1, R2, start_stop_id, end_stop_id, stop_id_to_include).ask()

    # Process the results
    optimal_routes = sorted([(r1, stop_id_to_include, r2) for r1, r2 in result])

    return optimal_routes

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    result = OptimalRoute(R1, R2, start_stop_id, end_stop_id, stop_id_to_include).ask()


    optimal_routes = sorted([(r1, stop_id_to_include, r2) for r1, r2 in result])
    ans =[]
    for i in range(len(optimal_routes)):
        ans.append((optimal_routes[i][-1],optimal_routes[i][-2],optimal_routes[i][-3]))

    return ans

# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    ans = []
    
    lis1 = query_direct_routes(start_stop_id, stop_id_to_include)
    
    lis2 = query_direct_routes(stop_id_to_include, end_stop_id)

    outer_iterator = iter(lis1)
    while True:
        try:
            i = next(outer_iterator)
            inner_iterator = iter(lis2)
            while True:
                try:
                    j = next(inner_iterator)
                    if i != j:
                        ans.append((i, stop_id_to_include, j))
                except StopIteration:
                    break
        except StopIteration:
            break
    
    return ans

# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    return merged_fare_df[merged_fare_df['price'] <= initial_fare]

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    route_summary = {}
    
    route_fares = pruned_df.groupby('route_id')['price'].min().to_dict()
    
    route_to_stops_iterator = iter(route_to_stops.items())
    while True:
        try:
            route_id, stops = next(route_to_stops_iterator)
            route_summary[route_id] = {'min_price': route_fares.get(route_id, float('inf')),'stops': set(stops)}
        except StopIteration:
            break
    
    return route_summary
    

    

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    queue = deque([(start_stop_id, None, [], 0, 0)])  
    visited = set()
    
    while queue:     
        current_stop, current_route, path_taken, total_fare, transfers = queue.popleft()

        if current_stop == end_stop_id:
            my_list=path_taken + [(current_route, current_stop)]
            ans=[]

            for i in range(len(my_list)-1):
                ans.append((my_list[i][0],my_list[i+1][1]))


            return ans 
        
        if total_fare > initial_fare or transfers > max_transfers:
            continue
        
        visited.add((current_stop, current_route))

        route_summary_iterator = iter(route_summary.items())
        while True:
            try:
                route_id, route_info = next(route_summary_iterator)
                if route_id == current_route:
                    continue
                
                if current_stop in route_info['stops'] and route_info['min_price'] + total_fare <= initial_fare:
                    new_path = path_taken + [(route_id, current_stop)]
                    new_fare = total_fare + route_info['min_price']
                    new_transfers = transfers + (1 if current_route else 0)

                    stops_iterator = iter(route_info['stops'])
                    while True:
                        try:
                            stop = next(stops_iterator)
                            if (stop, route_id) not in visited:
                                queue.append((stop, route_id, new_path, new_fare, new_transfers))
                        except StopIteration:
                            break
            except StopIteration:
                break
    
    return []