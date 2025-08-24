import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import heapq
import time
import random
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

# Set page config
st.set_page_config(
    page_title="Quantum Delivery Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .path-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .quantum-info {
        border-left-color: #ff6b6b;
    }
    .simulation-controls {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    .delivery-status {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        text-align: center;
    }
    .classical-status {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .quantum-status {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class DeliveryOptimizer:
    def __init__(self):
        # Define the city graph based on your images with better coordinates for visualization
        self.nodes = {
            'A': {'lat': 28.7041, 'lon': 77.1025, 'name': 'Node A'},
            'B': {'lat': 28.7141, 'lon': 77.1225, 'name': 'Node B'}, 
            'C': {'lat': 28.6941, 'lon': 77.1125, 'name': 'Node C'},
            'D': {'lat': 28.7241, 'lon': 77.1175, 'name': 'Node D'},
            'E': {'lat': 28.7341, 'lon': 77.1275, 'name': 'Node E'},
            'F': {'lat': 28.7191, 'lon': 77.1325, 'name': 'Node F'},
            'G': {'lat': 28.6841, 'lon': 77.1425, 'name': 'Node G'},
            'H': {'lat': 28.6741, 'lon': 77.1375, 'name': 'Node H'},
            'I': {'lat': 28.6741, 'lon': 77.1225, 'name': 'Node I'},
            'J': {'lat': 28.6641, 'lon': 77.1025, 'name': 'Node J'},
            'W': {'lat': 28.6941, 'lon': 77.1275, 'name': 'Warehouse'}
        }
        
        # Define edges with constraints
        self.edges = {
            ('A', 'D'): {'distance': 8, 'base_time': 9, 'road_type': 'emergency', 'traffic_conditions': {'rainy': 1.3, 'evening_rush': 1.5, 'morning_rush': 1.4}},
            ('A', 'C'): {'distance': 5, 'base_time': 6, 'road_type': 'normal', 'traffic_conditions': {'rainy': 1.2, 'evening_rush': 1.3, 'morning_rush': 1.2}},
            ('C', 'D'): {'distance': 7, 'base_time': 10, 'road_type': 'normal', 'traffic_conditions': {'rainy': 1.4, 'evening_rush': 1.6, 'morning_rush': 1.3}},
            ('C', 'W'): {'distance': 6, 'base_time': 7, 'road_type': 'normal', 'traffic_conditions': {'rainy': 1.2, 'evening_rush': 1.2, 'morning_rush': 1.1}},
            ('C', 'I'): {'distance': 8, 'base_time': 7, 'road_type': 'eco', 'traffic_conditions': {'rainy': 1.1, 'evening_rush': 1.1, 'morning_rush': 1.0}},
            ('D', 'E'): {'distance': 8, 'base_time': 10, 'road_type': 'normal', 'traffic_conditions': {'rainy': 1.3, 'evening_rush': 1.4, 'morning_rush': 1.3}},
            ('D', 'B'): {'distance': 6, 'base_time': 7, 'road_type': 'normal', 'traffic_conditions': {'rainy': 1.2, 'evening_rush': 1.3, 'morning_rush': 1.2}},
            ('D', 'W'): {'distance': 6, 'base_time': 7, 'road_type': 'normal', 'traffic_conditions': {'rainy': 1.2, 'evening_rush': 1.2, 'morning_rush': 1.1}},
            ('E', 'F'): {'distance': 9, 'base_time': 9, 'road_type': 'eco', 'traffic_conditions': {'rainy': 1.1, 'evening_rush': 1.0, 'morning_rush': 1.0}},
            ('B', 'F'): {'distance': 6, 'base_time': 8, 'road_type': 'normal', 'traffic_conditions': {'rainy': 1.2, 'evening_rush': 1.3, 'morning_rush': 1.2}},
            ('B', 'W'): {'distance': 5, 'base_time': 6, 'road_type': 'emergency', 'traffic_conditions': {'rainy': 1.1, 'evening_rush': 1.2, 'morning_rush': 1.1}},
            ('F', 'G'): {'distance': 7, 'base_time': 9, 'road_type': 'normal', 'traffic_conditions': {'rainy': 1.3, 'evening_rush': 1.4, 'morning_rush': 1.3}},
            ('W', 'G'): {'distance': 10, 'base_time': 12, 'road_type': 'emergency', 'traffic_conditions': {'rainy': 1.2, 'evening_rush': 1.3, 'morning_rush': 1.2}},
            ('W', 'H'): {'distance': 6, 'base_time': 7, 'road_type': 'eco', 'traffic_conditions': {'rainy': 1.0, 'evening_rush': 1.0, 'morning_rush': 1.0}},
            ('W', 'I'): {'distance': 4, 'base_time': 5, 'road_type': 'emergency', 'traffic_conditions': {'rainy': 1.1, 'evening_rush': 1.2, 'morning_rush': 1.1}},
            ('I', 'J'): {'distance': 4, 'base_time': 5, 'road_type': 'eco', 'traffic_conditions': {'rainy': 1.0, 'evening_rush': 1.0, 'morning_rush': 1.0}},
            ('I', 'H'): {'distance': 3, 'base_time': 4, 'road_type': 'eco', 'traffic_conditions': {'rainy': 1.0, 'evening_rush': 1.0, 'morning_rush': 1.0}},
            ('H', 'G'): {'distance': 6, 'base_time': 8, 'road_type': 'normal', 'traffic_conditions': {'rainy': 1.2, 'evening_rush': 1.3, 'morning_rush': 1.2}}
        }

    def calculate_edge_weight(self, edge_data, constraints, delivery_type):
        """Calculate dynamic edge weight based on constraints and delivery type"""
        base_time = edge_data['base_time']
        distance = edge_data['distance']
        road_type = edge_data['road_type']
        
        # Apply constraint multipliers
        multiplier = 1.0
        for constraint in constraints:
            if constraint in edge_data['traffic_conditions']:
                multiplier *= edge_data['traffic_conditions'][constraint]
        
        # Delivery type preferences
        if delivery_type == 'emergency':
            if road_type == 'emergency':
                weight = distance * 0.8  # Prefer emergency routes
            else:
                weight = distance * 1.2
        elif delivery_type == 'normal':
            if road_type == 'eco':
                weight = distance * 0.9  # Slightly prefer eco routes
            else:
                weight = distance
        
        # Apply constraint multiplier
        final_weight = weight * multiplier
        
        return final_weight, base_time * multiplier

    def dijkstra(self, start, end, constraints, delivery_type):
        """Classical Dijkstra's algorithm"""
        distances = {node: float('infinity') for node in self.nodes}
        previous = {node: None for node in self.nodes}
        distances[start] = 0
        unvisited = list(self.nodes.keys())
        
        while unvisited:
            current = min(unvisited, key=lambda x: distances[x])
            unvisited.remove(current)
            
            if current == end:
                break
                
            for edge, edge_data in self.edges.items():
                if edge[0] == current and edge[1] in unvisited:
                    neighbor = edge[1]
                elif edge[1] == current and edge[0] in unvisited:
                    neighbor = edge[0]
                else:
                    continue
                    
                weight, time = self.calculate_edge_weight(edge_data, constraints, delivery_type)
                new_distance = distances[current] + weight
                
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current
        
        # Reconstruct path
        path = []
        current = end
        total_time = 0
        total_distance = 0
        
        while current is not None:
            path.append(current)
            if previous[current] is not None:
                # Find edge data for time and distance calculation
                edge = tuple(sorted([current, previous[current]]))
                if edge in self.edges:
                    edge_data = self.edges[edge]
                else:
                    # Try reversed edge
                    edge = (previous[current], current)
                    if edge in self.edges:
                        edge_data = self.edges[edge]
                    else:
                        continue
                        
                _, time = self.calculate_edge_weight(edge_data, constraints, delivery_type)
                total_time += time
                total_distance += edge_data['distance']
                
            current = previous[current]
            
        path.reverse()
        return path, total_distance, total_time

    def qaoa_optimizer(self, start, end, constraints, delivery_type):
        """Quantum-inspired optimization using QAOA principles"""
        # Create adjacency matrix
        nodes_list = list(self.nodes.keys())
        n_nodes = len(nodes_list)
        node_to_idx = {node: i for i, node in enumerate(nodes_list)}
        
        # Build cost matrix
        cost_matrix = np.full((n_nodes, n_nodes), np.inf)
        np.fill_diagonal(cost_matrix, 0)
        
        for edge, edge_data in self.edges.items():
            i, j = node_to_idx[edge[0]], node_to_idx[edge[1]]
            weight, _ = self.calculate_edge_weight(edge_data, constraints, delivery_type)
            cost_matrix[i][j] = weight
            cost_matrix[j][i] = weight
        
        # QAOA-inspired optimization
        start_idx = node_to_idx[start]
        end_idx = node_to_idx[end]
        
        # Use quantum-inspired variational approach
        best_path = None
        best_cost = float('inf')
        
        # Multiple random initializations (simulating quantum superposition)
        for _ in range(100):  # Simulate multiple quantum measurements
            # Generate random path with quantum-inspired randomness
            path = [start_idx]
            current = start_idx
            visited = {start_idx}
            
            while current != end_idx and len(visited) < n_nodes:
                # Quantum-inspired probabilistic selection
                neighbors = []
                for i in range(n_nodes):
                    if i not in visited and cost_matrix[current][i] != np.inf:
                        neighbors.append(i)
                
                if not neighbors:
                    break
                    
                if end_idx in neighbors:
                    path.append(end_idx)
                    break
                else:
                    # Weighted random selection based on inverse cost
                    weights = [1.0 / (cost_matrix[current][neighbor] + 1e-6) for neighbor in neighbors]
                    total_weight = sum(weights)
                    weights = [w/total_weight for w in weights]
                    
                    next_node = np.random.choice(neighbors, p=weights)
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
            
            # Calculate path cost
            if path[-1] == end_idx:
                total_cost = 0
                total_time = 0
                total_distance = 0
                
                for i in range(len(path) - 1):
                    curr_node = nodes_list[path[i]]
                    next_node = nodes_list[path[i + 1]]
                    
                    edge = tuple(sorted([curr_node, next_node]))
                    if edge in self.edges:
                        edge_data = self.edges[edge]
                    else:
                        edge = (curr_node, next_node)
                        if edge in self.edges:
                            edge_data = self.edges[edge]
                        else:
                            continue
                    
                    weight, time = self.calculate_edge_weight(edge_data, constraints, delivery_type)
                    total_cost += weight
                    total_time += time
                    total_distance += edge_data['distance']
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = [nodes_list[i] for i in path]
                    best_distance = total_distance
                    best_time = total_time
        
        return best_path, best_distance, best_time

def create_enhanced_map(optimizer, classical_path, quantum_path, destination, classical_agent_pos=None, quantum_agent_pos=None):
    """Create enhanced Folium map with both paths and animated agents"""
    # Calculate center of map
    center_lat = sum([node['lat'] for node in optimizer.nodes.values()]) / len(optimizer.nodes)
    center_lon = sum([node['lon'] for node in optimizer.nodes.values()]) / len(optimizer.nodes)
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    # Add all nodes with clear labels
    for node_id, node_data in optimizer.nodes.items():
        if node_id == 'W':
            # Warehouse with special styling
            folium.Marker(
                [node_data['lat'], node_data['lon']],
                popup=f"🏪 {node_data['name']}",
                icon=folium.Icon(color='orange', icon='home', prefix='fa'),
                tooltip=f"🏪 Warehouse (Start Point)"
            ).add_to(m)
            
            # Add warehouse label
            folium.Marker(
                [node_data['lat'] + 0.003, node_data['lon']],
                icon=folium.DivIcon(
                    html=f"""<div style="font-size: 14px; font-weight: bold; 
                             color: white; background-color: orange; 
                             padding: 2px 6px; border-radius: 3px; 
                             border: 2px solid black;">W</div>""",
                    icon_size=(20, 20),
                    icon_anchor=(10, 10)
                )
            ).add_to(m)
            
        elif node_id == destination:
            # Destination with special styling
            folium.Marker(
                [node_data['lat'], node_data['lon']],
                popup=f"📍 {node_data['name']} (Destination)",
                icon=folium.Icon(color='red', icon='flag', prefix='fa'),
                tooltip=f"📍 Destination: {node_id}"
            ).add_to(m)
            
            # Add destination label
            folium.Marker(
                [node_data['lat'] + 0.003, node_data['lon']],
                icon=folium.DivIcon(
                    html=f"""<div style="font-size: 14px; font-weight: bold; 
                             color: white; background-color: red; 
                             padding: 2px 6px; border-radius: 3px; 
                             border: 2px solid black;">{node_id}</div>""",
                    icon_size=(20, 20),
                    icon_anchor=(10, 10)
                )
            ).add_to(m)
            
        else:
            # Regular nodes
            folium.CircleMarker(
                [node_data['lat'], node_data['lon']],
                popup=f"Node {node_id}",
                radius=12,
                fillOpacity=0.8,
                color='black',
                fillColor='lightblue',
                weight=2,
                tooltip=f"Node {node_id}"
            ).add_to(m)
            
            # Add node label
            folium.Marker(
                [node_data['lat'], node_data['lon']],
                icon=folium.DivIcon(
                    html=f"""<div style="font-size: 12px; font-weight: bold; 
                             color: black; text-align: center; 
                             margin-top: -6px;">{node_id}</div>""",
                    icon_size=(20, 20),
                    icon_anchor=(10, 10)
                )
            ).add_to(m)
    
    # Add classical path (blue)
    if classical_path and len(classical_path) > 1:
        classical_coords = [[optimizer.nodes[node]['lat'], optimizer.nodes[node]['lon']] for node in classical_path]
        folium.PolyLine(
            classical_coords,
            color='blue',
            weight=6,
            opacity=0.8,
            popup="🔵 Classical Path (Dijkstra)",
            tooltip="Classical Route"
        ).add_to(m)
        
        # Add path direction arrows
        for i in range(len(classical_coords) - 1):
            mid_lat = (classical_coords[i][0] + classical_coords[i+1][0]) / 2
            mid_lon = (classical_coords[i][1] + classical_coords[i+1][1]) / 2
            folium.Marker(
                [mid_lat, mid_lon],
                icon=folium.DivIcon(
                    html="""<div style="color: blue; font-size: 16px; font-weight: bold;">→</div>""",
                    icon_size=(20, 20),
                    icon_anchor=(10, 10)
                )
            ).add_to(m)
    
    # Add quantum path (red)
    if quantum_path and len(quantum_path) > 1:
        quantum_coords = [[optimizer.nodes[node]['lat'], optimizer.nodes[node]['lon']] for node in quantum_path]
        folium.PolyLine(
            quantum_coords,
            color='red',
            weight=6,
            opacity=0.8,
            popup="🔴 Quantum Path (QAOA)",
            tooltip="Quantum Route"
        ).add_to(m)
        
        # Add path direction arrows
        for i in range(len(quantum_coords) - 1):
            mid_lat = (quantum_coords[i][0] + quantum_coords[i+1][0]) / 2
            mid_lon = (quantum_coords[i][1] + quantum_coords[i+1][1]) / 2
            folium.Marker(
                [mid_lat, mid_lon],
                icon=folium.DivIcon(
                    html="""<div style="color: red; font-size: 16px; font-weight: bold;">→</div>""",
                    icon_size=(20, 20),
                    icon_anchor=(10, 10)
                )
            ).add_to(m)
    
    # Add animated delivery agents
    if classical_agent_pos and classical_path:
        agent_node = classical_path[classical_agent_pos] if classical_agent_pos < len(classical_path) else classical_path[-1]
        folium.Marker(
            [optimizer.nodes[agent_node]['lat'], optimizer.nodes[agent_node]['lon']],
            popup="🚚 Classical Delivery Agent",
            icon=folium.Icon(color='blue', icon='truck', prefix='fa'),
            tooltip="Classical Agent"
        ).add_to(m)
    
    if quantum_agent_pos and quantum_path:
        agent_node = quantum_path[quantum_agent_pos] if quantum_agent_pos < len(quantum_path) else quantum_path[-1]
        folium.Marker(
            [optimizer.nodes[agent_node]['lat'], optimizer.nodes[agent_node]['lon']],
            popup="🚚 Quantum Delivery Agent",
            icon=folium.Icon(color='darkred', icon='truck', prefix='fa'),
            tooltip="Quantum Agent"
        ).add_to(m)
    
    return m

def main():
    st.markdown('<h1 class="main-header">🚚 Quantum Delivery Route Optimizer</h1>', unsafe_allow_html=True)
    
    # Initialize optimizer
    optimizer = DeliveryOptimizer()
    
    # Initialize session state
    if 'classical_agent_pos' not in st.session_state:
        st.session_state.classical_agent_pos = 0
    if 'quantum_agent_pos' not in st.session_state:
        st.session_state.quantum_agent_pos = 0
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'paths_calculated' not in st.session_state:
        st.session_state.paths_calculated = False
    
    # Sidebar controls
    st.sidebar.header("🎛️ Delivery Configuration")
    
    # Destination selection
    destinations = [node for node in optimizer.nodes.keys() if node != 'W']
    destination = st.sidebar.selectbox("📍 Select Destination", destinations, index=destinations.index('G'))
    
    # Delivery type
    delivery_type = st.sidebar.radio(
        "🚚 Delivery Type",
        ["normal", "emergency"],
        format_func=lambda x: "🌱 Normal Delivery" if x == "normal" else "🚨 Emergency Delivery"
    )
    
    # Constraints
    st.sidebar.subheader("🌦️ Current Conditions")
    constraints = []
    rainy = st.sidebar.checkbox("🌧️ Rainy Day")
    if rainy:
        constraints.append("rainy")
    morning_rush = st.sidebar.checkbox("🌅 Morning Rush Hour")
    if morning_rush:
        constraints.append("morning_rush")
    evening_rush = st.sidebar.checkbox("🌆 Evening Rush Hour")
    if evening_rush:
        constraints.append("evening_rush")
    
    # Auto-calculate paths when constraints change
    current_config = (destination, delivery_type, tuple(constraints))
    if 'last_config' not in st.session_state or st.session_state.last_config != current_config:
        st.session_state.last_config = current_config
        st.session_state.paths_calculated = False
        st.session_state.classical_agent_pos = 0
        st.session_state.quantum_agent_pos = 0
    
    # Calculate paths
    if not st.session_state.paths_calculated:
        with st.spinner("🧠 Calculating optimal routes..."):
            classical_path, classical_distance, classical_time = optimizer.dijkstra('W', destination, constraints, delivery_type)
            quantum_path, quantum_distance, quantum_time = optimizer.qaoa_optimizer('W', destination, constraints, delivery_type)
            
            st.session_state.classical_path = classical_path
            st.session_state.quantum_path = quantum_path
            st.session_state.classical_distance = classical_distance
            st.session_state.quantum_distance = quantum_distance
            st.session_state.classical_time = classical_time
            st.session_state.quantum_time = quantum_time
            st.session_state.paths_calculated = True
    
    # Live Demo Simulations (above the map)
    st.markdown('<div class="simulation-controls">', unsafe_allow_html=True)
    st.subheader("🎬 Live Delivery Simulations")
    
    simulation_col1, simulation_col2, simulation_col3 = st.columns([1, 1, 1])
    
    with simulation_col1:
        if st.button("▶️ Start Classical Simulation", key="start_classical"):
            st.session_state.classical_agent_pos = 0
            st.session_state.classical_simulation_running = True
    
    with simulation_col2:
        if st.button("▶️ Start Quantum Simulation", key="start_quantum"):
            st.session_state.quantum_agent_pos = 0
            st.session_state.quantum_simulation_running = True
    
    with simulation_col3:
        if st.button("🔄 Reset Simulations", key="reset_sims"):
            st.session_state.classical_agent_pos = 0
            st.session_state.quantum_agent_pos = 0
            st.session_state.classical_simulation_running = False
            st.session_state.quantum_simulation_running = False
    
    # Delivery status displays
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        if hasattr(st.session_state, 'classical_path'):
            classical_progress = (st.session_state.classical_agent_pos / max(1, len(st.session_state.classical_path) - 1)) * 100
            current_classical_node = st.session_state.classical_path[st.session_state.classical_agent_pos] if st.session_state.classical_agent_pos < len(st.session_state.classical_path) else st.session_state.classical_path[-1]
            
            st.markdown(f"""
            <div class="delivery-status classical-status">
                🔵 Classical Agent<br>
                Location: Node {current_classical_node}<br>
                Progress: {classical_progress:.0f}%
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(classical_progress / 100)
    
    with status_col2:
        if hasattr(st.session_state, 'quantum_path'):
            quantum_progress = (st.session_state.quantum_agent_pos / max(1, len(st.session_state.quantum_path) - 1)) * 100
            current_quantum_node = st.session_state.quantum_path[st.session_state.quantum_agent_pos] if st.session_state.quantum_agent_pos < len(st.session_state.quantum_path) else st.session_state.quantum_path[-1]
            
            st.markdown(f"""
            <div class="delivery-status quantum-status">
                🔴 Quantum Agent<br>
                Location: Node {current_quantum_node}<br>
                Progress: {quantum_progress:.0f}%
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(quantum_progress / 100)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto-advance simulations
    if getattr(st.session_state, 'classical_simulation_running', False):
        if st.session_state.classical_agent_pos < len(st.session_state.classical_path) - 1:
            time.sleep(1)
            st.session_state.classical_agent_pos += 1
            st.rerun()
        else:
            st.session_state.classical_simulation_running = False
            st.success("🔵 Classical delivery completed!")
    
    if getattr(st.session_state, 'quantum_simulation_running', False):
        if st.session_state.quantum_agent_pos < len(st.session_state.quantum_path) - 1:
            time.sleep(1)
            st.session_state.quantum_agent_pos += 1
            st.rerun()
        else:
            st.session_state.quantum_simulation_running = False
            st.success("🔴 Quantum delivery completed!")
    
    # Enhanced Map Display
    st.subheader("🗺️ Interactive Route Visualization")
    
    if st.session_state.paths_calculated:
        # Create map with both paths and agent positions
        map_obj = create_enhanced_map(
            optimizer, 
            st.session_state.classical_path, 
            st.session_state.quantum_path, 
            destination,
            st.session_state.classical_agent_pos,
            st.session_state.quantum_agent_pos
        )
        
        # Display map
        map_data = st_folium(map_obj, width=1000, height=600, returned_objects=["last_object_clicked"])
    
    # Route Information and Comparison
    if st.session_state.paths_calculated:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="path-info">', unsafe_allow_html=True)
            st.markdown(f"<strong>Classical Path:</strong> {st.session_state.classical_path}", unsafe_allow_html=True)
            st.markdown(f"<strong>Current Node:</strong> {st.session_state.classical_path[st.session_state.classical_agent_pos]}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="path-info">', unsafe_allow_html=True)
            st.markdown(f"<strong>Quantum Path:</strong> {st.session_state.quantum_path}", unsafe_allow_html=True)
            st.markdown(f"<strong>Current Node:</strong> {st.session_state.quantum_path[st.session_state.quantum_agent_pos]}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance Metrics and Comparison
    if st.session_state.paths_calculated:
        st.subheader("📊 Performance Comparison")
        
        # Create metrics columns
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Classical Distance", f"{st.session_state.classical_distance:.1f} km")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Quantum Distance", f"{st.session_state.quantum_distance:.1f} km")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Classical Time", f"{st.session_state.classical_time:.1f} min")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Quantum Time", f"{st.session_state.quantum_time:.1f} min")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Calculate improvements
        time_improvement = ((st.session_state.classical_time - st.session_state.quantum_time) / st.session_state.classical_time) * 100
        distance_improvement = ((st.session_state.classical_distance - st.session_state.quantum_distance) / st.session_state.classical_distance) * 100
        
        # Improvement metrics
        improvement_col1, improvement_col2 = st.columns(2)
        
        with improvement_col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Time Improvement", f"{time_improvement:.1f}%", 
                     delta_color="inverse" if time_improvement < 0 else "normal")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with improvement_col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Distance Improvement", f"{distance_improvement:.1f}%", 
                     delta_color="inverse" if distance_improvement < 0 else "normal")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed analysis
        st.markdown('<div class="path-info quantum-info">', unsafe_allow_html=True)
        st.markdown("### 🔍 Route Analysis")
        
        if time_improvement > 0:
            st.markdown(f"✅ **Quantum advantage**: Quantum algorithm found a route that is **{time_improvement:.1f}% faster** than classical Dijkstra!")
        else:
            st.markdown(f"⚠️ **No quantum advantage**: Classical algorithm performed better in this scenario.")
        
        if distance_improvement > 0:
            st.markdown(f"✅ **Route efficiency**: Quantum path is **{distance_improvement:.1f}% shorter** in distance!")
        
        # Additional insights based on delivery type
        if delivery_type == 'emergency':
            st.markdown("🚨 **Emergency delivery**: Quantum algorithm prioritizes fastest routes for critical deliveries.")
        else:
            st.markdown("🌱 **Normal delivery**: Both algorithms consider balanced route optimization.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show current constraints
        if constraints:
            st.markdown('<div class="path-info">', unsafe_allow_html=True)
            st.markdown("### 🌦️ Active Constraints")
            constraint_icons = {
                'rainy': '🌧️ Rainy Conditions',
                'morning_rush': '🌅 Morning Rush Hour', 
                'evening_rush': '🌆 Evening Rush Hour'
            }
            for constraint in constraints:
                if constraint in constraint_icons:
                    st.markdown(f"- {constraint_icons[constraint]}")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
