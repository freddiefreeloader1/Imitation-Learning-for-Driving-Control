from graphviz import Digraph

# Create a directed graph
dot = Digraph()

# Set the graph to be vertical
dot.attr(rankdir='TB')  # Top-to-bottom layout for the whole graph

# Add nodes and edges for Methods and BC-related nodes
dot.node('A', 'Methods')
dot.node('B', 'BC')
dot.node('C', 'BC with Data Augmentation with adding noise outside of distribution')
dot.node('D', 'BC with Data Augmentation with adding noise everywhere')

# Add nodes for Noise-related items
dot.node('F', 'Training Noise')
dot.node('G', 'Inference Noise')
dot.node('H', 'Path to Stabilize')

# Add nodes for Hyperparameters and Trajectories
dot.node('I', 'Hyperparameters')
dot.node("J", "Mean Trajectory")
dot.node("K", "Each Trajectory")

# Define the edges
dot.edge('A', 'B')
dot.edge('A', 'C')
dot.edge('A', 'D')
dot.edge('I', 'F')
dot.edge('I', 'G')
dot.edge('I', 'H')
dot.edge('H', 'J')
dot.edge('H', 'K')

# Render the tree to a file
dot.render('tree_visualization', format='png', view=True)
