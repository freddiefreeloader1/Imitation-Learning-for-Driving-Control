from graphviz import Digraph

# Create a directed graph
dot = Digraph()

# Add nodes and edges to the tree
dot.node('A', 'Methods')
dot.node('B', 'BC')
dot.node('C', 'BC with Data Augmentation')
dot.node('D', 'Ellipsoids')
dot.node('E', 'No Ellipsoids')
dot.node('F', 'Training Noise')
dot.node('G', 'Inference Noise')
dot.node('H', 'Path to Stabilize')
dot.node('I', 'Hyperparameters')

dot.node("J", "Mean Trajectory")
dot.node("K", "Each Trajectory")





dot.edges(['AB', 'AC', 'CD', 'CE', 'DI', 'EI', "IF","IG","IH","HJ","HK"])

# Render the tree to a file
dot.render('tree_visualization', format='png', view=True)