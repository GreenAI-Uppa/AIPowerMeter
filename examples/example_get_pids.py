import networkx as nx 
import measure_utils


process_tree = measure_utils.get_pids()
labels = nx.get_node_attributes(process_tree, 'user')
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
pos = graphviz_layout(process_tree, prog="twopi", args="")
plt.figure(1, figsize=(8, 8))
nx.draw(process_tree, pos, labels=labels, node_size=20, alpha=0.5, node_color="blue", with_labels=True)
plt.axis("equal")

labels = nx.get_node_attributes(process_tree, 'name')
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
pos = graphviz_layout(process_tree, prog="twopi", args="")
plt.figure(2, figsize=(8, 8))
nx.draw(process_tree, pos, labels=labels, node_size=20, alpha=0.5, node_color="blue", with_labels=True)
plt.axis("equal")
plt.show()
