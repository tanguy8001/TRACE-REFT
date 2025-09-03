import torch
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

base_dir = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/REFT-CL"
layers = [4, 6, 10, 12]
num_tasks = 8

# Define colors per layer
colors = plt.get_cmap("tab10").colors
layer_colors = {layer: colors[i] for i, layer in enumerate(layers)}

# Markers for matrix types
markers = {'weight': 'o', 'rotate_layer': '^'}

# Collect matrices
data = {'weight': [], 'rotate_layer': []}
layer_labels = {'weight': [], 'rotate_layer': []}
task_labels = {'weight': [], 'rotate_layer': []}

for layer in layers:
    for task_nb in range(num_tasks):
        file_path = os.path.join(
            base_dir, str(task_nb),
            f"intkey_layer_{layer}_comp_block_output_unit_pos_nunit_1#0.bin"
        )
        if not os.path.exists(file_path):
            print(f"⚠️ Missing file: {file_path}")
            continue

        state = torch.load(file_path, map_location="cpu")

        for mat_type in ['weight', 'rotate_layer']:
            key = f"tasks.{task_nb}.{mat_type}"
            if key in state:
                data[mat_type].append(state[key].float().flatten().numpy())
                layer_labels[mat_type].append(layer)
                task_labels[mat_type].append(task_nb)

# Convert to arrays
weight_data = np.stack(data['weight'])
rotate_data = np.stack(data['rotate_layer'])

# PCA to 3 components
weight_pca = PCA(n_components=3, svd_solver='randomized', random_state=42).fit_transform(weight_data)
rotate_pca = PCA(n_components=3, svd_solver='randomized', random_state=42).fit_transform(rotate_data)

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot weight matrices
for i, point in enumerate(weight_pca):
    ax.scatter(point[0], point[1], point[2],
               c=[layer_colors[layer_labels['weight'][i]]],
               marker=markers['weight'], s=60,
               label=f"Layer {layer_labels['weight'][i]}" if i==0 else "")

# Plot rotate_layer matrices
for i, point in enumerate(rotate_pca):
    ax.scatter(point[0], point[1], point[2],
               c=[layer_colors[layer_labels['rotate_layer'][i]]],
               marker=markers['rotate_layer'], s=60,
               label=f"Layer {layer_labels['rotate_layer'][i]}" if i==0 else "")

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA of Weight & Rotate_layer Matrices (Layers 4,6,10,12)")

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Weight', linestyle='None', markersize=8),
    Line2D([0], [0], marker='^', color='k', label='Rotate_layer', linestyle='None', markersize=8)
]
for layer in layers:
    legend_elements.append(Line2D([0], [0], marker='o', color=layer_colors[layer], label=f'Layer {layer}', linestyle='None', markersize=6))

ax.legend(handles=legend_elements, loc='best')
plt.tight_layout()

output_file = os.path.join(base_dir, "pca_layers_4_6_10_12.png")
plt.savefig(output_file)
plt.close()
print(f"✅ Saved PCA plot to {output_file}")
