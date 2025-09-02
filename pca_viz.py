import torch
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Headless backend
plt.switch_backend("Agg")

base_dir = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/REFT-CL"
num_tasks = 8
layer_name = "layer_4"

weight_matrices = []
rotate_matrices = []

for task_nb in range(num_tasks):
    file_path = os.path.join(base_dir, str(task_nb), f"intkey_{layer_name}_comp_block_output_unit_pos_nunit_1#0.bin")
    if not os.path.exists(file_path):
        print(f"⚠️ File missing for task {task_nb}: {file_path}")
        continue

    state = torch.load(file_path, map_location="cpu")
    weight_key = f"tasks.{task_nb}.weight"
    rotate_key = f"tasks.{task_nb}.rotate_layer"

    if weight_key in state:
        weight_matrices.append(state[weight_key].float().flatten().numpy())
    else:
        print(f"⚠️ Weight not found for task {task_nb}")

    if rotate_key in state:
        rotate_matrices.append(state[rotate_key].float().flatten().numpy())
    else:
        print(f"⚠️ Rotate_layer not found for task {task_nb}")

# Stack into arrays
weight_data = np.stack(weight_matrices)
rotate_data = np.stack(rotate_matrices)

# PCA
pca_weight = PCA(n_components=3, svd_solver='randomized', random_state=42)
weight_pca = pca_weight.fit_transform(weight_data)

pca_rotate = PCA(n_components=3, svd_solver='randomized', random_state=42)
rotate_pca = pca_rotate.fit_transform(rotate_data)

# Plotting both in same figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot weight matrices
ax.scatter(weight_pca[:,0], weight_pca[:,1], weight_pca[:,2], c='blue', label='Weight', s=50)

# Plot rotate_layer matrices
ax.scatter(rotate_pca[:,0], rotate_pca[:,1], rotate_pca[:,2], c='red', label='Rotate_layer', s=50)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title(f"PCA of Weight & Rotate_layer Matrices for {layer_name}")
ax.legend()

plt.tight_layout()
output_file = os.path.join(base_dir, f"pca_weight_rotate_layer_final_task_rounds_{layer_name}.png")
plt.savefig(output_file)
plt.close()
print(f"✅ Saved combined PCA plot to {output_file}")
