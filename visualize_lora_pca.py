#!/usr/bin/env python3
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from safetensors import safe_open
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pathlib import Path

TASK_NAMES = {
    0: "C-STANCE",
    1: "FOMC", 
    2: "MeetingBank",
    3: "Py150",
    4: "ScienceQA",
    5: "NumGLUE-cm",
    6: "NumGLUE-ds",
    7: "20Minuten"
}

# Color palette for different tasks
COLORS = plt.cm.Set3(np.linspace(0, 1, 8))

def load_specific_lora_matrices(adapter_path, target_layers=[6, 12], matrix_type='q_proj'):
    """Load specific LoRA matrices from a single adapter for specific layers and matrix type."""
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    
    A_matrices = []
    B_matrices = []
    
    with safe_open(adapter_file, framework='pt') as f:
        for key in f.keys():
            # Check if this is the right layer and matrix type
            if any(f'layers.{layer}.self_attn.{matrix_type}' in key for layer in target_layers):
                tensor = f.get_tensor(key)
                
                if 'lora_A' in key:
                    A_matrices.append(tensor.float().flatten().numpy())
                elif 'lora_B' in key:
                    B_matrices.append(tensor.float().flatten().numpy())
    
    return A_matrices, B_matrices

def load_specific_adapters(base_path, target_layers=[6, 12], matrix_type='q_proj', num_tasks=8):
    """Load specific LoRA matrices from all adapters for specific layers and matrix type."""
    all_adapters = {}
    
    for task_id in range(num_tasks):
        adapter_path = os.path.join(base_path, str(task_id))
        if os.path.exists(adapter_path):
            print(f"Loading {matrix_type} matrices from layers {target_layers} for task {task_id} ({TASK_NAMES.get(task_id, f'Task-{task_id}')})...")
            A_matrices, B_matrices = load_specific_lora_matrices(adapter_path, target_layers, matrix_type)
            all_adapters[task_id] = (A_matrices, B_matrices)
            print(f"  Loaded {len(A_matrices)} A matrices and {len(B_matrices)} B matrices")
        else:
            print(f"Warning: Adapter path {adapter_path} not found")
    
    return all_adapters

def prepare_data_for_pca(all_adapters):
    all_matrices = []
    labels = []
    task_ids = []
    
    for task_id, (A_matrices, B_matrices) in all_adapters.items():
        # Combine A and B matrices for the current task
        task_matrices = A_matrices + B_matrices
        
        for i, matrix in enumerate(task_matrices):
            all_matrices.append(matrix)
            labels.append(f"{TASK_NAMES.get(task_id, f'Task-{task_id}')}_{'A' if i < len(A_matrices) else 'B'}")
            task_ids.append(task_id)
    
    return np.array(all_matrices), labels, task_ids

def create_focused_plots(data, labels, task_ids, target_layers, matrix_type, output_dir="lora_pca_plots"):
    """Create focused plots for specific layers and matrix types."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_scaled)
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")
    
    # Separate A and B matrices
    A_indices = [i for i, label in enumerate(labels) if 'A' in label]
    B_indices = [i for i, label in enumerate(labels) if 'B' in label]
    
    # Create 2x2 grid: A matrices (top row), B matrices (bottom row)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # A matrices - PC1 vs PC2
    for task_id in range(8):
        if task_id in task_ids:
            mask = np.array(task_ids) == task_id
            task_A_mask = mask & np.isin(range(len(task_ids)), A_indices)
            task_data = pca_result[task_A_mask]
            task_name = TASK_NAMES.get(task_id, f'T{task_id}')
            
            if len(task_data) > 0:
                axes[0, 0].scatter(task_data[:, 0], task_data[:, 1], 
                                  c=[COLORS[task_id]], label=task_name, alpha=0.7, s=60)
    
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0, 0].set_title(f'{matrix_type} A matrices - Layers {target_layers}')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    # A matrices - PC1 vs PC3
    for task_id in range(8):
        if task_id in task_ids:
            mask = np.array(task_ids) == task_id
            task_A_mask = mask & np.isin(range(len(task_ids)), A_indices)
            task_data = pca_result[task_A_mask]
            task_name = TASK_NAMES.get(task_id, f'T{task_id}')
            
            if len(task_data) > 0:
                axes[0, 1].scatter(task_data[:, 0], task_data[:, 2], 
                                  c=[COLORS[task_id]], label=task_name, alpha=0.7, s=60)
    
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0, 1].set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    axes[0, 1].set_title(f'{matrix_type} A matrices - Layers {target_layers}')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # B matrices - PC1 vs PC2
    for task_id in range(8):
        if task_id in task_ids:
            mask = np.array(task_ids) == task_id
            task_B_mask = mask & np.isin(range(len(task_ids)), B_indices)
            task_data = pca_result[task_B_mask]
            task_name = TASK_NAMES.get(task_id, f'T{task_id}')
            
            if len(task_data) > 0:
                axes[1, 0].scatter(task_data[:, 0], task_data[:, 1], 
                                  c=[COLORS[task_id]], label=task_name, alpha=0.7, s=60)
    
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[1, 0].set_title(f'{matrix_type} B matrices - Layers {target_layers}')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # B matrices - PC1 vs PC3
    for task_id in range(8):
        if task_id in task_ids:
            mask = np.array(task_ids) == task_id
            task_B_mask = mask & np.isin(range(len(task_ids)), B_indices)
            task_data = pca_result[task_B_mask]
            task_name = TASK_NAMES.get(task_id, f'T{task_id}')
            
            if len(task_data) > 0:
                axes[1, 1].scatter(task_data[:, 0], task_data[:, 2], 
                                  c=[COLORS[task_id]], label=task_name, alpha=0.7, s=60)
    
    axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[1, 1].set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    axes[1, 1].set_title(f'{matrix_type} B matrices - Layers {target_layers}')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{matrix_type}_layers_{"_".join(map(str, target_layers))}_pca.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    
    return pca, pca_result

def main():
    base_path = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/lora"
    output_dir = "/cluster/home/tdieudonne/clmm/TRACE/lora_pca_plots"
    
    # Focus on query matrices at layers 6 and 12
    target_layers = [6, 12]
    matrix_type = 'q_proj'
    
    print("⚠️  IMPORTANT: In continual learning, adapters are trained sequentially!")
    print("   - Task 0: Trained only on C-STANCE")
    print("   - Task 1: Trained on C-STANCE + FOMC (with forgetting)")
    print("   - Task 2: Trained on C-STANCE + FOMC + MeetingBank (more forgetting)")
    print("   - etc...")
    print("\nThis means each adapter contains different amounts of task knowledge.")
    print("Consider analyzing specific task pairs or early tasks only.\n")
    
    print(f"Loading {matrix_type} matrices from layers {target_layers}...")
    all_adapters = load_specific_adapters(base_path, target_layers, matrix_type)
    
    if not all_adapters:
        print("No adapters found! Please check the path.")
        return
    
    print(f"\nLoaded adapters for {len(all_adapters)} tasks")
    
    print("\nPreparing data for PCA...")
    data, labels, task_ids = prepare_data_for_pca(all_adapters)
    
    print(f"Total matrices: {len(data)}")
    print(f"Matrix dimension: {data.shape[1]}")
    print(f"Tasks represented: {set(task_ids)}")
    
    print(f"\nCreating focused PCA visualizations for {matrix_type} at layers {target_layers}...")
    pca, pca_result = create_focused_plots(data, labels, task_ids, target_layers, matrix_type, output_dir)
    
    print(f"\nVisualization complete! Plot saved to: {output_dir}")
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")
    
    print("\n💡 SUGGESTIONS for more meaningful analysis:")
    print("   - plot_lora_matrices([6, 12], 'q_proj')  # All tasks (current)")
    print("   - plot_early_tasks([6, 12], 'q_proj', max_task=3)  # First 3 tasks only")
    print("   - plot_task_pairs([6, 12], 'q_proj', [0, 1])  # Compare specific tasks")

def plot_lora_matrices(target_layers=[6, 12], matrix_type='q_proj'):
    """Simple function to plot LoRA matrices for specific layers and matrix type."""
    base_path = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/lora"
    output_dir = "/cluster/home/tdieudonne/clmm/TRACE/lora_pca_plots"
    
    print(f"Loading {matrix_type} matrices from layers {target_layers}...")
    all_adapters = load_specific_adapters(base_path, target_layers, matrix_type)
    
    if not all_adapters:
        print("No adapters found! Please check the path.")
        return
    
    data, labels, task_ids = prepare_data_for_pca(all_adapters)
    print(f"Total matrices: {len(data)}")
    
    pca, pca_result = create_focused_plots(data, labels, task_ids, target_layers, matrix_type, output_dir)
    print(f"Plot saved! PCA explained variance: {sum(pca.explained_variance_ratio_):.3f}")

def plot_early_tasks(target_layers=[6, 12], matrix_type='q_proj', max_task=3):
    """Plot only the first few tasks to avoid catastrophic forgetting effects."""
    base_path = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/lora"
    output_dir = "/cluster/home/tdieudonne/clmm/TRACE/lora_pca_plots"
    
    print(f"Loading {matrix_type} matrices from layers {target_layers} for tasks 0-{max_task}...")
    all_adapters = {}
    
    for task_id in range(max_task + 1):
        adapter_path = os.path.join(base_path, str(task_id))
        if os.path.exists(adapter_path):
            print(f"Loading task {task_id} ({TASK_NAMES.get(task_id, f'Task-{task_id}')})...")
            A_matrices, B_matrices = load_specific_lora_matrices(adapter_path, target_layers, matrix_type)
            all_adapters[task_id] = (A_matrices, B_matrices)
            print(f"  Loaded {len(A_matrices)} A matrices and {len(B_matrices)} B matrices")
        else:
            print(f"Warning: Adapter path {adapter_path} not found")
    
    if not all_adapters:
        print("No adapters found! Please check the path.")
        return
    
    data, labels, task_ids = prepare_data_for_pca(all_adapters)
    print(f"Total matrices: {len(data)}")
    
    pca, pca_result = create_focused_plots(data, labels, task_ids, target_layers, f"{matrix_type}_early_{max_task}", output_dir)
    print(f"Plot saved! PCA explained variance: {sum(pca.explained_variance_ratio_):.3f}")

def plot_task_pairs(target_layers=[6, 12], matrix_type='q_proj', task_pairs=[(0, 1), (0, 2)]):
    """Compare specific task pairs to analyze forgetting patterns."""
    base_path = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/lora"
    output_dir = "/cluster/home/tdieudonne/clmm/TRACE/lora_pca_plots"
    
    for task1, task2 in task_pairs:
        print(f"\nComparing tasks {task1} and {task2}...")
        all_adapters = {}
        
        for task_id in [task1, task2]:
            adapter_path = os.path.join(base_path, str(task_id))
            if os.path.exists(adapter_path):
                A_matrices, B_matrices = load_specific_lora_matrices(adapter_path, target_layers, matrix_type)
                all_adapters[task_id] = (A_matrices, B_matrices)
            else:
                print(f"Warning: Adapter path {adapter_path} not found")
        
        if len(all_adapters) == 2:
            data, labels, task_ids = prepare_data_for_pca(all_adapters)
            pca, pca_result = create_focused_plots(data, labels, task_ids, target_layers, 
                                                 f"{matrix_type}_pair_{task1}_{task2}", output_dir)
            print(f"Pair {task1}-{task2} plot saved! PCA explained variance: {sum(pca.explained_variance_ratio_):.3f}")
        else:
            print(f"Could not load both tasks {task1} and {task2}")

if __name__ == "__main__":
    # Default: query matrices at layers 6 and 12
    main()
    
    # You can also easily create other plots:
    # plot_lora_matrices([6, 12], 'v_proj')  # Value matrices
    # plot_lora_matrices([0, 15, 31], 'q_proj')  # Different layers
    # plot_lora_matrices([6], 'q_proj')  # Single layer
