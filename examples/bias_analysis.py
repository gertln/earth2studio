import torch
import numpy as np
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # avoid Qt plugin errors

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # headless backend for plots

data = torch.load("all_bias.pt", weights_only=False)

mbias = data["mbias"]
gbias = data["gbias"]
times = data["times"]
sigma_levels = data.get("sigma_levels", [2, 4, 6, 8])  # fallback for older files
var_indices = data.get("var_indices", [4])  # fallback for older files (only var 4)
mout = {sigma: data[f"mout{sigma}"] for sigma in sigma_levels}

# Load climatological reference data if available
x_era5_mean = data.get("x_era5_mean", None)
x_era5_std = data.get("x_era5_std", None)

print(f"Data shapes:")
print(f"  mbias: {mbias.shape}")
print(f"  gbias: {gbias.shape}")
print(f"  times: {times.shape}")
print(f"  sigma_levels: {sigma_levels}")
print(f"  var_indices: {var_indices}")
for sigma in sigma_levels:
    print(f"  mout{sigma}: {mout[sigma].shape}")

if x_era5_mean is not None:
    print(f"  x_era5_mean: {x_era5_mean.shape} (range: {x_era5_mean.min():.2f} to {x_era5_mean.max():.2f})")
if x_era5_std is not None:
    print(f"  x_era5_std: {x_era5_std.shape} (range: {x_era5_std.min():.2f} to {x_era5_std.max():.2f})")

# Convert lead times from steps to hours
lead_times_hours = np.arange(mbias.shape[0]) * 6  # 6-hour steps

# Create plots directory if it doesn't exist
import os
os.makedirs('plots', exist_ok=True)

# Variable names for plotting (you can customize these)
var_names = {0: "Variable_0", 1: "Variable_1", 4: "Temperature", 7: "Variable_7"}

# Create plots for first few variables (to avoid too many plots)
vars_to_plot = var_indices[:min(4, len(var_indices))]  # Plot first 4 variables max

# Process each variable separately
for v_idx, var_idx in enumerate(vars_to_plot):
    var_name = var_names.get(var_idx, f"Variable_{var_idx}")
    print(f"\nProcessing {var_name} (index {var_idx})...")
    
    # ---------------------------------------------------------------------------
    # Plot 1: Global bias evolution for this variable
    # ---------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Average global bias across all initial times for this variable
    gbias_mean = gbias[:, v_idx, :].mean(dim=1).numpy()
    gbias_std = gbias[:, v_idx, :].std(dim=1).numpy()
    
    plt.plot(lead_times_hours, gbias_mean, 'b-', linewidth=2, label='Mean global bias')
    plt.fill_between(lead_times_hours, 
                     gbias_mean - gbias_std, 
                     gbias_mean + gbias_std, 
                     alpha=0.3, color='blue', label='±1σ')
    
    plt.xlabel('Lead time (hours)')
    plt.ylabel('Global bias (K)')
    plt.title(f'Global {var_name} Bias Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/global_bias_evolution_{var_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: plots/global_bias_evolution_{var_name.lower()}.png")

# ---------------------------------------------------------------------------
# Plot 2: Spatial bias maps at different lead times (for first variable as example)
# ---------------------------------------------------------------------------
v_idx = 0  # Use first variable for spatial maps
var_name = var_names.get(vars_to_plot[v_idx], f"Variable_{vars_to_plot[v_idx]}")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
lead_times_to_plot = [0, 12, 24, 48, 96, 168]  # hours
lead_indices = [lt // 6 for lt in lead_times_to_plot]  # convert to step indices

for i, (lt_hours, lt_idx) in enumerate(zip(lead_times_to_plot, lead_indices)):
    if lt_idx >= mbias.shape[0]:
        continue
    
    ax = axes[i // 3, i % 3]
    bias_map = mbias[lt_idx, v_idx].numpy()
    
    # Center colorbar at 0 for better bias visualization
    vmax = max(abs(bias_map.min()), abs(bias_map.max()))
    im = ax.imshow(bias_map, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_title(f'Lead time: {lt_hours}h')
    ax.set_xlabel('Longitude index')
    ax.set_ylabel('Latitude index')
    plt.colorbar(im, ax=ax, label='Bias (K)')

plt.suptitle(f'Spatial {var_name} Bias Maps')
plt.tight_layout()
plt.savefig(f'plots/spatial_bias_maps_{var_name.lower()}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: plots/spatial_bias_maps_{var_name.lower()}.png")

# ---------------------------------------------------------------------------
# Plot 3: Outlier frequency evolution
# ---------------------------------------------------------------------------
# Create subplot for each variable (limit to 4 for readability)
fig, axes = plt.subplots(len(vars_to_plot), 1, figsize=(12, 6 * len(vars_to_plot)))
if len(vars_to_plot) == 1:
    axes = [axes]  # Make it a list for consistency

colors = ['green', 'red', 'orange', 'purple']

for v_idx, var_idx in enumerate(vars_to_plot):
    var_name = var_names.get(var_idx, f"Variable_{var_idx}")
    ax = axes[v_idx]
    
    # Calculate mean outlier fraction over time for this variable
    outlier_means = {}
    for i, sigma in enumerate(sigma_levels):
        outlier_means[sigma] = (mout[sigma][:, v_idx].mean(dim=[1, 2]) * 100).numpy()  # convert to percentage
        color = colors[i % len(colors)]  # cycle through colors if more sigma levels than colors
        ax.plot(lead_times_hours, outlier_means[sigma], 
                color=color, linewidth=2, label=f'±{sigma}σ outliers')
    
    ax.set_xlabel('Lead time (hours)')
    ax.set_ylabel('Outlier fraction (%)')
    ax.set_title(f'{var_name}: Fraction of Grid Points Beyond Climatological Thresholds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

plt.tight_layout()
plt.savefig('plots/outlier_evolution_selected_vars.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/outlier_evolution_selected_vars.png")

# ---------------------------------------------------------------------------
# Plot 4: Bias statistics summary
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(vars_to_plot), 2, figsize=(14, 5 * len(vars_to_plot)))
if len(vars_to_plot) == 1:
    axes = axes.reshape(1, -1)  # Make it 2D for consistency

for v_idx, var_idx in enumerate(vars_to_plot):
    var_name = var_names.get(var_idx, f"Variable_{var_idx}")
    
    # Left plot: Bias magnitude evolution
    bias_magnitude = torch.abs(mbias[:, v_idx]).mean(dim=[1, 2]).numpy()
    axes[v_idx, 0].plot(lead_times_hours, bias_magnitude, 'b-', linewidth=2)
    axes[v_idx, 0].set_xlabel('Lead time (hours)')
    axes[v_idx, 0].set_ylabel('Mean absolute bias (K)')
    axes[v_idx, 0].set_title(f'{var_name}: Bias Magnitude Evolution')
    axes[v_idx, 0].grid(True, alpha=0.3)
    
    # Right plot: Bias variance evolution  
    bias_variance = mbias[:, v_idx].var(dim=[1, 2]).numpy()
    axes[v_idx, 1].plot(lead_times_hours, bias_variance, 'r-', linewidth=2)
    axes[v_idx, 1].set_xlabel('Lead time (hours)')
    axes[v_idx, 1].set_ylabel('Bias variance (K²)')
    axes[v_idx, 1].set_title(f'{var_name}: Spatial Bias Variance Evolution')
    axes[v_idx, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/bias_statistics_selected_vars.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/bias_statistics_selected_vars.png")

print("\nAll plots saved successfully!")
print("Generated files:")
for v_idx, var_idx in enumerate(vars_to_plot):
    var_name = var_names.get(var_idx, f"Variable_{var_idx}")
    print(f"  - plots/global_bias_evolution_{var_name.lower()}.png")
    if v_idx == 0:  # Only first variable gets spatial maps
        print(f"  - plots/spatial_bias_maps_{var_name.lower()}.png")
print("  - plots/outlier_evolution_selected_vars.png")
print("  - plots/bias_statistics_selected_vars.png")