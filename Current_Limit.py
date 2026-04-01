import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# Academic Plot Formatting
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['font.size'] = 12

# Grid spacing (20 micrometers)
dx = 20e-6 

def load_data(filename):
    """Helper function to load CSV safely."""
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        print(f"Warning: '{filename}' not found. Make sure you ran the C++ code.")
        return None

def interpolate_to_centers(E_df, num_cells):
    """
    Interpolates face-centered electric field data (N-1) 
    to cell-centered data (N) using an arithmetic mean.
    """
    if E_df is None:
        return None
        
    E_raw = E_df['electric_field'].values
    E_center = np.zeros(num_cells)
    
    # Average the two faces for all internal cells
    E_center[1:-1] = 0.5 * (E_raw[:-1] + E_raw[1:])
    
    # For the boundaries, just copy the adjacent face value
    E_center[0] = E_raw[0]
    E_center[-1] = E_raw[-1]
    
    return E_center

def plot_teunissen_figure_2():
    # 1. Load Density Data (Defined at Cell Centers)
    den_exp  = load_data('density_data.csv')             # Explicit RK4 (Reference)
    den_semi = load_data('density_data_semi_euler.csv')        # Semi-Implicit Euler
    den_clim = load_data('density_data_current_lim.csv') # Current-Limited

    # 2. Load Electric Field Data (Defined at Cell Faces)
    E_exp  = load_data('efield_data.csv')
    E_semi = load_data('efield_data_semi_euler.csv')
    E_clim = load_data('efield_data_current_lim.csv')

    if den_exp is None or E_exp is None:
        print("Error: Explicit reference data is missing. Cannot plot.")
        return

    # Total number of cells (N)
    N = len(den_exp)
    
    # Common x-axis for both density and electric field (Cell Centers)
    x_mm = np.arange(N) * dx * 1000 

    # 3. Interpolate Electric Fields to Cell Centers
    E_exp_center  = interpolate_to_centers(E_exp, N)
    E_semi_center = interpolate_to_centers(E_semi, N)
    E_clim_center = interpolate_to_centers(E_clim, N)

    # Create the figure with 2 subplots (stacked vertically)
    fig, axs = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    fig.subplots_adjust(hspace=0.1) # Close the gap between plots

    # ==========================================
    # Top Plot: Electric Field (E vs x)
    # ==========================================
    ax_E = axs[0]

    # Plot Semi-Implicit Euler (Purple solid line)
    if E_semi_center is not None:
        ax_E.plot(x_mm, E_semi_center / 1e6, color='#9467bd', linestyle='-', lw=1.2, label='semi-impl.')
        
    # Plot Current-Limited (Green solid line)
    if E_clim_center is not None:
        ax_E.plot(x_mm, E_clim_center / 1e6, color='#2ca02c', linestyle='-', lw=1.2, label='current-lim.')

    # Plot Explicit RK4 Reference (Black dotted line)
    ax_E.plot(x_mm, E_exp_center / 1e6, color='black', linestyle=':', lw=1.4, label='solution')

    ax_E.set_ylabel('E (MV/m)')
    ax_E.set_ylim(-2.0, 3.0)
    ax_E.legend(frameon=False, loc='upper right', fontsize=11)
    ax_E.tick_params(top=True, right=True)

    # ==========================================
    # Bottom Plot: Electron Density (n_e vs x)
    # ==========================================
    ax_n = axs[1]

    # Plot Semi-Implicit Euler
    if den_semi is not None:
        ax_n.plot(x_mm, den_semi['electron_density'], color='#9467bd', linestyle='-', lw=1.3, label='semi-impl.')

    # Plot Current-Limited
    if den_clim is not None:
        ax_n.plot(x_mm, den_clim['electron_density'], color='#2ca02c', linestyle='-', lw=1.3, label='current-lim.')

    # Plot Explicit RK4 Reference
    ax_n.plot(x_mm, den_exp['electron_density'], color='black', linestyle=':', lw=1.3, label='solution')

    ax_n.set_ylabel(r'$n_e$ (m$^{-3}$)')
    ax_n.set_xlabel('$x$ (mm)')
    ax_n.set_yscale('log')
    ax_n.set_ylim(1e12, 5e20)
    ax_n.set_xlim(1, 7) # Zoom in on the active plasma region
    
    # Clean up ticks for log scale
    ax_n.tick_params(top=True, right=True, which='both')
    ax_n.legend(frameon=False, loc='lower right', fontsize=11)

    # Save and show
    plt.tight_layout()
    #plt.savefig('teunissen_figure_2_replication.png', dpi=700, bbox_inches='tight')
    #print("Plot generated successfully: 'teunissen_figure_2_replication.png'")
    plt.show()

if __name__ == '__main__':
    plot_teunissen_figure_2()
