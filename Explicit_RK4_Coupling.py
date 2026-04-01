import pandas as pd
import matplotlib.pyplot as plt

# Apply academic paper styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1.0

def plot_explicit_solution(density_file, efield_file, dx=20e-6):
    try:
        df_den = pd.read_csv(density_file)
        df_E = pd.read_csv(efield_file)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure the C++ code has been executed first.")
        return

    # 1. Extract C++ Explicit Simulation Data
    x_mm_E = df_E['cell'].values * dx * 1000
    E_sim_MVm = df_E['electric_field'].values / 1e6  

    x_mm_den = df_den['cell'].values * dx * 1000
    n_e_sim = df_den['electron_density'].values

    # 2. Create Figure with 2 Subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    
    # ---------------------------------------------------------
    # TOP PLOT: Electric Field
    # ---------------------------------------------------------
    # Explicit RK4 Solution (black dotted line to match the paper)
    ax1.plot(x_mm_E, E_sim_MVm, color='black', linestyle=':', linewidth=1, label='solution (explicit)')
    
    ax1.set_ylabel(r'$E$ (MV/m)', fontsize=14)
    ax1.set_xlabel(r'$x$ (mm)', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12, top=True, right=True)
    
    # Axis limits matching the paper
    ax1.set_xlim(1, 7)
    ax1.set_ylim(-2, 3)
    ax1.legend(frameon=False, fontsize=12, loc='upper right')

    # ---------------------------------------------------------
    # BOTTOM PLOT: Electron Density
    # ---------------------------------------------------------
    # Explicit RK4 Solution
    ax2.plot(x_mm_den, n_e_sim, color='black', linestyle=':', linewidth=1, label='solution (explicit)')
    
    ax2.set_yscale('log')
    ax2.set_ylabel(r'$n_e$ (m$^{-3}$)', fontsize=14)
    ax2.set_xlabel(r'$x$ (mm)', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12, top=True, right=True)
    
    # Axis limits matching the paper
    ax2.set_xlim(1, 7)
    ax2.set_ylim(1e12, 1e21)
    ax2.legend(frameon=False, fontsize=12, loc='lower right')

    # Adjust layout and save
    plt.tight_layout()
    output_filename = 'explicit_solution.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved as: {output_filename}")
    plt.show()

# Run the plotter
plot_explicit_solution('density_data.csv', 'efield_data.csv', dx=20e-6)
