import pandas as pd
import matplotlib.pyplot as plt

# Apply academic paper styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1.0

def plot_full_comparison(density_semi_file, efield_semi_file, density_exp_file, efield_exp_file, dx=20e-6):
    try:
        # Load Semi-Implicit Data
        df_den_semi = pd.read_csv(density_semi_file)
        df_E_semi = pd.read_csv(efield_semi_file)
        
        # Load Explicit (Reference Solution) Data
        df_den_exp = pd.read_csv(density_exp_file)
        df_E_exp = pd.read_csv(efield_exp_file)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure both C++ codes have been executed first.")
        return

    # Convert coordinates and units for Semi-Implicit
    x_mm_E_semi = df_E_semi['cell'].values * dx * 1000
    E_semi_MVm = df_E_semi['electric_field'].values / 1e6  
    x_mm_den_semi = df_den_semi['cell'].values * dx * 1000
    n_e_semi = df_den_semi['electron_density'].values

    # Convert coordinates and units for Explicit
    x_mm_E_exp = df_E_exp['cell'].values * dx * 1000
    E_exp_MVm = df_E_exp['electric_field'].values / 1e6  
    x_mm_den_exp = df_den_exp['cell'].values * dx * 1000
    n_e_exp = df_den_exp['electron_density'].values

    # Create Figure with 2 Subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    
    # ---------------------------------------------------------
    # TOP PLOT: Electric Field
    # ---------------------------------------------------------
    # 1. Semi-Implicit (Purple Solid Line)
    ax1.plot(x_mm_E_semi, E_semi_MVm, color='mediumorchid', linestyle='-', linewidth=0.7, label='semi-impl. (Euler)')
    
    # 2. Explicit RK4 (Black Dotted Line)
    ax1.plot(x_mm_E_exp, E_exp_MVm, color='black', linestyle=':', linewidth=1, label='solution')
    
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
    # 1. Semi-Implicit (Purple Solid Line)
    ax2.plot(x_mm_den_semi, n_e_semi, color='mediumorchid', linestyle='-', linewidth=0.7, label='semi-impl. (Euler)')
    
    # 2. Explicit RK4 (Black Dotted Line)
    ax2.plot(x_mm_den_exp, n_e_exp, color='black', linestyle=':', linewidth=1, label='solution')
    
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
    output_filename = 'semi_implicit_euler.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved as: {output_filename}")
    plt.show()

# Run the plotter with the 4 files
plot_full_comparison(
    density_semi_file='density_data_semi.csv', 
    efield_semi_file='efield_data_semi.csv', 
    density_exp_file='density_data.csv', 
    efield_exp_file='efield_data.csv', 
    dx=20e-6
)
