import pandas as pd
import matplotlib.pyplot as plt

def plot_combined_laplace(potential_file, efield_file, output_name):
    try:
        df_pot = pd.read_csv(potential_file)
        df_eff = pd.read_csv(efield_file)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure the C++ code has been executed first.")
        return

    # Create a figure with 2 subplots sharing the same X-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # 1. Electric Potential Plot (Top)
    ax1.plot(df_pot['cell'], df_pot['potential'], color='blue', linewidth=2)
    ax1.set_title('Electric Potential Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Voltage (V)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. Electric Field Plot (Bottom)
    ax2.plot(df_eff['cell'], df_eff['electric_field'], color='red', linewidth=2)
    ax2.set_title('Electric Field Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Field (V/m)', fontsize=12)
    ax2.set_xlabel('Cell Number', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Clean up layout
    plt.tight_layout()
    plt.savefig(output_name, bbox_inches='tight', dpi=300)
    print(f"Combined plot saved as: {output_name}")
    plt.show()

# Run the function
plot_combined_laplace('potential_data.csv', 'efield_data.csv', 'combined_laplace_plots.png')
