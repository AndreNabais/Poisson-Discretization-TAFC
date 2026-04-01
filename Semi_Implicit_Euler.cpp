#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm> 

using namespace std;

// ==========================================
// GLOBAL CONSTANTS (Teunissen Test Case 3.4)
// ==========================================
const double epsilon = 8.854187817e-12;  
const double e_charge = 1.602176634e-19; 
const int n = 500;                       // 10 mm / 20 um = 500 cells
const double dx = 20e-6;                 // 20 um
const double dt = 80e-12;                // 80 ps time step for semi-implicit!

class SemiImplicitSolver {
public:
    double x_pos[n], n_e[n], n_p[n], Gamma[n];
    double eps_eff[n]; // Effective permittivity at faces
    
    // Thomas Algorithm Arrays
    double a[n], b[n], c[n], d[n], solution[n], ElectricField[n];

    double mu = 0.03; // Constant mobility
    double D_e = 0.1; // Constant diffusion

    SemiImplicitSolver() {
        for (int i = 0; i < n; i++) {
            x_pos[i] = i * dx;
            n_e[i] = 0.0;
            n_p[i] = 0.0;
            Gamma[i] = 0.0;
        }

        // Initial condition: 10^20 m^-3 between 4 mm and 6 mm
        for(int i = 0; i < n; i++) {
            if (x_pos[i] >= 4.0e-3 && x_pos[i] <= 6.0e-3) {
                n_e[i] = 1e20;
                n_p[i] = 1e20;
            }
        }
    }

    double koren(double r) { 
        return max(0.0, min({1.0, (2.0 + r) / 6.0, r})); 
    }

    void solve_modified_poisson(double V_left, double V_right) {
        // 1. Calculate effective permittivity at cell faces (i+1/2)
        for (int i = 0; i < n - 1; i++) {
            double n_e_face = 0.5 * (n_e[i] + n_e[i+1]);
            eps_eff[i] = epsilon + e_charge * dt * mu * n_e_face;
        }

        // 2. Setup Thomas Algorithm diagonals for the interior cells
        for (int i = 1; i < n - 1; i++) {
            a[i] = -eps_eff[i-1] / (dx * dx);
            c[i] = -eps_eff[i]   / (dx * dx);
            b[i] = -(a[i] + c[i]);

            // Divergence of diffusion: D * d^2(n_e)/dx^2
            double diff_div = D_e * (n_e[i+1] - 2.0 * n_e[i] + n_e[i-1]) / (dx * dx);
            
            // RHS = rho - dt * div(J_diff)
            d[i] = e_charge * (n_p[i] - n_e[i]) - e_charge * dt * diff_div;
        }

        // 3. Boundary Conditions (Dirichlet)
        // Left boundary (Cell 0)
        c[0] = -eps_eff[0] / (dx * dx);
        a[0] = 0.0;
        b[0] = -c[0] + 2.0 * epsilon / (dx * dx); // Wall is at dx/2
        double diff_div_0 = D_e * (n_e[1] - 2.0 * n_e[0] + 0.0) / (dx * dx);
        d[0] = e_charge * (n_p[0] - n_e[0]) - e_charge * dt * diff_div_0 + (2.0 * epsilon / (dx * dx)) * V_left;

        // Right boundary (Cell n-1)
        a[n-1] = -eps_eff[n-2] / (dx * dx);
        c[n-1] = 0.0;
        b[n-1] = -a[n-1] + 2.0 * epsilon / (dx * dx);
        double diff_div_n = D_e * (0.0 - 2.0 * n_e[n-1] + n_e[n-2]) / (dx * dx);
        d[n-1] = e_charge * (n_p[n-1] - n_e[n-1]) - e_charge * dt * diff_div_n + (2.0 * epsilon / (dx * dx)) * V_right;

        // 4. Solve Thomas Algorithm
        for (int i = 1; i < n; i++) {
            double m = a[i] / b[i-1];
            b[i] -= m * c[i-1];
            d[i] -= m * d[i-1];
        }
        solution[n-1] = d[n-1] / b[n-1];
        for (int i = n - 2; i >= 0; i--) {
            solution[i] = (d[i] - c[i] * solution[i+1]) / b[i];
        }

        // 5. Extract predicted Electric Field
        for (int i = 0; i < n - 1; i++) {
            ElectricField[i] = -(solution[i+1] - solution[i]) / dx;
        }
    }

    // =========================================================================
    // >>>>>>>>>>>>>>>>>>>>>>>>>>> FIRST CHANGE  START <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // added this  function. It solves the true physics without eps_eff at the end of the loop.
    // =========================================================================
    void solve_standard_poisson(double V_left, double V_right) {
        double eps_over_dx2 = epsilon / (dx * dx);
        
        for (int i = 1; i < n - 1; i++) {
            a[i] = -eps_over_dx2;
            c[i] = -eps_over_dx2;
            b[i] = 2.0 * eps_over_dx2;
            d[i] = e_charge * (n_p[i] - n_e[i]); 
        }

        // Boundaries
        c[0] = -eps_over_dx2;
        a[0] = 0.0;
        b[0] = 3.0 * eps_over_dx2; 
        d[0] = e_charge * (n_p[0] - n_e[0]) + (2.0 * eps_over_dx2) * V_left;

        a[n-1] = -eps_over_dx2;
        c[n-1] = 0.0;
        b[n-1] = 3.0 * eps_over_dx2;
        d[n-1] = e_charge * (n_p[n-1] - n_e[n-1]) + (2.0 * eps_over_dx2) * V_right;

        for (int i = 1; i < n; i++) {
            double m = a[i] / b[i-1];
            b[i] -= m * c[i-1];
            d[i] -= m * d[i-1];
        }
        solution[n-1] = d[n-1] / b[n-1];
        for (int i = n - 2; i >= 0; i--) {
            solution[i] = (d[i] - c[i] * solution[i+1]) / b[i];
        }
        for (int i = 0; i < n - 1; i++) {
            ElectricField[i] = -(solution[i+1] - solution[i]) / dx;
        }
    }
    // =========================================================================
    // >>>>>>>>>>>>>>>>>>>>>>>>>>>> FIRST CHANGE  END <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // =========================================================================

    void euler_transport_step() {
        // Compute explicit flux using the newly predicted electric field
        double eps_div = 1e-15; 

        for (int i = 1; i < n - 2; i++) {
            double E_face = ElectricField[i];
            double v = -mu * E_face;
            double gamma_drift = 0.0;

            if (v >= 0.0) {
                double r_i = (n_e[i] - n_e[i-1]) / (n_e[i+1] - n_e[i] + eps_div);
                gamma_drift = v * (n_e[i] + koren(r_i) * (n_e[i+1] - n_e[i]));
            } else {
                double inv_r = (n_e[i+2] - n_e[i+1]) / (n_e[i+1] - n_e[i] + eps_div);
                gamma_drift = v * (n_e[i+1] - koren(inv_r) * (n_e[i+1] - n_e[i]));
            }

            // Total Flux = Drift + Diffusion
            Gamma[i] = gamma_drift - D_e * (n_e[i+1] - n_e[i]) / dx;
        }

        // Apply first-order explicit Euler update
        for (int i = 2; i < n - 2; i++) {
            n_e[i] += dt * (-(Gamma[i] - Gamma[i-1]) / dx);
        }
    }

    void step() {
        solve_modified_poisson(10000.0, 0.0); // 10 kV applied on the left
        euler_transport_step();
    }
};

int main() {
    SemiImplicitSolver plasma;

    double target_time = 50e-9; // 50 ns
    int total_steps = target_time / dt;

    cout << "Starting Semi-Implicit Time Integration for " << total_steps << " steps..." << endl;
    
    for (int t = 0; t < total_steps; t++) {
        plasma.step();
        if (t % (total_steps / 10) == 0) cout << "Progress: " << (t * 100) / total_steps << "%" << endl;
    }

    // Export Data
    ofstream denFile("density_data_semi_euler.csv");
    denFile << "cell,electron_density\n";
    for (int i = 0; i < n; i++) denFile << i << "," << plasma.n_e[i] << "\n";
    denFile.close();

    // =========================================================================
    // >>>>>>>>>>>>>>>>>>>>>>>>>>> CHANGE 2 START <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // Replaced plasma.solve_modified_poisson(10000.0, 0.0);
    // With this line below to extract the true final field:
    // =========================================================================
    plasma.solve_standard_poisson(10000.0, 0.0);
    // =========================================================================
    // >>>>>>>>>>>>>>>>>>>>>>>>>>>> CHANGE 2 END <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // =========================================================================

    ofstream eFile("efield_data_semi_euler.csv");
    eFile << "cell,electric_field\n";
    for (int i = 0; i < n - 1; i++) eFile << i << "," << plasma.ElectricField[i] << "\n";
    eFile.close();

    cout << "Simulation complete. Data saved." << endl;
    return 0;
}
