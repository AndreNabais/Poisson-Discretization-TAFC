#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm> 

using namespace std;

// =========================================================================
// GLOBAL CONSTANTS (Based on Teunissen 2020 - Test Case 3.4)
// =========================================================================
const double epsilon = 8.854187817e-12;  // Vacuum permittivity (F/m)
const double e_charge = 1.602176634e-19; // Elementary charge (C)

// Grid Definition: 1D domain of 10 mm discretized into 500 cells
const int n = 500;                       
const double dx = 20e-6;                 // Cell width: 20 micrometers

// Explicit Time Step
// Note: 80 ps heavily violates the dielectric relaxation time (tau) for this density.
// However, it remains absolutely stable here due to the Current-Limiting approach.
const double dt = 80e-12;                

class CurrentLimitedSolver {
public:
    // Physical state arrays
    // n_e, n_p are stored at cell centers. Gamma (flux) is stored at cell faces.
    double x_pos[n], n_e[n], n_p[n], Gamma[n];
    
    // Arrays for the Trapezoidal Predictor-Corrector time integration
    double n_e_temp[n]; // Temporary predicted density state (n_e*)
    double k1[n];       // Density rate of change at current time t
    double k2[n];       // Density rate of change at predicted time t+dt

    // Arrays for the Thomas Algorithm to solve the Poisson equation
    double a[n], b[n], c[n], d[n]; 
    double solution[n];      // Electrostatic potential (V) at cell centers
    double ElectricField[n]; // Electric field (E) at cell faces

    // Transport coefficients (assumed constant for this specific test case)
    double mu = 0.03; // Electron mobility (m^2/Vs)
    double D_e = 0.1; // Electron diffusion coefficient (m^2/s)

    CurrentLimitedSolver() {
        // Initialize arrays to zero
        for (int i = 0; i < n; i++) {
            x_pos[i] = i * dx;
            n_e[i] = 0.0;
            n_p[i] = 0.0;
            Gamma[i] = 0.0;
            n_e_temp[i] = 0.0;
            k1[i] = 0.0;
            k2[i] = 0.0;
        }

        // Apply Initial Condition: 
        // A stationary block of plasma (10^20 m^-3) between x = 4 mm and 6 mm
        for(int i = 0; i < n; i++) {
            if (x_pos[i] >= 4.0e-3 && x_pos[i] <= 6.0e-3) {
                n_e[i] = 1e20;
                n_p[i] = 1e20; // Assuming quasi-neutrality at t=0
            }
        }
    }

    // The Koren slope limiter function (Total Variation Diminishing)
    // Prevents non-physical numerical oscillations near sharp density gradients.
    double koren(double r) { 
        return max(0.0, min({1.0, (2.0 + r) / 6.0, r})); 
    }

    // =========================================================================
    // STANDARD POISSON SOLVER
    // Solves the constant-coefficient elliptic PDE: -\nabla^2 V = \rho / epsilon_0
    // =========================================================================
    void solve_standard_poisson(const double* n_e_in, double V_left, double V_right) {
        
        // 1. Setup tridiagonal matrix coefficients for the 1D Laplacian
        double eps_over_dx2 = epsilon / (dx * dx);
        for (int i = 1; i < n - 1; i++) {
            a[i] = -eps_over_dx2;                  // Lower diagonal
            c[i] = -eps_over_dx2;                  // Upper diagonal
            b[i] = 2.0 * eps_over_dx2;             // Main diagonal
            d[i] = e_charge * (n_p[i] - n_e_in[i]);// Right-Hand Side (Space charge: rho)
        }

        // 2. Apply Dirichlet Boundary Conditions at the edges (Cell 0 and n-1)
        // Note: The physical boundary wall is assumed to be at dx/2 from the cell center
        c[0] = -eps_over_dx2;
        a[0] = 0.0;
        b[0] = 3.0 * eps_over_dx2; 
        d[0] = e_charge * (n_p[0] - n_e_in[0]) + (2.0 * eps_over_dx2) * V_left;

        a[n-1] = -eps_over_dx2;
        c[n-1] = 0.0;
        b[n-1] = 3.0 * eps_over_dx2;
        d[n-1] = e_charge * (n_p[n-1] - n_e_in[n-1]) + (2.0 * eps_over_dx2) * V_right;

        // 3. Solve the linear system using the Thomas Algorithm (O(N) complexity)
        // Forward sweep to eliminate lower diagonal
        for (int i = 1; i < n; i++) {
            double m = a[i] / b[i-1];
            b[i] -= m * c[i-1];
            d[i] -= m * d[i-1];
        }
        // Back substitution to find the potential
        solution[n-1] = d[n-1] / b[n-1];
        for (int i = n - 2; i >= 0; i--) {
            solution[i] = (d[i] - c[i] * solution[i+1]) / b[i];
        }

        // 4. Extract the Electric Field at cell faces (E = -\nabla V)
        for (int i = 0; i < n - 1; i++) {
            ElectricField[i] = -(solution[i+1] - solution[i]) / dx;
        }
    }

    // =========================================================================
    // RATE EVALUATION & CURRENT LIMITER
    // Computes the temporal derivative (dn/dt = -\nabla \cdot \Gamma) 
    // =========================================================================
    void compute_rates(const double* n_e_in, double* k_out) {
        // Step 1: Obtain the true electric field for the provided density state
        // 10 kV applied on the left boundary, 0 V on the right.
        solve_standard_poisson(n_e_in, 10000.0, 0.0); 

        double eps_div = 1e-15; // Small constant to prevent division by zero

        // Step 2: Compute fluxes at the internal cell faces
        for (int i = 1; i < n - 2; i++) {
            double E_face = ElectricField[i];
            double v = -mu * E_face; // Drift velocity
            
            // A. Advective Flux (Drift) using the Koren upwind scheme
            double gamma_drift = 0.0;
            if (v >= 0.0) {
                // Flow is to the right
                double r_i = (n_e_in[i] - n_e_in[i-1]) / (n_e_in[i+1] - n_e_in[i] + eps_div);
                gamma_drift = v * (n_e_in[i] + koren(r_i) * (n_e_in[i+1] - n_e_in[i]));
            } else {
                // Flow is to the left
                double inv_r = (n_e_in[i+2] - n_e_in[i+1]) / (n_e_in[i+1] - n_e_in[i] + eps_div);
                gamma_drift = v * (n_e_in[i+1] - koren(inv_r) * (n_e_in[i+1] - n_e_in[i]));
            }

            // B. Diffusive Flux using 2nd-order central difference (Fick's Law)
            double gamma_diff = -D_e * (n_e_in[i+1] - n_e_in[i]) / dx;
            
            // C. Raw Total Flux
            double Gamma_raw = gamma_drift + gamma_diff;

            // =========================================================
            // CURRENT-LIMITED FLUX CLIPPING (Teunissen Sec. 3.3)
            // Mathematically prevents dielectric relaxation instability
            // =========================================================
            
            // Compute the relative density gradient robustly (Eq 20)
            double n_diff_abs = abs(n_e_in[i+1] - n_e_in[i]) / dx;
            double n_max = max({n_e_in[i], n_e_in[i+1], 1e-15});
            double grad_n_over_n = n_diff_abs / n_max;

            // Compute E*: The critical field that balances drift & diffusion (Eq 19)
            double E_star = max(abs(E_face), (D_e * grad_n_over_n) / mu);

            // Compute the strict maximum allowable flux (Gamma_max) (Eq 18)
            double max_allowed_flux = (epsilon * E_star) / (e_charge * dt);

            // Apply the piecewise limit logic (Eq 53)
            // If raw flux is too fast, clip its magnitude but preserve its sign
            if (abs(Gamma_raw) > max_allowed_flux) {
                Gamma[i] = (Gamma_raw > 0.0 ? 1.0 : -1.0) * max_allowed_flux;
            } else {
                Gamma[i] = Gamma_raw;
            }
            // =========================================================
        }

        // Step 3: Calculate the discrete divergence of the clipped flux
        // This yields the rate of change dn/dt for the control volume
        for (int i = 2; i < n - 2; i++) {
            k_out[i] = -(Gamma[i] - Gamma[i-1]) / dx;
        }
    }

    // =========================================================================
    // TIME INTEGRATION (2-Stage Predictor-Corrector)
    // Yields O(dt^2) temporal accuracy using the Explicit Trapezoidal Rule
    // =========================================================================
    void trapezoidal_explicit_step() {
        // --- PHASE 1: PREDICTOR (Euler Step) ---
        // Evaluate the rate of change at the current state (t)
        compute_rates(n_e, k1);

        // Project a temporary future density state (t + dt)
        for (int i = 0; i < n; i++) {
            n_e_temp[i] = n_e[i] + dt * k1[i]; 
        }

        // --- PHASE 2: CORRECTOR (Evaluate Future) ---
        // Evaluate the rate of change based on the predicted future state
        // This inherently calls the Poisson solver on n_e_temp
        compute_rates(n_e_temp, k2);

        // --- PHASE 3: FINAL TRAPEZOIDAL UPDATE ---
        // Average the two slopes (k1 and k2) to take the final explicit step
        for (int i = 2; i < n - 2; i++) {
            n_e[i] += (dt / 2.0) * (k1[i] + k2[i]);
        }
    }
};

// =========================================================================
// MAIN SIMULATION LOOP
// =========================================================================
int main() {
    CurrentLimitedSolver plasma;

    double target_time = 50e-9; // End simulation at 50 ns
    int total_steps = target_time / dt;

    cout << "Starting Current-Limited Explicit Trapezoidal Integration..." << endl;
    
    // Time marching loop
    for (int t = 0; t < total_steps; t++) {
        plasma.trapezoidal_explicit_step();
        
        // Print progress every 10%
        if (t % (total_steps / 10) == 0) {
            cout << "Progress: " << (t * 100) / total_steps << "%" << endl;
        }
    }

    // 1. Export the final electron density data (Cell Centers)
    ofstream denFile("density_data_current_lim.csv");
    denFile << "cell,electron_density\n";
    for (int i = 0; i < n; i++) denFile << i << "," << plasma.n_e[i] << "\n";
    denFile.close();

    // 2. Export the final Electric Field data (Cell Faces)
    // Critical Step: Because the final trapezoidal step updated n_e, the 
    // ElectricField array currently holds the field from a previous sub-step.
    // We must call the standard Poisson solver one last time to sync the 
    // observable electric field precisely to the final physical density.
    plasma.solve_standard_poisson(plasma.n_e, 10000.0, 0.0);
    
    ofstream eFile("efield_data_current_lim.csv");
    eFile << "cell,electric_field\n";
    for (int i = 0; i < n - 1; i++) eFile << i << "," << plasma.ElectricField[i] << "\n";
    eFile.close();

    cout << "Simulation complete. Data saved." << endl;
    return 0;
}
