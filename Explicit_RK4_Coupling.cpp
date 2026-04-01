#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm> 

using namespace std;

// ==========================================
// FORWARD DECLARATIONS
// ==========================================
class Coefficients;
class ThomasAlgorithm;
class BoundaryConditions;

// ==========================================
// GLOBAL CONSTANTS (Teunissen Test Case 3.4)
// ==========================================
const double epsilon = 8.854187817e-12;  
const double e_charge = 1.602176634e-19; 
const int n = 500;                       // 10 mm / 20 um = 500 cells
const double dx = 20e-6;                 // 20 um
const double dt = 0.1e-12;               // 0.1 ps required for explicit RK4 stability

class Coefficients {
public:
    double x_position[n], Voltage[n], rho[n];
    double phi_west[n], phi_east[n], phi_center[n];

    Coefficients() {
        for (int i = 0; i < n; i++) {
            x_position[i] = i * dx;
            phi_west[i] = phi_east[i] = phi_center[i] = 0.0;
            Voltage[i] = rho[i] = 0.0; 
        }
        for (int i = 0; i < n - 1; i++) phi_east[i] = -epsilon / dx;
        for (int i = 1; i < n; i++) phi_west[i] = -epsilon / dx;
        for (int i = 1; i < n - 1; i++) phi_center[i] = -(phi_west[i] + phi_east[i]);
    }
};

class BoundaryConditions {
public:
    void apply_Dirichlet_left(Coefficients& coeffs, ThomasAlgorithm& thomas, double V_app);
    void apply_Dirichlet_right(Coefficients& coeffs, ThomasAlgorithm& thomas, double V_app);
};

class ThomasAlgorithm {
public:
    double a[n], b[n], c[n], d[n], solution[n], ElectricField[n]; 

    void ThomasAlgorithm_setup(Coefficients& coeffs, BoundaryConditions& bc, double V_left, double V_right) {
        for (int i = 1; i < n - 1; i++) {
            a[i] = coeffs.phi_west[i]; b[i] = coeffs.phi_center[i]; c[i] = coeffs.phi_east[i];
            d[i] = dx * coeffs.rho[i]; solution[i] = coeffs.Voltage[i];
        }
        bc.apply_Dirichlet_left(coeffs, *this, V_left);
        bc.apply_Dirichlet_right(coeffs, *this, V_right);
    }

    void ThomasAlgorithm_solver() {
        for (int i = 1; i < n; i++) {
            double m = a[i] / b[i-1];
            b[i] -= m * c[i-1]; d[i] -= m * d[i-1];
        }
        solution[n-1] = d[n-1] / b[n-1];
        for (int i = n - 2; i >= 0; i--) solution[i] = (d[i] - c[i] * solution[i+1]) / b[i];
        for (int i = 0; i < n - 1; i++) ElectricField[i] = -(solution[i+1] - solution[i]) / dx;
    }
};

inline void BoundaryConditions::apply_Dirichlet_left(Coefficients& coeffs, ThomasAlgorithm& thomas, double V_app) {
    double phi_west = -epsilon / (dx / 2.0);
    thomas.a[0] = 0.0; thomas.b[0] = -(phi_west + coeffs.phi_east[0]);
    thomas.c[0] = coeffs.phi_east[0]; thomas.d[0] = dx * coeffs.rho[0] - (phi_west * V_app);
}

inline void BoundaryConditions::apply_Dirichlet_right(Coefficients& coeffs, ThomasAlgorithm& thomas, double V_app) {
    double phi_east = -epsilon / (dx / 2.0);
    thomas.a[n-1] = coeffs.phi_west[n-1]; thomas.b[n-1] = -(coeffs.phi_west[n-1] + phi_east);
    thomas.c[n-1] = 0.0; thomas.d[n-1] = dx * coeffs.rho[n-1] - (phi_east * V_app);
}

class DD_LFA_Solver {
public:
    vector<double> n_e, n_p, n_temp, Gamma, k1, k2, k3, k4;
    Coefficients& coeffs; ThomasAlgorithm& thomas; BoundaryConditions& bc;

    DD_LFA_Solver(Coefficients& c, ThomasAlgorithm& t, BoundaryConditions& b) 
        : coeffs(c), thomas(t), bc(b) {
        n_e.assign(n, 0.0); n_p.assign(n, 0.0); n_temp.assign(n, 0.0); Gamma.assign(n, 0.0);
        k1.assign(n, 0.0); k2.assign(n, 0.0); k3.assign(n, 0.0); k4.assign(n, 0.0);

        // Initial condition: 10^20 m^-3 between 4 mm and 6 mm
        for(int i = 0; i < n; i++) {
            double x = i * dx;
            if (x >= 4.0e-3 && x <= 6.0e-3) {
                n_e[i] = 1e20;
                n_p[i] = 1e20;
            }
        }
    }

    double koren(double r) {
        return max(0.0, min({1.0, (2.0 + r) / 6.0, r}));
    }

    void evaluate_rate(const vector<double>& current_n, vector<double>& k_out) {
        for (int i = 0; i < n; i++) coeffs.rho[i] = (n_p[i] - current_n[i]) * e_charge;
        
        // 10 kV applied on the left
        thomas.ThomasAlgorithm_setup(coeffs, bc, 10000.0, 0.0);
        thomas.ThomasAlgorithm_solver();

        double eps_div = 1e-15; 
        double mu = 0.03; // Constant mobility
        double D = 0.1;   // Constant diffusion

        for (int i = 1; i < n - 2; i++) {
            double E_face = thomas.ElectricField[i];
            double v = -mu * E_face;
            double gamma_drift = 0.0;

            //flow to the right (v < 0) or to the left (v >= 0) determines the upwind direction for the drift term
            if (v >= 0.0) {
                double r_i = (current_n[i] - current_n[i-1]) / (current_n[i+1] - current_n[i] + eps_div);
                gamma_drift = v * (current_n[i] + koren(r_i) * (current_n[i+1] - current_n[i]));
            } else {
                double inv_r = (current_n[i+2] - current_n[i+1]) / (current_n[i+1] - current_n[i] + eps_div);
                gamma_drift = v * (current_n[i+1] - koren(inv_r) * (current_n[i+1] - current_n[i]));
            }

            Gamma[i] = gamma_drift - D * (current_n[i+1] - current_n[i]) / dx;
        }

        for (int i = 2; i < n - 2; i++) k_out[i] = -(Gamma[i] - Gamma[i-1]) / dx; // Source is 0
    }

    void rk4_step() {
        evaluate_rate(n_e, k1);
        for (int i = 0; i < n; i++) n_temp[i] = n_e[i] + 0.5 * dt * k1[i];
        evaluate_rate(n_temp, k2);
        for (int i = 0; i < n; i++) n_temp[i] = n_e[i] + 0.5 * dt * k2[i];
        evaluate_rate(n_temp, k3);
        for (int i = 0; i < n; i++) n_temp[i] = n_e[i] + dt * k3[i];
        evaluate_rate(n_temp, k4);
        
        for (int i = 2; i < n - 2; i++) {
            n_e[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }
};

int main() {
    Coefficients coeffs;
    ThomasAlgorithm thomas;
    BoundaryConditions bc;
    DD_LFA_Solver plasma(coeffs, thomas, bc);

    double target_time = 50e-9; 
    int total_steps = target_time / dt;

    cout << "Starting Time Integration for " << total_steps << " steps..." << endl;
    
    for (int t = 0; t < total_steps; t++) {
        plasma.rk4_step();
        if (t % (total_steps / 10) == 0) cout << "Progress: " << (t * 100) / total_steps << "%" << endl;
    }

    ofstream denFile("density_data.csv");
    denFile << "cell,x_pos,electron_density\n";
    for (int i = 0; i < n; i++) denFile << i << "," << coeffs.x_position[i] << "," << plasma.n_e[i] << "\n";
    denFile.close();

    plasma.evaluate_rate(plasma.n_e, plasma.k1);
    ofstream eFile("efield_data.csv");
    eFile << "cell,electric_field\n";
    for (int i = 0; i < n - 1; i++) eFile << i << "," << thomas.ElectricField[i] << "\n";
    eFile.close();

    cout << "Simulation complete. Data saved." << endl;
    return 0;
}
