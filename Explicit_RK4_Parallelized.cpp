#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm> 
#include <omp.h>

using namespace std;

// ==========================================
// CONSTANTES GLOBAIS (Teunissen Test Case)
// ==========================================
const double epsilon = 8.854187817e-12;  // Permissividade 
const double e_charge = 1.602176634e-19; // Carga elementar 
const int n = 500;                       // Número de células 
const double dx = 20e-6;                 // Espaçamento espacial (20 um) 
const double dt = 0.1e-12;               // Passo de tempo para RK4 (0.1 ps) 

class SpikeSolver;

// ==========================================
// CLASSE COEFFICIENTS
// ==========================================
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
        // Coeficientes tridiagonais para Poisson em grelha uniforme 
        for (int i = 0; i < n - 1; i++) phi_east[i] = -epsilon / dx;
        for (int i = 1; i < n; i++) phi_west[i] = -epsilon / dx;
        for (int i = 1; i < n - 1; i++) phi_center[i] = -(phi_west[i] + phi_east[i]);
    }
};

class BoundaryConditions {
public:
    void apply_Dirichlet_left(Coefficients& coeffs, SpikeSolver& spike, double V_app);
    void apply_Dirichlet_right(Coefficients& coeffs, SpikeSolver& spike, double V_app);
};

// ==========================================
// CLASSE SPIKE SOLVER (Algoritmo de SPIKE)
// ==========================================
class SpikeSolver {
public:
    double a[n], b[n], c[n], d[n]; 
    double v[n], w[n]; 
    double solution[n], ElectricField[n];

    void solve(Coefficients& coeffs, BoundaryConditions& bc, double V_left, double V_right) {
        int p = omp_get_max_threads(); 
        int m = n / p; 

        // FASE 1: Solução Local e Criação de Spikes (Paralela) 
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int start = tid * m;
            int end = (tid == p - 1) ? n : (tid + 1) * m;

            for (int i = start; i < end; i++) {
                if (i > 0 && i < n - 1) {
                    a[i] = coeffs.phi_west[i]; 
                    b[i] = coeffs.phi_center[i]; 
                    c[i] = coeffs.phi_east[i];
                    d[i] = dx * coeffs.rho[i];
                }
                v[i] = w[i] = 0.0;
            }

            if (tid == 0) bc.apply_Dirichlet_left(coeffs, *this, V_left);
            if (tid == p - 1) bc.apply_Dirichlet_right(coeffs, *this, V_right);

            if (tid > 0) v[start] = a[start];
            if (tid < p - 1) w[end - 1] = c[end - 1];

            // Thomas Local Modificado para incluir vetores spike 
            for (int i = start + 1; i < end; i++) {
                double factor = a[i] / b[i - 1];
                b[i] -= factor * c[i - 1];
                d[i] -= factor * d[i - 1];
                v[i] -= factor * v[i - 1];
            }

            solution[end - 1] = d[end - 1] / b[end - 1];
            v[end - 1] /= b[end - 1];
            w[end - 1] /= b[end - 1];

            for (int i = end - 2; i >= start; i--) {
                solution[i] = (d[i] - c[i] * solution[i + 1]) / b[i];
                v[i] = (v[i] - c[i] * v[i + 1]) / b[i];
                w[i] = (w[i] - c[i] * w[i + 1]) / b[i];
            }
        }

        // FASE 2: Sistema Reduzido Global (Eliminação Gaussiana) 
        int n_red = 2 * (p - 1); //Define o tamanho da matriz do sistema reduzido (se p=i então temos p=i-1 interfaces)
        vector<vector<double>> M(n_red, vector<double>(n_red + 1, 0.0));

        for (int k = 0; k < p - 1; k++) { //ligação entre os nós
            int i_end  = (k + 1) * m - 1;
            int i_next = (k + 1) * m;
            int r_end  = 2 * k;
            int r_next = 2 * k + 1;

            M[r_end][r_end] = 1.0;
            if (k > 0) M[r_end][r_end - 2] = v[i_end]; 
            M[r_end][r_next] = w[i_end];
            M[r_end][n_red] = solution[i_end];

            M[r_next][r_next] = 1.0;
            M[r_next][r_end] = v[i_next];
            if (k < p - 2) M[r_next][r_next + 2] = w[i_next];
            M[r_next][n_red] = solution[i_next];
        }

        // Solver denso para o sistema reduzido (Pivotagem Parcial)
        for (int col = 0; col < n_red; col++) { //Ao contrário do algoritmo de Thomas (que só limpa as diagonais), este método limpa todos os elementos fora da diagonal, permitindo resolver matrizes que não sejam puramente tridiagonais
            int pivot = col;
            for (int row = col + 1; row < n_red; row++)
                if (fabs(M[row][col]) > fabs(M[pivot][col])) pivot = row;
            swap(M[col], M[pivot]);

            for (int row = 0; row < n_red; row++) {
                if (row != col && fabs(M[col][col]) > 1e-30) {
                    double f = M[row][col] / M[col][col];
                    for (int j = col; j <= n_red; j++) M[row][j] -= f * M[col][j];
                }
            }
        }

        vector<double> res(n_red);
        for (int i = 0; i < n_red; i++)
            res[i] = (fabs(M[i][i]) > 1e-30) ? M[i][n_red] / M[i][i] : 0.0;

        for (int k = 0; k < p - 1; k++) { //O código substitui as soluções preliminares na matriz solution global pelos valores corretos obtidos pelo sistema reduzido
            solution[(k + 1) * m - 1] = res[2 * k];
            solution[(k + 1) * m] = res[2 * k + 1];
        }

        // FASE 3: Correção Final dos Nós Interiores (Paralela)
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int start = tid * m;
            int end = (tid == p - 1) ? n : (tid + 1) * m;
            double sol_L = (tid > 0) ? solution[start - 1] : 0.0;
            double sol_R = (tid < p - 1) ? solution[end] : 0.0;

            for (int i = start; i < end; i++) { 
                if (!((tid > 0 && i == start) || (tid < p - 1 && i == end - 1))) {
                   solution[i] -= (v[i] * sol_L + w[i] * sol_R);
                }
            }
        }

        // Cálculo do Campo Elétrico nas faces
        #pragma omp parallel for
        for (int i = 0; i < n - 1; i++) 
            ElectricField[i] = -(solution[i + 1] - solution[i]) / dx;
    }
};

inline void BoundaryConditions::apply_Dirichlet_left(Coefficients& coeffs, SpikeSolver& spike, double V_app) {
    double phi_west_bc = -epsilon / (dx / 2.0);
    spike.a[0] = 0.0; 
    spike.b[0] = -(phi_west_bc + coeffs.phi_east[0]);
    spike.c[0] = coeffs.phi_east[0]; 
    spike.d[0] = dx * coeffs.rho[0] - (phi_west_bc * V_app);
}

inline void BoundaryConditions::apply_Dirichlet_right(Coefficients& coeffs, SpikeSolver& spike, double V_app) {
    double phi_east_bc = -epsilon / (dx / 2.0);
    spike.a[n-1] = coeffs.phi_west[n-1]; 
    spike.b[n-1] = -(coeffs.phi_west[n-1] + phi_east_bc);
    spike.c[n-1] = 0.0; 
    spike.d[n-1] = dx * coeffs.rho[n-1] - (phi_east_bc * V_app);
}

// ==========================================
// CLASSE DD_LFA_SOLVER (RK4 + Koren Limiter)
// ==========================================
class DD_LFA_Solver {
public:
    vector<double> n_e, n_p, n_temp, Gamma, k1, k2, k3, k4;
    Coefficients& coeffs; SpikeSolver& spike; BoundaryConditions& bc;

    DD_LFA_Solver(Coefficients& c, SpikeSolver& s, BoundaryConditions& b) 
        : coeffs(c), spike(s), bc(b) {
        n_e.assign(n, 0.0); n_p.assign(n, 0.0); n_temp.assign(n, 0.0); Gamma.assign(n, 0.0);
        k1.assign(n, 0.0); k2.assign(n, 0.0); k3.assign(n, 0.0); k4.assign(n, 0.0);

        // Condição Inicial: Neutral Plasma Slab (4mm a 6mm) 
        for(int i = 0; i < n; i++) {
            double x = i * dx;
            if (x >= 4.0e-3 && x <= 6.0e-3) {
                n_e[i] = 1e20; n_p[i] = 1e20;
            }
        }
    }

    double koren(double r) { return max(0.0, min({1.0, (2.0 + r) / 6.0, r})); }

    void evaluate_rate(const vector<double>& current_n, vector<double>& k_out) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) coeffs.rho[i] = (n_p[i] - current_n[i]) * e_charge;
        
        spike.solve(coeffs, bc, 10000.0, 0.0); // 10 kV de diferença de potencial 

        double eps_div = 1e-15; 
        double mu = 0.03; 
        double D = 0.1;   

        #pragma omp parallel for
        for (int i = 1; i < n - 2; i++) {
            double E_face = spike.ElectricField[i];
            double v_drift = -mu * E_face; // v = -mu*E 
            double g_drift = 0.0;

            if (v_drift >= 0.0) {
                double r_i = (current_n[i] - current_n[i-1]) / (current_n[i+1] - current_n[i] + eps_div); 
                g_drift = v_drift * (current_n[i] + koren(r_i) * (current_n[i+1] - current_n[i])); 
            } else {
                double inv_r = (current_n[i+2] - current_n[i+1]) / (current_n[i+1] - current_n[i] + eps_div); 
                g_drift = v_drift * (current_n[i+1] - koren(inv_r) * (current_n[i+1] - current_n[i]));
            }
            Gamma[i] = g_drift - D * (current_n[i+1] - current_n[i]) / dx; // Fluxo total 
        }

        #pragma omp parallel for
        for (int i = 2; i < n - 2; i++) k_out[i] = -(Gamma[i] - Gamma[i-1]) / dx;
    }

    void rk4_step() { 
        evaluate_rate(n_e, k1);
        #pragma omp parallel for
        for (int i = 0; i < n; i++) n_temp[i] = n_e[i] + 0.5 * dt * k1[i];
        
        evaluate_rate(n_temp, k2);
        #pragma omp parallel for
        for (int i = 0; i < n; i++) n_temp[i] = n_e[i] + 0.5 * dt * k2[i];
        
        evaluate_rate(n_temp, k3);
        #pragma omp parallel for
        for (int i = 0; i < n; i++) n_temp[i] = n_e[i] + dt * k3[i];
        
        evaluate_rate(n_temp, k4);
        #pragma omp parallel for
        for (int i = 2; i < n - 2; i++) {
            n_e[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }
};

int main() {
    Coefficients coeffs;
    SpikeSolver spike;
    BoundaryConditions bc;
    DD_LFA_Solver plasma(coeffs, spike, bc);

    double target_time = 50e-9; 
    int total_steps = target_time / dt;

    cout << "Threads OpenMP: " << omp_get_max_threads() << endl;
    cout << "Iniciando simulacao..." << endl;
    
    double start_time = omp_get_wtime();

    for (int t = 0; t <= total_steps; t++) {
        plasma.rk4_step();
        if (t % (total_steps / 10) == 0) {
            double elapsed = omp_get_wtime() - start_time;
            cout << "Progresso: " << (t * 100) / total_steps << "% | Tempo: " << elapsed << " s" << endl;
        }
    }

    double duration = omp_get_wtime() - start_time;

    // Guardar resultados
    ofstream denFile("density_spike2.csv");
    denFile << "cell,x_pos,electron_density\n";
    for (int i = 0; i < n; i++) denFile << i << "," << coeffs.x_position[i] << "," << plasma.n_e[i] << "\n";
    denFile.close();

    plasma.evaluate_rate(plasma.n_e, plasma.k1);
    ofstream eFile("efield_spike2.csv");
    eFile << "cell,electric_field\n";
    for (int i = 0; i < n - 1; i++) eFile << i << "," << spike.ElectricField[i] << "\n";
    eFile.close();

    cout << "Simulacao concluida em " << duration << " segundos." << endl;
    return 0;
}
