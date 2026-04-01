#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <omp.h>

using namespace std;

// ==========================================
// PARÂMETROS GLOBAIS
// ==========================================
const int n = 500;               // 500 células 
const double L = 0.01;            // 10 mm
const double dx = 20e-6;          // 20 um
const double dt = 80e-12;         // 80 ps (Passo grande do Semi-Implícito)
const double epsilon = 8.854187817e-12;
const double e_charge = 1.602176634e-19;
const int p = omp_get_max_threads();                 // o código pergunta ao computador quantas threads tem

// ==========================================
// SOLVER SPIKE PARA SEMI-IMPLÍCITO
// ==========================================
class SpikeSolver {
public:
    double a[n], b[n], c[n], d[n], solution[n], v[n], w[n], ElectricField[n];
    int m;

    SpikeSolver() {
        m = n / p;
        for(int i=0; i<n; i++) { solution[i] = 0.0; ElectricField[i] = 0.0; }
    }

    void solve_spike(double V_L, double V_R) {
        // FASE 1: Eliminação Local (Paralela)
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int start = tid * m;
            int end = (tid == p - 1) ? n : (tid + 1) * m;

            // Inicializar Spikes
            for(int i=start; i<end; i++) v[i] = w[i] = 0.0;
            if (tid > 0) v[start] = a[start];
            if (tid < p - 1) w[end - 1] = c[end - 1];

            // Thomas Local (Eliminação)
            for (int i = start + 1; i < end; i++) {
                double mf = a[i] / b[i - 1];
                b[i] -= mf * c[i - 1];
                d[i] -= mf * d[i - 1];
                v[i] -= mf * v[i - 1];
            }

            // Back-substitution local
            solution[end - 1] = d[end - 1] / b[end - 1];
            v[end - 1] /= b[end - 1];
            w[end - 1] /= b[end - 1];

            for (int i = end - 2; i >= start; i--) {
                solution[i] = (d[i] - c[i] * solution[i + 1]) / b[i];
                v[i] = (v[i] - c[i] * v[i + 1]) / b[i];
                w[i] = (w[i] - c[i] * w[i + 1]) / b[i];
            }
        }

        // FASE 2: Sistema Reduzido (Sequencial)
        int n_red = 2 * (p - 1);
        vector<double> ra(n_red, 0.0), rb(n_red, 1.0), rc(n_red, 0.0), rd(n_red, 0.0);

        for (int k = 0; k < p - 1; k++) {
            int i_end = (k + 1) * m - 1;
            int i_next = (k + 1) * m;
            rd[2 * k] = solution[i_end];
            if (k > 0) ra[2 * k] = v[i_end]; 
            rc[2 * k] = w[i_end];
            rd[2 * k + 1] = solution[i_next];
            ra[2 * k + 1] = v[i_next];
            if (k < p - 2) rc[2 * k + 1] = w[i_next];
        }

        for (int i = 1; i < n_red; i++) {
            double mf = ra[i] / rb[i - 1];
            rb[i] -= mf * rc[i - 1];
            rd[i] -= mf * rd[i - 1];
        }
        vector<double> res_vec(n_red);
        res_vec[n_red - 1] = rd[n_red - 1] / rb[n_red - 1];
        for (int i = n_red - 2; i >= 0; i--) res_vec[i] = (rd[i] - rc[i] * res_vec[i + 1]) / rb[i];

        for (int k = 0; k < p - 1; k++) {
            solution[(k + 1) * m - 1] = res_vec[2 * k];
            solution[(k + 1) * m] = res_vec[2 * k + 1];
        }

        // FASE 3: Correção Final (Paralela)
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

        // Extrair Campo Elétrico
        #pragma omp parallel for
        for (int i = 0; i < n - 1; i++) 
            ElectricField[i] = -(solution[i + 1] - solution[i]) / dx;
    }
};

// ==========================================
// CLASSE SEMI-IMPLICIT SOLVER
// ==========================================
class SemiImplicitSolver {
public:
    double n_e[n], n_p[n], Gamma[n], eps_eff[n], x_pos[n];
    SpikeSolver spike;
    double mu = 0.03, D_e = 0.1;

    SemiImplicitSolver() {
        for (int i = 0; i < n; i++) {
            x_pos[i] = i * dx;
            n_e[i] = n_p[i] = (x_pos[i] >= 4.0e-3 && x_pos[i] <= 6.0e-3) ? 1e20 : 0.0;
            Gamma[i] = 0.0;
        }
    }

    double koren(double r) { return max(0.0, min({1.0, (2.0 + r) / 6.0, r})); }

    void solve_modified_poisson(double V_L, double V_R) {
        // 1. Permissividade Efetiva (Paralelo)
        #pragma omp parallel for
        for (int i = 0; i < n - 1; i++) {
            double n_e_face = 0.5 * (n_e[i] + n_e[i+1]);
            eps_eff[i] = epsilon + e_charge * dt * mu * n_e_face;
        }

        // 2. Montar Matriz Spike (Paralelo)
        #pragma omp parallel for
        for (int i = 1; i < n - 1; i++) {
            spike.a[i] = -eps_eff[i-1] / (dx * dx);
            spike.c[i] = -eps_eff[i] / (dx * dx);
            spike.b[i] = -(spike.a[i] + spike.c[i]);
            double diff_div = D_e * (n_e[i+1] - 2.0 * n_e[i] + n_e[i-1]) / (dx * dx);
            spike.d[i] = e_charge * (n_p[i] - n_e[i]) - e_charge * dt * diff_div;
        }

        // 3. Fronteiras Dirichlet
        spike.c[0] = -eps_eff[0] / (dx * dx); spike.a[0] = 0.0;
        spike.b[0] = -spike.c[0] + 2.0 * epsilon / (dx * dx);
        double diff_div_0 = D_e * (n_e[1] - 2.0 * n_e[0] + 0.0) / (dx * dx);
        spike.d[0] = e_charge * (n_p[0] - n_e[0]) - e_charge * dt * diff_div_0 + (2.0 * epsilon / (dx * dx)) * V_L;

        spike.a[n-1] = -eps_eff[n-2] / (dx * dx); spike.c[n-1] = 0.0;
        spike.b[n-1] = -spike.a[n-1] + 2.0 * epsilon / (dx * dx);
        double diff_div_n = D_e * (0.0 - 2.0 * n_e[n-1] + n_e[n-2]) / (dx * dx);
        spike.d[n-1] = e_charge * (n_p[n-1] - n_e[n-1]) - e_charge * dt * diff_div_n + (2.0 * epsilon / (dx * dx)) * V_R;

        spike.solve_spike(V_L, V_R);
    }

    void solve_standard_poisson(double V_L, double V_R) {
        double eps_over_dx2 = epsilon / (dx * dx);
        #pragma omp parallel for
        for (int i = 1; i < n - 1; i++) {
            spike.a[i] = spike.c[i] = -eps_over_dx2;
            spike.b[i] = 2.0 * eps_over_dx2;
            spike.d[i] = e_charge * (n_p[i] - n_e[i]); 
        }
        spike.c[0] = -eps_over_dx2; spike.a[0] = 0.0; spike.b[0] = 3.0 * eps_over_dx2; 
        spike.d[0] = e_charge * (n_p[0] - n_e[0]) + (2.0 * eps_over_dx2) * V_L;
        spike.a[n-1] = -eps_over_dx2; spike.c[n-1] = 0.0; spike.b[n-1] = 3.0 * eps_over_dx2;
        spike.d[n-1] = e_charge * (n_p[n-1] - n_e[n-1]) + (2.0 * eps_over_dx2) * V_R;
        spike.solve_spike(V_L, V_R);
    }

    void euler_transport_step() {
        double eps_div = 1e-15; 
        #pragma omp parallel for
        for (int i = 1; i < n - 2; i++) {
            double v = -mu * spike.ElectricField[i];
            double g_drift = 0.0;
            if (v >= 0.0) {
                double r = (n_e[i] - n_e[i-1]) / (n_e[i+1] - n_e[i] + eps_div);
                g_drift = v * (n_e[i] + koren(r) * (n_e[i+1] - n_e[i]));
            } else {
                double r_inv = (n_e[i+2] - n_e[i+1]) / (n_e[i+1] - n_e[i] + eps_div);
                g_drift = v * (n_e[i+1] - koren(r_inv) * (n_e[i+1] - n_e[i]));
            }
            Gamma[i] = g_drift - D_e * (n_e[i+1] - n_e[i]) / dx;
        }
        #pragma omp parallel for
        for (int i = 2; i < n - 2; i++) n_e[i] += dt * (-(Gamma[i] - Gamma[i-1]) / dx);
    }
};

int main() {
    SemiImplicitSolver plasma;
    double target_time = 50e-9;
    int total_steps = target_time / dt;

    cout << "Threads OpenMP: " << omp_get_max_threads() << " | Passos: " << total_steps << endl;
    double start = omp_get_wtime();

    for (int t = 0; t <= total_steps; t++) {
        plasma.solve_modified_poisson(10000.0, 0.0);
        plasma.euler_transport_step();
        if (t % (total_steps / 10) == 0) 
            cout << "Progresso: " << (t * 100) / total_steps << "% | Tempo: " << omp_get_wtime() - start << "s" << endl;
    }

    ofstream denFile("density_semi_spike.csv");
    denFile << "cell,electron_density\n";
    for (int i = 0; i < n; i++) denFile << i << "," << plasma.n_e[i] << "\n";
    denFile.close();

    plasma.solve_standard_poisson(10000.0, 0.0);
    ofstream eFile("efield_semi_spike.csv");
    eFile << "cell,electric_field\n";
    for (int i = 0; i < n - 1; i++) eFile << i << "," << plasma.spike.ElectricField[i] << "\n";
    eFile.close();

    cout << "Concluido em: " << omp_get_wtime() - start << " s" << endl;
    return 0;
}
