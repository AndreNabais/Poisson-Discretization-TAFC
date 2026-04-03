#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <omp.h>

using namespace std;

// ==========================================
// CONSTANTES GLOBAIS (Teunissen Test Case)
// ==========================================
const double epsilon = 8.854187817e-12;
const double e_charge = 1.602176634e-19;
const int n = 500;
const double dx = 20e-6;
const double dt = 80e-12; 

// ==========================================
// SPIKE SOLVER 
// ==========================================
class SpikeSolver {
public:
    double a[n], b[n], c[n], d[n], solution[n], v[n], w[n], ElectricField[n];
    vector<int> starts, ends;
    int p;

    void solve_spike() {
        p = omp_get_max_threads();
        if (p < 2) p = 2;
        int m = n / p;
        
        starts.assign(p, 0); ends.assign(p, 0);

        // FASE 1: Solução Local e Geração de Spikes (Paralela) 
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            starts[tid] = tid * m;
            ends[tid] = (tid == p - 1) ? n : (tid + 1) * m;
            int s = starts[tid], e = ends[tid];

            for(int i = s; i < e; i++) v[i] = w[i] = 0.0;
            if (tid > 0) v[s] = a[s];
            if (tid < p - 1) w[e - 1] = c[e - 1];

            // Thomas Local Modificado 
            for (int i = s + 1; i < e; i++) {
                double factor = a[i] / b[i - 1];
                b[i] -= factor * c[i - 1];
                d[i] -= factor * d[i - 1];
                v[i] -= factor * v[i - 1];
            }
            solution[e - 1] = d[e - 1] / b[e - 1];
            v[e - 1] /= b[e - 1]; w[e - 1] /= b[e - 1];

            for (int i = e - 2; i >= s; i--) {
                solution[i] = (d[i] - c[i] * solution[i + 1]) / b[i];
                v[i] = (v[i] - c[i] * v[i + 1]) / b[i];
                w[i] = (w[i] - c[i] * w[i + 1]) / b[i];
            }
        }

        // FASE 2: Sistema Reduzido Global (Eliminação Gaussiana)
        int n_red = 2 * (p - 1);
        vector<vector<double>> M(n_red, vector<double>(n_red + 1, 0.0));
        
        for (int k = 0; k < p - 1; k++) {
            int i_end = ends[k] - 1; 
            int i_next = starts[k+1];
            int r_end = 2 * k;
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

        // Solver por Eliminação Gaussiana com Pivotagem Parcial 
        for (int col = 0; col < n_red; col++) {
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

        for (int k = 0; k < p - 1; k++) {
            solution[ends[k]-1] = res[2*k]; 
            solution[starts[k+1]] = res[2*k+1];
        }

        // FASE 3: Correção Final e Sincronização (Paralela)
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int s = starts[tid], e = ends[tid];
            double sol_L = (tid > 0) ? solution[s - 1] : 0.0;
            double sol_R = (tid < p - 1) ? solution[e] : 0.0;
            for (int i = s; i < e; i++) {
                // Proteção dos nós de interface já resolvidos 
                if (!((tid > 0 && i == s) || (tid < p - 1 && i == e - 1)))
                    solution[i] -= (v[i] * sol_L + w[i] * sol_R);
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < n - 1; i++) 
            ElectricField[i] = -(solution[i + 1] - solution[i]) / dx;
    }
};

// ==========================================
// CURRENT-LIMITED SOLVER PARALELIZADO
// ==========================================
class CurrentLimitedSolver {
public:
    double x_pos[n], n_e[n], n_p[n], Gamma[n], n_e_temp[n], k1[n], k2[n];
    SpikeSolver spike;
    double mu = 0.03, D_e = 0.1;

    CurrentLimitedSolver() {
        for (int i = 0; i < n; i++) {
            x_pos[i] = i * dx;
            // Inicialização do Slab de Plasma Neutro 
            n_e[i] = n_p[i] = (x_pos[i] >= 4.0e-3 && x_pos[i] <= 6.0e-3) ? 1e20 : 0.0;
        }
    }

    void solve_poisson(const double* n_in, double V_L, double V_R) {
        double e_dx2 = epsilon / (dx * dx);
        #pragma omp parallel for
        for (int i = 1; i < n - 1; i++) {
            spike.a[i] = spike.c[i] = -e_dx2;
            spike.b[i] = 2.0 * e_dx2;
            spike.d[i] = e_charge * (n_p[i] - n_in[i]);
        }
        // Condições de Dirichlet com parede a dx/2 
        spike.a[0] = 0.0; spike.c[0] = -e_dx2; spike.b[0] = 3.0 * e_dx2;
        spike.d[0] = e_charge * (n_p[0] - n_in[0]) + (2.0 * e_dx2) * V_L;
        spike.c[n-1] = 0.0; spike.a[n-1] = -e_dx2; spike.b[n-1] = 3.0 * e_dx2;
        spike.d[n-1] = e_charge * (n_p[n-1] - n_in[n-1]) + (2.0 * e_dx2) * V_R;
        spike.solve_spike();
    }

    double koren(double r) { return max(0.0, min({1.0, (2.0 + r) / 6.0, r})); } 

    void compute_rates(const double* n_in, double* k_out) {
        solve_poisson(n_in, 10000.0, 0.0);
        double eps_div = 1e-15;

        #pragma omp parallel for
        for (int i = 1; i < n - 2; i++) {
            double v = -mu * spike.ElectricField[i];
            double g_drift = 0.0;
            // Esquema Upwind para Fluxo de Drift 
            if (v >= 0.0) {
                double r = (n_in[i] - n_in[i-1]) / (n_in[i+1] - n_in[i] + eps_div);
                g_drift = v * (n_in[i] + koren(r) * (n_in[i+1] - n_in[i]));
            } else {
                double r_inv = (n_in[i+2] - n_in[i+1]) / (n_in[i+1] - n_in[i] + eps_div);
                g_drift = v * (n_in[i+1] - koren(r_inv) * (n_in[i+1] - n_in[i]));
            }
            double Gamma_raw = g_drift - D_e * (n_in[i+1] - n_in[i]) / dx;

            // LÓGICA CURRENT-LIMITER (Teunissen Sec. 3.3) 
            double n_grad = abs(n_in[i+1] - n_in[i]) / dx;
            double n_max = max({n_in[i], n_in[i+1], 1e-15});
            // Campo crítico E* que equilibra drift e difusão 
            double E_star = max(abs(spike.ElectricField[i]), (D_e * (n_grad / n_max)) / mu);
            double max_flux = (epsilon * E_star) / (e_charge * dt); // Eq 18 

            // Clipping do Fluxo 
            Gamma[i] = (abs(Gamma_raw) > max_flux) ? (Gamma_raw > 0 ? 1.0 : -1.0) * max_flux : Gamma_raw;
        }

        #pragma omp parallel for
        for (int i = 2; i < n - 2; i++) k_out[i] = -(Gamma[i] - Gamma[i-1]) / dx; 
    }

    void step() {
        // Integração Trapezoidal Explícita (2nd order) 
        compute_rates(n_e, k1);
        #pragma omp parallel for
        for (int i = 0; i < n; i++) n_e_temp[i] = n_e[i] + dt * k1[i];
        compute_rates(n_e_temp, k2);
        #pragma omp parallel for
        for (int i = 2; i < n - 2; i++) n_e[i] += (dt / 2.0) * (k1[i] + k2[i]); 
    }
};

int main() {
    CurrentLimitedSolver sim;
    int steps = 50e-9 / dt;
    double start = omp_get_wtime();

    cout << "Threads OpenMP: " << omp_get_max_threads() << " | Metodo: Current-Limiter" << endl;

    for (int t = 0; t <= steps; t++) {
        sim.step();
        if (t % (steps / 10) == 0) 
            cout << "Progresso: " << (t * 100) / steps << "% | Tempo parcial: " << omp_get_wtime() - start << " s" << endl;
    }

    ofstream f1("density_current_lim_spike2.csv");
    f1 << "cell,x_pos,electron_density\n";
    for (int i = 0; i < n; i++) f1 << i << "," << sim.x_pos[i] << "," << sim.n_e[i] << "\n";
    f1.close();

    sim.solve_poisson(sim.n_e, 10000.0, 0.0);
    ofstream f2("efield_current_lim_spike2.csv");
    f2 << "cell,electric_field\n";
    for (int i = 0; i < n - 1; i++) f2 << i << "," << sim.spike.ElectricField[i] << "\n";
    f2.close();

    cout << "Simulacao concluida em: " << omp_get_wtime() - start << " s" << endl;
    return 0;
}
