#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <omp.h>

using namespace std;

// ==========================================
// CONSTANTES GLOBAIS
// ==========================================
const double epsilon = 8.854187817e-12;
const double e_charge = 1.602176634e-19;
const int n = 500;
const double dx = 20e-6;
const double dt = 80e-12; // Passo largo estabilizado pelo Current-Limiting

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

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            starts[tid] = tid * m;
            ends[tid] = (tid == p - 1) ? n : (tid + 1) * m;
            int s = starts[tid], e = ends[tid];

            for(int i = s; i < e; i++) v[i] = w[i] = 0.0;
            if (tid > 0) v[s] = a[s];
            if (tid < p - 1) w[e - 1] = c[e - 1];

            // Thomas Local
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

        // Sistema Reduzido
        int n_red = 2 * (p - 1);
        vector<double> ra(n_red, 0.0), rb(n_red, 1.0), rc(n_red, 0.0), rd(n_red, 0.0);
        for (int k = 0; k < p - 1; k++) {
            int i_end = ends[k] - 1; int i_next = starts[k+1];
            rd[2*k] = solution[i_end]; if (k > 0) ra[2*k] = v[i_end]; rc[2*k] = w[i_end];
            rd[2*k+1] = solution[i_next]; ra[2*k+1] = v[i_next]; if (k < p - 2) rc[2*k+1] = w[i_next];
        }
        for (int i = 1; i < n_red; i++) {
            double mf = ra[i] / rb[i-1]; rb[i] -= mf * rc[i-1]; rd[i] -= mf * rd[i-1];
        }
        vector<double> res(n_red); res[n_red-1] = rd[n_red-1] / rb[n_red-1];
        for (int i = n_red - 2; i >= 0; i--) res[i] = (rd[i] - rc[i] * res[i+1]) / rb[i];

        for (int k = 0; k < p - 1; k++) {
            solution[ends[k]-1] = res[2*k]; solution[starts[k+1]] = res[2*k+1];
        }

        // Correção Final
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int s = starts[tid], e = ends[tid];
            double sol_L = (tid > 0) ? solution[s - 1] : 0.0;
            double sol_R = (tid < p - 1) ? solution[e] : 0.0;
            for (int i = s; i < e; i++) {
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
            if (v >= 0.0) {
                double r = (n_in[i] - n_in[i-1]) / (n_in[i+1] - n_in[i] + eps_div);
                g_drift = v * (n_in[i] + koren(r) * (n_in[i+1] - n_in[i]));
            } else {
                double r_inv = (n_in[i+2] - n_in[i+1]) / (n_in[i+1] - n_in[i] + eps_div);
                g_drift = v * (n_in[i+1] - koren(r_inv) * (n_in[i+1] - n_in[i]));
            }
            double Gamma_raw = g_drift - D_e * (n_in[i+1] - n_in[i]) / dx;

            // LÓGICA CURRENT-LIMITER (Teunissen)
            double n_grad = abs(n_in[i+1] - n_in[i]) / dx;
            double n_max = max({n_in[i], n_in[i+1], 1e-15});
            double E_star = max(abs(spike.ElectricField[i]), (D_e * (n_grad / n_max)) / mu);
            double max_flux = (epsilon * E_star) / (e_charge * dt);

            Gamma[i] = (abs(Gamma_raw) > max_flux) ? (Gamma_raw > 0 ? 1 : -1) * max_flux : Gamma_raw;
        }

        #pragma omp parallel for
        for (int i = 2; i < n - 2; i++) k_out[i] = -(Gamma[i] - Gamma[i-1]) / dx;
    }

    void step() {
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

    for (int t = 0; t <= steps; t++) {
        sim.step();
        if (t % (steps / 10) == 0) cout << "Progresso: " << (t * 100) / steps << "%" << endl;
    }

    ofstream f1("density_current_lim_spike.csv");
    f1 << "cell,electron_density\n";
    for (int i = 0; i < n; i++) f1 << i << "," << sim.n_e[i] << "\n";
    f1.close();

    sim.solve_poisson(sim.n_e, 10000.0, 0.0);
    ofstream f2("efield_current_lim_spike.csv");
    f2 << "cell,electric_field\n";
    for (int i = 0; i < n - 1; i++) f2 << i << "," << sim.spike.ElectricField[i] << "\n";
    f2.close();

    cout << "Tempo: " << omp_get_wtime() - start << " s" << endl;
    return 0;
}
