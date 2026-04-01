#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm> 
#include <omp.h>

using namespace std;

// ==========================================
// CONSTANTES GLOBAIS (Projeto 6 - TAFC)
// ==========================================
const double epsilon = 8.854187817e-12;  // Permissividade 
const double e_charge = 1.602176634e-19; // Carga elementar
const int n = 500;                     // 1000 células 
const double dx = 20e-6;                // 20 um 
const double dt = 0.1e-12;              // Passo de tempo para RK4 

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
        // Coeficientes para a malha uniforme 
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
// CLASSE SPIKE SOLVER
// ==========================================
class SpikeSolver {
public:
    double a[n], b[n], c[n], d[n]; //definir onde os dados são guardados
    double v[n], w[n]; 
    double solution[n], ElectricField[n];

    void solve(Coefficients& coeffs, BoundaryConditions& bc, double V_left, double V_right) {
        int p = omp_get_max_threads(); //o código pergunta quantas threads temos disponíveis (8 no meu pc)
        int m = n / p; //calcula o tamanho de cada bloco, no meu pc: 1000/8=125

        // FASE 1: Solução Local por Bloco (Paralela)
        #pragma omp parallel //esta diretiva diz ao compilador que tudo o que estiver dentro deste bloco deve ser executado por todas as threads ao mesmo tempo
        {
            int tid = omp_get_thread_num(); //atribui um ID a cada thread de 0 a 7
            int start = tid * m;
            int end = (tid == p - 1) ? n : (tid + 1) * m; //definem o intervalo de células de cada thread

            for (int i = start; i < end; i++) { //as threads copiam os coeficientes da classe Coefficients para a sua memória
                if (i > 0 && i < n - 1) {
                    a[i] = coeffs.phi_west[i]; 
                    b[i] = coeffs.phi_center[i]; 
                    c[i] = coeffs.phi_east[i];
                    d[i] = dx * coeffs.rho[i];
                }
                v[i] = w[i] = 0.0;
            }

            if (tid == 0) bc.apply_Dirichlet_left(coeffs, *this, V_left);
            if (tid == p - 1) bc.apply_Dirichlet_right(coeffs, *this, V_right); //aplicação das condições de Dirichlet

            if (tid > 0) v[start] = a[start];
            if (tid < p - 1) w[end - 1] = c[end - 1]; //guardar os valores das ligações entre os vizinhos de threads diferentes

            // Thomas Local
            for (int i = start + 1; i < end; i++) {
                double factor = a[i] / b[i - 1]; //fator de eliminação
                b[i] -= factor * c[i - 1]; //atualizar o coeficiente da diagonal principal
                d[i] -= factor * d[i - 1]; //atualização do termo da direita da equação (densidade de carga)
                v[i] -= factor * v[i - 1]; //este vetor rastreia como a solução de cada bloco é afetada pelo bloco anterior
            }

            // Back-substitution local
            solution[end - 1] = d[end - 1] / b[end - 1];
            v[end - 1] /= b[end - 1];
            w[end - 1] /= b[end - 1];

            for (int i = end - 2; i >= start; i--) {
                solution[i] = (d[i] - c[i] * solution[i + 1]) / b[i];
                v[i] = (v[i] - c[i] * v[i + 1]) / b[i];
                w[i] = (w[i] - c[i] * w[i + 1]) / b[i];
            } //cada thread tem uma solução preliminar para o potencial elétrico
        } //Cada thread gerou a sua assinatura de fronteira (v e w), que descreve como o seu pedaço de plasma comunica com os vizinhos

        // FASE 2: Sistema Reduzido Global
        int n_red = 2 * (p - 1); //8 threads significa 7 interfaces que significa 14 equações (uma para o último nó do bloco anterior e outra para o primeiro nó do bloco seguinte)
        vector<double> ra(n_red, 0.0), rb(n_red, 1.0), rc(n_red, 0.0), rd(n_red, 0.0); //sistema reduzido

        for (int k = 0; k < p - 1; k++) {
            int i_end = (k + 1) * m - 1;
            int i_next = (k + 1) * m;

            rd[2 * k] = solution[i_end]; //Colocamos a solução preliminar da thread atual no vetor de resultados do sistema reduzido
            if (k > 0) ra[2 * k] = v[i_end]; 
            rc[2 * k] = w[i_end]; // Usamos o spike da direita ($w$) para descrever a ligação com o bloco seguinte

            rd[2 * k + 1] = solution[i_next];
            ra[2 * k + 1] = v[i_next]; //Usamos o spike da esquerda (v) para descrever como o início do bloco afeta o fim
            if (k < p - 2) rc[2 * k + 1] = w[i_next];
        } //serve para sincronizar todas as threads

        // Solver Tridiagonal para o Sistema Reduzido
        for (int i = 1; i < n_red; i++) {
            double mf = ra[i] / rb[i - 1];
            rb[i] -= mf * rc[i - 1];
            rd[i] -= mf * rd[i - 1];
        }
        vector<double> res(n_red);
        res[n_red - 1] = rd[n_red - 1] / rb[n_red - 1];
        for (int i = n_red - 2; i >= 0; i--) res[i] = (rd[i] - rc[i] * res[i + 1]) / rb[i];

        for (int k = 0; k < p - 1; k++) {
            solution[(k + 1) * m - 1] = res[2 * k];
            solution[(k + 1) * m] = res[2 * k + 1];
        } //no fim deste bloco os 14 pontos fronteira já têm os seus valores finais e definitivos

        // FASE 3: Correção Final 
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int start = tid * m;
            int end = (tid == p - 1) ? n : (tid + 1) * m;
            double sol_L = (tid > 0) ? solution[start - 1] : 0.0; //valor do potencial no nó imediatamente à esquerda do bloco atual. Se fores a Thread 1, este é o último nó da Thread 0
            double sol_R = (tid < p - 1) ? solution[end] : 0.0; //valor do potencial no nó imediatamente à direita do bloco atual

            for (int i = start; i < end; i++) { 
                if (!((tid > 0 && i == start) || (tid < p - 1 && i == end - 1))) { //esta condição garante que não alteramos os nós que já foram corrigidos pelo Sistema Reduzido na fase anterior
                   solution[i] -= (v[i] * sol_L + w[i] * sol_R);
                }
            }
        }

        // Cálculo do Campo Elétrico 
        #pragma omp parallel for //dividir o cálculo d campo pelas 8 threads
        for (int i = 0; i < n - 1; i++) 
            ElectricField[i] = -(solution[i + 1] - solution[i]) / dx;
    } //cada thread usa os valores de fronteira obtidos na fase anterior para ajustar a sua solução local
};

inline void BoundaryConditions::apply_Dirichlet_left(Coefficients& coeffs, SpikeSolver& spike, double V_app) {
    double phi_west = -epsilon / (dx / 2.0);
    spike.a[0] = 0.0; spike.b[0] = -(phi_west + coeffs.phi_east[0]);
    spike.c[0] = coeffs.phi_east[0]; spike.d[0] = dx * coeffs.rho[0] - (phi_west * V_app);
}

inline void BoundaryConditions::apply_Dirichlet_right(Coefficients& coeffs, SpikeSolver& spike, double V_app) {
    double phi_east = -epsilon / (dx / 2.0);
    spike.a[n-1] = coeffs.phi_west[n-1]; spike.b[n-1] = -(coeffs.phi_west[n-1] + phi_east);
    spike.c[n-1] = 0.0; spike.d[n-1] = dx * coeffs.rho[n-1] - (phi_east * V_app);
}

// ==========================================
// CLASSE DD_LFA_SOLVER
// ==========================================
class DD_LFA_Solver {
public:
    vector<double> n_e, n_p, n_temp, Gamma, k1, k2, k3, k4;
    Coefficients& coeffs; SpikeSolver& spike; BoundaryConditions& bc;

    DD_LFA_Solver(Coefficients& c, SpikeSolver& s, BoundaryConditions& b) 
        : coeffs(c), spike(s), bc(b) {
        n_e.assign(n, 0.0); n_p.assign(n, 0.0); n_temp.assign(n, 0.0); Gamma.assign(n, 0.0);
        k1.assign(n, 0.0); k2.assign(n, 0.0); k3.assign(n, 0.0); k4.assign(n, 0.0);

        // Condição Inicial: Slab Neutro
        for(int i = 0; i < n; i++) {
            double x = i * dx;
            if (x >= 4.0e-3 && x <= 6.0e-3) {
                n_e[i] = 1e20; n_p[i] = 1e20;
            }
        }
    }//camada de plasma neutro entre 4mm e 6mm (carga total=0)

    double koren(double r) { return max(0.0, min({1.0, (2.0 + r) / 6.0, r})); } //limitador de koren para que a solução seja de alta resolução mas estável

    void evaluate_rate(const vector<double>& current_n, vector<double>& k_out) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) coeffs.rho[i] = (n_p[i] - current_n[i]) * e_charge;
        
        spike.solve(coeffs, bc, 10000.0, 0.0); //calcula a densidade elétrica e chama o spike solver para descobrir o campo elétrico criado por essa carga

        double eps_div = 1e-15; 
        double mu = 0.03; // m^2/(V.s) 
        double D = 0.1;   // m^2/s 

        #pragma omp parallel for
        for (int i = 1; i < n - 2; i++) {
            double E_face = spike.ElectricField[i];
            double v_drift = -mu * E_face;
            double g_drift = 0.0;

            if (v_drift >= 0.0) {
                double r_i = (current_n[i] - current_n[i-1]) / (current_n[i+1] - current_n[i] + eps_div); 
                g_drift = v_drift * (current_n[i] + koren(r_i) * (current_n[i+1] - current_n[i])); 
            } else {
                double inv_r = (current_n[i+2] - current_n[i+1]) / (current_n[i+1] - current_n[i] + eps_div); 
                g_drift = v_drift * (current_n[i+1] - koren(inv_r) * (current_n[i+1] - current_n[i]));
            }
            Gamma[i] = g_drift - D * (current_n[i+1] - current_n[i]) / dx;
        }

        #pragma omp parallel for //divide as 1000 células pelas threads
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
}; //runge-kutta de 4ªordem (4 avaliações da física por passo de tempo dt)

// ==========================================
// FUNÇÃO PRINCIPAL
// ==========================================
int main() {
    Coefficients coeffs;
    SpikeSolver spike;
    BoundaryConditions bc;
    DD_LFA_Solver plasma(coeffs, spike, bc);

    double target_time = 50e-9; //simulação durante 50 nanosegundos. como dt=0.1 o computador tem de calcular 500.000 frames
    int total_steps = target_time / dt;

    cout << "Threads OpenMP: " << omp_get_max_threads() << endl;
    cout << "Simulacao a decorrer..." << endl;
    
// Cronómetro
    double start_time = omp_get_wtime();

    for (int t = 0; t <= total_steps; t++) {
        plasma.rk4_step();

        // Mostrar progresso e tempo parcial
        if (t % (total_steps / 10) == 0) {
            double current_time = omp_get_wtime();
            double elapsed = current_time - start_time;
            cout << "Progresso: " << (t * 100) / total_steps << "% | "
                 << "Tempo decorrido: " << elapsed << " s" << endl;
        }
    }

    //FIM DO CRONÓMETRO
    double end_time = omp_get_wtime();
    double duration = end_time - start_time;

    // Guardar Densidade
    ofstream denFile("density_spike.csv");
    denFile << "cell,x_pos,electron_density\n";
    for (int i = 0; i < n; i++) denFile << i << "," << coeffs.x_position[i] << "," << plasma.n_e[i] << "\n";
    denFile.close();

    // Atualizar e guardar Campo Elétrico final
    plasma.evaluate_rate(plasma.n_e, plasma.k1);
    ofstream eFile("efield_spike.csv");
    eFile << "cell,electric_field\n";
    for (int i = 0; i < n - 1; i++) eFile << i << "," << spike.ElectricField[i] << "\n";
    eFile.close();

    cout << "------------------------------------------" << endl;
    cout << "Simulacao CONCLUIDA!" << endl;
    cout << "Tempo Total de Execucao: " << duration << " segundos." << endl;
    cout << "Ficheiros gerados com sucesso." << endl;
    cout << "------------------------------------------" << endl;

    return 0;
}
