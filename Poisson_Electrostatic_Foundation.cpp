#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
using namespace std;

//Forward declaration of classes
class Coefficients;
class ThomasAlgorithm;
class BoundaryConditions;

double epsilon = 8.854187817e-12; // Vacuum permittivity
const int n = 500; // Number of spatial points
double dx = 20e-6; // Spatial step size

class Coefficients {
public:
double x_position[n]; // Spatial positions
double Voltage[n]; // Electric potential at each spatial point
double rho[n]; // Charge density at each spatial point
double phi_west[n]; // Coefficient for west neighbor
double phi_east[n]; // Coefficient for east neighbor
double phi_center[n]; // Coefficient for center point

Coefficients() {
    for (int i = 0; i < n; i++) {
        x_position[i] = i * dx; // Spatial positions
        phi_west[i] = phi_east[i] = phi_center[i] = 0.0; // Initialize coefficients to zero
        Voltage[i] = 0.0; // Initialize voltage to zero
        rho[i] = 0.0; // Start with Laplace
        //rho[i] = 1.0e-2; //Fixed charge density to calculate Poisson's equation and get non-zero electric field for testing
    }

    //define west links
    for (int i = 0; i < n-1; i++) {
        phi_east[i] = - epsilon / (x_position[i+1]-x_position[i]); //Coefficient for west neighbor
    }

    //define east links
    for (int i = 1; i < n; i++) {
        phi_west[i] = - epsilon / (x_position[i]-x_position[i-1]); //Coefficient for east neighbor
    }

    //define center coefficients only for interior points
    for (int i = 1; i < n-1; i++) {
        phi_center[i] = - (phi_west[i] + phi_east[i]); //Coefficient for center point
        //rho[i] = 0.0; // Start with Laplace
    }
}

};

class BoundaryConditions {
    public: 
    void apply_Neumann_left(Coefficients& coeffs, ThomasAlgorithm& thomas);
    void apply_Neumann_right(Coefficients& coeffs, ThomasAlgorithm& thomas);
    void apply_Dirichlet_left(Coefficients& coeffs, ThomasAlgorithm& thomas, double V_app);
    void apply_Dirichlet_right(Coefficients& coeffs, ThomasAlgorithm& thomas, double V_app);
};

class ThomasAlgorithm {
public:
double a[n]; // Sub-diagonal coefficients
double b[n]; // Main diagonal coefficients
double c[n]; // Super-diagonal coefficients
double d[n]; // Right-hand side vector
double solution[n]; // Solution vector
double ElectricField[n]; // Electric field at each spatial point

    void ThomasAlgorithm_setup(Coefficients& coeffs, BoundaryConditions& bc) {
        // Implement the Thomas algorithm for solving tridiagonal systems
        // This will be used to solve for the electric potential at each time step

        //bc.apply_Neumann_left(coeffs, *this); // Apply Neumann boundary condition at the left boundary
        //bc.apply_Neumann_right(coeffs, *this); // Apply Neumann boundary condition at the right boundary
        bc.apply_Dirichlet_left(coeffs, *this, 10000.0); // Apply Dirichlet boundary condition at the left boundary with V_app = 10.0 kV
        bc.apply_Dirichlet_right(coeffs, *this, 0.0); // Apply Dirichlet boundary condition at the right boundary with V_app = 0.0 V

        for (int i = 1; i < n-1; i++) {
            a[i] = coeffs.phi_west[i]; // Coefficient for west neighbor
            b[i] = coeffs.phi_center[i]; // Coefficient for center point
            c[i] = coeffs.phi_east[i]; // Coefficient for east neighbor
            d[i] = dx * coeffs.rho[i]; // Right-hand side vector based on charge density times the spatial step size
            solution[i] = coeffs.Voltage[i]; // Initialize solution vector
        }
    }

    void ThomasAlgorithm_solver() {
        for (int i = 1; i < n; i++) {
            // Forward elimination
                double m = a[i] / b[i-1]; //to eliminate the left neighbor a[i]
                b[i] = b[i] - m * c[i-1];
                d[i] = d[i] - m * d[i-1];
            }
            // Back substitution
            solution[n-1] = d[n-1] / b[n-1];
            for (int i = n-2; i >= 0; i--) {
                solution[i] = (d[i] - c[i] * solution[i+1]) / b[i];
            }

        for (int i = 0; i < n - 1; i++) {
            // E = -(V_next - V_current) / dx
            ElectricField[i] = -(solution[i+1] - solution[i]) / dx;
        }
    }
};

    void BoundaryConditions::apply_Neumann_left(Coefficients& coeffs, ThomasAlgorithm& thomas) {
        // Implement Neumann boundary condition at the left boundary (cell 0)
        // This will involve modifying the coefficients in the Thomas algorithm to account for the specified electric field at the boundary
        thomas.a[0] = 0.0; // No west neighbor
        thomas.b[0] = -coeffs.phi_east[0]; // Coefficient for center point
        thomas.c[0] = coeffs.phi_east[0]; // Coefficient for east neighbor
        thomas.d[0] = dx*coeffs.rho[0]; // Right-hand side is the specified voltage at the boundary
    };

    void BoundaryConditions::apply_Neumann_right(Coefficients& coeffs, ThomasAlgorithm& thomas) {
        // Implement Neumann boundary condition at the right boundary (cell n-1)
        // This will involve modifying the coefficients in the Thomas algorithm to account for the specified electric field at the boundary
        thomas.a[n-1] = coeffs.phi_west[n-1]; // Coefficient for west neighbor
        thomas.b[n-1] = -coeffs.phi_west[n-1]; // Coefficient for center point
        thomas.c[n-1] = 0.0; // No east neighbor
        thomas.d[n-1] = dx*coeffs.rho[n-1]; // Right-hand side is the specified voltage at the boundary
    };

    void BoundaryConditions::apply_Dirichlet_left(Coefficients& coeffs, ThomasAlgorithm& thomas, double V_app) {
        // Implement Dirichlet boundary condition at the left boundary (cell 0)
        // This will involve modifying the coefficients in the Thomas algorithm to set the voltage at the boundary
        // Distance from center of cell 0 to the wall (x=0) is dx/2
        double dist_to_wall = dx / 2.0;
        double phi_west = -epsilon / dist_to_wall; // Coefficient for west neighbor (using the same spacing as interior points)
        thomas.a[0] = 0.0; // No west neighbor
        thomas.b[0] = -(phi_west + coeffs.phi_east[0]); // Coefficient for center point
        thomas.c[0] = coeffs.phi_east[0]; // Coefficient for east neighbor
        thomas.d[0] = dx*coeffs.rho[0] - (phi_west * V_app); // Right-hand side is the specified voltage at the boundary
    };

    void BoundaryConditions::apply_Dirichlet_right(Coefficients& coeffs, ThomasAlgorithm& thomas, double V_app) {
        // Implement Dirichlet boundary condition at the right boundary (cell n-1)
        // This will involve modifying the coefficients in the Thomas algorithm to set the voltage at the boundary
        // Distance from center of cell n-1 to the wall (x=L) is dx/2
        double dist_to_wall = dx / 2.0;
        double phi_east = -epsilon / dist_to_wall; // Coefficient for east neighbor (using the same spacing as interior points)
        thomas.a[n-1] = coeffs.phi_west[n-1]; // Coefficient for west neighbor
        thomas.b[n-1] = -(coeffs.phi_west[n-1] + phi_east); // Coefficient for center point
        thomas.c[n-1] = 0.0; // No east neighbor
        thomas.d[n-1] = dx*coeffs.rho[n-1] - (phi_east * V_app); // Right-hand side is the specified voltage at the boundary
    };
    
int main(){
    Coefficients coeffs;
    BoundaryConditions bc;
    ThomasAlgorithm thomas;

    thomas.ThomasAlgorithm_setup(coeffs, bc); // Set up the Thomas algorithm with the coefficients and boundary conditions
    thomas.ThomasAlgorithm_solver(); // Solve for the electric potential at each spatial point

    // Output the results
    for (int i = 0; i < n; i++) {
        cout << "Spatial Position (m): " << coeffs.x_position[i] << " | Electric Potential (V): " << thomas.solution[i] << endl;
        cout << "west coefficient: " << coeffs.phi_west[i] << "center coefficient: " << coeffs.phi_center[i] << "east coefficient: " << coeffs.phi_east[i] << endl;
        cout << "Electric Field: " << thomas.ElectricField[i] << endl;
        cout << "Charge Density: " << coeffs.rho[i] << endl;
    }


ofstream potFile("potential_data.csv");
    potFile << "cell,potential\n";
    for (int i = 0; i < n; i++) {
        potFile << i << "," << thomas.solution[i] << "\n";
    }
    potFile.close();

    // Save Electric Field Data
    ofstream eFile("efield_data.csv");
    eFile << "cell,electric_field\n";
    for (int i = 0; i < n - 1; i++) {
        eFile << i << "," << thomas.ElectricField[i] << "\n";
    }
    eFile.close();

    cout << "Data saved to potential_data.csv and efield_data.csv" << endl;

    return 0;

}
