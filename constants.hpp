#ifndef __CONSTANTS_HH

#include "ueigen.hpp"
#include <vector>
#include <math.h>
#include <vector>

#define PauliMatrix

#define e_charge 1.60217662e-19 // Elementary charge (in Coulombs)
#define imaginary std::complex<double>(0, 1) // Imaginary unit
#define hbar 1.0545718e-34 // Reduced Planck constant (in Joule seconds)
#define e_mass 9.10938356e-31 // Mass of an electron (in kilograms)
#define SMP 299792458 // Speed of light (in metres per second)
#define K 1e+150
#define g 1.0

const double pi = acos(-1);

#define mu0 (4 * pi)
#define epsilon0 8.854187817620e-12

extern const std::vector<std::vector<std::vector<std::complex<double>>>> make__DiracMatrix;
extern const std::vector<std::vector<std::vector<std::complex<double>>>> make__Pauli;

extern uEigen::Vector<uEigen::Matrix<std::complex<double>>> DiracMatrix;
extern uEigen::Vector<uEigen::Matrix<std::complex<double>>> Pauli;

int make_DiracMatrix();
int make_Pauli();

extern int mDM_state;
extern int mP_state;

#define __CONSTANTS_HH
#endif