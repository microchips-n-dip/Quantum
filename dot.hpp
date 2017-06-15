#ifndef __DOT_HH
// Dot product functions to get Eigen to actually work!
#include "ueigen.hpp"
#include <math.h>
//#include "constants.hpp"

uEigen::Matrix<double> Dot(const uEigen::Matrix<double>& A, const uEigen::Matrix<double>& B);
uEigen::Matrix<std::complex<double>> Dot(const uEigen::Matrix<std::complex<double>>& A, const uEigen::Matrix<double>& B);
uEigen::Matrix<std::complex<double>> Dot(const uEigen::Matrix<double>& A, const uEigen::Matrix<std::complex<double>>& B);
uEigen::Matrix<std::complex<double>> Dot(const uEigen::Matrix<std::complex<double>>& A, const uEigen::Matrix<std::complex<double>>& B);
uEigen::Matrix<double> Tanh(const uEigen::Matrix<double>& A);
double Sum(const uEigen::Matrix<double>& A);
double Delta(uEigen::Vector<double> a);
double Delta(double a);

#ifdef PauliMatrix
uEigen::Matrix<std::complex<double>> PauliDot(uEigen::Vector<double> a);
#endif

#define __DOT_HH
#endif