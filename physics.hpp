#ifndef __PHYSICS_HH

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "ueigen.hpp"
#include "constants.hpp"
#include "dot.hpp"

typedef typename uEigen::Matrix<double> md_t;
typedef typename uEigen::Matrix<std::complex<double>> mc_t;
template <typename T> using Group = uEigen::Vector<T>;

double Lagrangian(const md_t& field, const md_t& dfield, const md_t& Tetrad, const Group<md_t>& dTetrad, const md_t& Metric, const md_t& Minkowski);

#define __PHYSICS_HH
#endif