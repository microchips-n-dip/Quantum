#ifndef __LIEALGEBRA_HH

//#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "ueigen.hpp"
//#include "parallel.hpp"
//#include "mpiGQ.hpp"
//#include "constants.hpp"
//#include "dot.hpp"

typedef typename uEigen::Vector<std::complex<double>> uvc_t;
typedef typename uEigen::Matrix<std::complex<double>> umc_t;
typedef typename uEigen::Vector<double> uv_t;
typedef typename uEigen::Matrix<double> um_t;

typedef typename uEigen::Vector<int> Root;
template <typename T> using GroupSpace = uEigen::Vector<T>;
typedef typename uEigen::Matrix<int> CartanMatrix;

GroupSpace<int> basisVector(int dim, int n);

class RootSystem
{
  private:
	// Private variables
	GroupSpace<Root> Alpha;
	GroupSpace<Root> upperRoots;
	GroupSpace<Root> lowerRoots;
	GroupSpace<Root> roots;
	
	int CartanSize;
	CartanMatrix Cartan;
	CartanMatrix r;
	
	bool flagRComputed = false;
	bool flagAlphaInitialized = false;
	bool flagUpperRootsComputed = false;
	bool flagLowerRootsComputed = false;
	
	// Private functions
	void InitializeAlpha();
	void FromCartanComputeR();
	void ComputeUpperRoots();
	void ComputeLowerRoots();
	void ComputeRoots();
	
  public:
	Root operator[](int rootNumber);
	
	GroupSpace<Root> getFullList();
	void manualConstructor();
	void manualConstructor(const CartanMatrix& _Cartan);
	void manualSetCartan(const CartanMatrix& _Cartan);
	void manualSetDefaultAlpha(const GroupSpace<Root>& _Alpha);
	
	// Constructors
	RootSystem();
	RootSystem(const CartanMatrix& _Cartan);
};

#define __LIEALGEBRA_HH
#endif