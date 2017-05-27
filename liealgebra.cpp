#include "liealgebra.hpp"

void RootSystem::InitializeAlpha()
{
	// Set some ints for the row number and column number for convenience
	int C_size = this->CartanSize;
	
	// Allocate space for the Alpha array
	this->Alpha.resize(C_size);
	
	// Set the flag for this function
	flagAlphaInitialized = true;
}

void RootSystem::FromCartanComputeR()
{
	// Set some ints for the row number and column number for convenience
	int C_size = this->CartanSize;
	// Allocate the r
	this->r.resize(C_size, C_size);
	
	// Compute the values of r from the Cartan matrix
	for (int i = 0; i < C_size; i++)
	for (int j = 0; j < C_size; j++)
	if (i != j) this->r(i, j) = -this->Cartan(i, j);
	else this->r(i, j) = 0;
	
	// Set the flag for this function
	this->flagRComputed = true;
}

void RootSystem::ComputeUpperRoots()
{
	// Double check that all the required functions have been run
	if (!flagRComputed)
		this->FromCartanComputeR();
	if (!flagAlphaInitialized)
		this->InitializeAlpha();
	
	// Set some ints for the row number and column number for convenience
	int C_size = this->CartanSize;
	
	// Calculate the size of the upper root array
	int upperRootsSize = 0;
	for (int i = 0; i < C_size; i++)
	for (int j = 0; j < C_size; j++)
	if (i != j)
		upperRootsSize += 1 + this->r(i, j);
	this->upperRoots.resize(upperRootsSize);
	
	int r_current = 0;
	int l = 0;
	int computedRoot;
	
	// Compute the upper roots from the Cartan matrix
	for (int i = 0; i < C_size; i++)
	for (int j = 0; j < C_size; j++)
	if (i != j)
	{
		r_current = this->r(j, i);
		
		for (int k = 0; k <= r_current; k++)
		{
			// TODO: Set some default alphas
			computedRoot = this-> Alpha[i] + k * this->Alpha[j];
			this->upperRoots[l + k] = computedRoot;
		}
		
		l += 1 + r_current;
	}
	
	// Set the flag for this function
	this->flagUpperRootsComputed = true;
}

void RootSystem::ComputeRoots()
{
	this->ComputeUpperRoots();
	//this->ComputeLowerRoots();
	
	int ur_size = this->upperRoots.size();
	int lr_size = 0;//this->lowerRoots.size();
	int tr_size = ur_size + lr_size;
	
	this->roots.resize(tr_size);
	for (int i = 0; i < ur_size; i++)
		this->roots[i] = this->upperRoots[i];
	//for (int i = 0; i < lr_size; i++)
	//	this->roots[ur_size + i] = this->lowerRoots[i];
}

RootSystem::RootSystem(const CartanMatrix& _Cartan)
{
	this->Cartan = _Cartan;
	if (this->Cartan.rows() != this->Cartan.cols())
	{
		std::cout << "Fatal Error!: Attempted to input improper Cartan matrix\n";
		exit(-1);
	}
	this->CartanSize = this->Cartan.rows();
	
	this->ComputeRoots();
}

GroupSpace<Root> RootSystem::getFullList()
{
	return this->roots;
}