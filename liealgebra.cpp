#include "liealgebra.hpp"

GroupSpace<int> basisVector(int dim, int n)
{
	GroupSpace<int> v(dim);
	for (int i = 0; i < dim; i++)
	if (i == n) v[i] = 1;
	else v[i] = 0;
	
	return v;
}

void RootSystem::InitializeAlpha()
{
	// Set some ints for the row number and column number for convenience
	int C_size = this->CartanSize;
	
	// Allocate space for the Alpha array
	this->Alpha.resize(C_size);
	
	for (int i = 0; i < C_size; i++)
	{
		if (i + 1 < C_size)
			this->Alpha[i] = basisVector(C_size, i) - basisVector(C_size, i + 1);
		else this->Alpha[i] = basisVector(C_size, i);
	}
	
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
	GroupSpace<Root> temp;
	for (int i = 0; i < C_size; i++)
	for (int j = 0; j < C_size; j++)
	if (i != j)
		upperRootsSize += 1 + this->r(i, j);
	temp.resize(upperRootsSize);
	
	int r_current = 0;
	int l = 0;
	Root computedRoot;
	
	// Compute the upper roots from the Cartan matrix
	for (int i = 0; i < C_size; i++)
	for (int j = 0; j < C_size; j++)
	if (i != j)
	{
		r_current = this->r(j, i);
		
		for (int k = 0; k <= r_current; k++)
		{
			// TODO: Set some default alphas
			computedRoot = this->Alpha[i] + k * this->Alpha[j];
			temp[l + k] = computedRoot;
		}
		
		l += 1 + r_current;
	}
	
	int newUpperRootsSize = upperRootsSize;
	
	for (int i = 0; i < upperRootsSize; i++)
	for (int j = 0; j < i; j++)
	if (temp[i] == temp[j] && i != j)
	{
		newUpperRootsSize--;
		temp[i] = -basisVector(C_size, 0);
		break;
	}
	
	this->upperRoots.resize(newUpperRootsSize);
	
	for (int i = 0, j = 0; i < upperRootsSize; i++)
	if (temp[i] != -basisVector(C_size, 0))
	{
		this->upperRoots[j] = temp[i];
		j++;
	}
	
	// Set the flag for this function
	this->flagUpperRootsComputed = true;
}

void RootSystem::ComputeLowerRoots()
{
	this->lowerRoots = -this->upperRoots;
}

void RootSystem::ComputeRoots()
{
	this->ComputeUpperRoots();
	this->ComputeLowerRoots();
	
	int ur_size = this->upperRoots.size();
	int lr_size = this->lowerRoots.size();
	int tr_size = ur_size + lr_size;
	
	this->roots.resize(tr_size);
	for (int i = 0; i < ur_size; i++)
		this->roots[i] = this->upperRoots[i];
	for (int i = 0; i < lr_size; i++)
		this->roots[ur_size + i] = this->lowerRoots[i];
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