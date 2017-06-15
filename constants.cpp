#include "constants.hpp"

const std::vector<std::vector<std::vector<std::complex<double>>>> make__DiracMatrix =
{
	{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, -1}}, 
	{{0, 0, 0, 1}, {0, 0, 1, 0}, {0, -1, 0, 0}, {-1, 0, 0, 0}},
	{{0, 0, 0, -imaginary}, {0, 0, imaginary, 0}, {0, imaginary, 0, 0}, {-imaginary, 0, 0, 0}},
	{{0, 0, 1, 0}, {0, 0, -1, 0}, {-1, 0, 0, 0}, {0, 1, 0, 0}}
};

uEigen::Vector<uEigen::Matrix<std::complex<double>>> DiracMatrix(4);

int make_DiracMatrix()
{
	for (unsigned int i = 0; i < 4; i++)
	{
		DiracMatrix[i].resize(4, 4);
		for (unsigned int j = 0; j < 4; j++)
		{
			for (unsigned int k = 0; k < 4; k++)
			{
				DiracMatrix[i](j, k) = make__DiracMatrix[i][j][k];
			}
		}
	}
	
	return 1;
}

const std::vector<std::vector<std::vector<std::complex<double>>>> make__Pauli = 
{
	{{0, 1}, {1, 0}},
	{{0, -imaginary}, {imaginary, 0}},
	{{1, 0}, {0, -1}}
};

uEigen::Vector<uEigen::Matrix<std::complex<double>>> Pauli(3);

int make_Pauli()
{
	for (int i = 0; i < 3; i++)
	{
		Pauli[i].resize(2, 2);
		for (int j = 0; j < 2; j++)
		{
			for (int k = 0; k < 2; k++)
			{
				Pauli[i](j, k) = make__Pauli[i][j][k];
			}
		}
	}
	
	return 1;
}

int mDM_state = make_DiracMatrix();
int mP_state = make_Pauli();