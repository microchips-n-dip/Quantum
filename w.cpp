#include "tensor.hpp"

int main(void)
{
	Tensor<double> A = Tensor<double>({2, 3, 3});
	Tensor<double> B = Tensor<double>({2, 3, 3});
	
	A.setIndices("^ij");
	B.setIndices("^jk");
	
	for (unsigned int i = 0; i < 3; i++)
	for (unsigned int j = 0; j < 3; j++)
	{
		A({i, j}) = 1;
		B({i, j}) = 1;
	}
	
	Tensor<double> C = A * B;
	
	for (unsigned int i = 0; i < 3; i++)
	for (unsigned int j = 0; j < 3; j++)
	if (C({i, j}) != 0)
		printf("C{%d, %d} = %f\n", i, j, C({i, j}));
}