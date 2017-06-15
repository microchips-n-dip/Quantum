#include "dot.hpp"

bool debug = true;
int nlogged = 0;

uEigen::Matrix<double> Dot(const uEigen::Matrix<double>& A, const uEigen::Matrix<double>& B)
{
	if (debug) {printf("[%d] Computing Dot product\n", nlogged); nlogged++;}
	assert(A.cols() == B.rows());
	uEigen::Matrix<double> R(A.rows(), B.cols());
	for (int i = 0; i < R.rows(); i++)
		for (int j = 0; j < A.cols(); j++)
			for (int k = 0; k < R.cols(); k++)
				R(i, k) = A(i, j) * B(j, k);
			
	return R;
}

uEigen::Matrix<std::complex<double>> Dot(const uEigen::Matrix<std::complex<double>>& A, const uEigen::Matrix<double>& B)
{
	if (debug) {printf("[%d] Computing Dot product\n", nlogged); nlogged++;}
	assert(A.cols() == B.rows());
	uEigen::Matrix<std::complex<double>> R(A.rows(), B.cols());
	for (int i = 0; i < R.rows(); i++)
		for (int j = 0; j < A.cols(); j++)
			for (int k = 0; k < R.cols(); k++)
				R(i, k) = A(i, j) * B(j, k);
			
	return R;
}

uEigen::Matrix<std::complex<double>> Dot(const uEigen::Matrix<double>& A, const uEigen::Matrix<std::complex<double>>& B)
{
	if (debug) {printf("[%d] Computing Dot product\n", nlogged); nlogged++;}
	assert(A.cols() == B.rows());
	uEigen::Matrix<std::complex<double>> R(A.rows(), B.cols());
	for (int i = 0; i < R.rows(); i++)
		for (int j = 0; j < A.cols(); j++)
			for (int k = 0; k < R.cols(); k++)
				R(i, k) = A(i, j) * B(j, k);
			
	return R;
}



uEigen::Matrix<std::complex<double>> Dot(const uEigen::Matrix<std::complex<double>>& A, const uEigen::Matrix<std::complex<double>>& B)
{
	if (debug) {printf("[%d] Computing Dot product\n", nlogged); nlogged++;}
	assert(A.cols() == B.rows());
	uEigen::Matrix<std::complex<double>> R(A.rows(), B.cols());
	for (int i = 0; i < R.rows(); i++)
		for (int j = 0; j < A.cols(); j++)
			for (int k = 0; k < R.cols(); k++)
				R(i, k) = A(i, j) * B(j, k);
			
	return R;
}

uEigen::Matrix<double> Tanh(const uEigen::Matrix<double>& A)
{
	if (debug) {printf("[%d] Computing matrix sigmoid\n", nlogged); nlogged++;}
	uEigen::Matrix<double> R(A.rows(), A.cols());
	for (int i = 0; i < A.rows(); i++)
	for (int j = 0; j < A.cols(); j++)
		R(i, j) = tanh(A(i, j));
	return R;
}

double Sum(const uEigen::Matrix<double>& A)
{
	if (debug) {printf("[%d] Computing matrix sum\n", nlogged); nlogged++;}
	double R = 0;
	for (int i = 0; i < A.rows(); i++)
	for (int j = 0; j < A.cols(); j++)
		R += A(i, j);
	return R;
}

double Delta(uEigen::Vector<double> a)
{
	double zero = 1.0;
	for (int i = 0; i < a.size(); i++) if (a[i] != 0) zero = 0.0;
	return zero;
}

double Delta(double a)
{
	if (a == 0) return 1.0;
	else return 0.0;
}

#ifdef PauliMatrix
uEigen::Matrix<std::complex<double>> PauliDot(uEigen::Vector<double> a)
{
	uEigen::Matrix<std::complex<double>> r(2, 2);
	for (int i = 0; i < 3; i++)
	{
		r += a[i] * Pauli[i];
	}
	
	return r;
}
#endif