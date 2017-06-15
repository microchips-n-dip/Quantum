#include "physics.hpp"

mc_t S(int a, int b)
{
	mc_t s1 = Dot(DiracMatrix[a], DiracMatrix[b]);
	mc_t s2 = Dot(DiracMatrix[b], DiracMatrix[a]);
	mc_t s3 = s1 - s2;
	return 0.25 * s3;
}

double Lagrangian(const md_t& field, const md_t& dfield, const md_t& Tetrad, const Group<md_t>& dTetrad, const md_t& Metric, const md_t& Minkowski)
{
	auto _OMEGA = [&Tetrad, &dTetrad](int b, int c, int a)
	{
		double omega = 0;
		
		for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
		{
			// TODO: Add index lowering on dTetrad terms
			double omega1 = Tetrad(i, b) * Tetrad(j, c) * dTetrad[i](j, a);
			double omega2 = Tetrad(i, b) * Tetrad(j, c) * dTetrad[j](i, a);
			omega += omega1 - omega2;
		}
		
		return omega;
	};
	
	Group<md_t> OMEGA(4);
	Group<md_t> spinConnection(4);
	
	mc_t herm = field.adjoint();
	mc_t DC = Dot(herm, DiracMatrix[0]);
	mc_t SC = mc_t::Zero(4, 4);
	mc_t L1 = mc_t::Zero(4, 1);
	mc_t L2 = mc_t::Zero(4, 1);
	//mc_t L3 = 0
	
	for (int i = 0; i < 4; i++)
	{
		OMEGA[i] = md_t::Zero(4, 4);
		spinConnection[i] = md_t::Zero(4, 4);
		
		for (int j = 0; j < 4; j++)
		for (int k = 0; k < 4; k++)
			OMEGA[i](j, k) = _OMEGA(i, j, k);
	}
	
	//printf("[1/2] Done making spin connections\n");
	
	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	for (int k = 0; k < 4; k++)
	for (int l = 0; l < 4; l++)
	for (int m = 0; m < 4; m++)
	for (int n = 0; n < 4; n++)
		spinConnection[i](j, k) = Metric(i, l) * Minkowski(m, n) * Tetrad(l, m) * (OMEGA[n](j, k) + OMEGA[k](n, j) - OMEGA[j](k, n));
	
	//printf("[2/2] Done making spin connections\n");
	
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		for (int k = 0; k < 4; k++)
		for (int l = 0; l < 4; l++)
		for (int m = 0; m < 4; m++)
		{
			double sctemp1 = Tetrad(i, m) * Tetrad(i, j) * spinConnection[j](k, l);
			mc_t sctemp2 = S(l, k);
			sctemp2 *= sctemp1;
			SC += Dot(DiracMatrix[m], sctemp2);
		}
		
		md_t dfctemp = dfield.col(i);
		L1 += Dot(DiracMatrix[i], dfctemp);
		L2 += Dot(SC, field);
	}
	
	L1 *= imaginary;
	L2 *= -0.5;
	mc_t L4 = L1 + L2;
	mc_t L = Dot(DC, L4);
	assert(L.rows() == 1);
	assert(L.cols() == 1);
	
	return L(0, 0).real();
}