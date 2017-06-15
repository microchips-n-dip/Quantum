#ifndef __TETRADNN_HH

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "ueigen.hpp"
#include "physics.hpp"
#include "dot.hpp"

// It's not that important to see the hpp file

typedef typename uEigen::Matrix<double> Weight_t;
typedef typename uEigen::Vector<Weight_t> WeightGroup_t;
typedef typename uEigen::Vector<int> Config_t;
typedef typename uEigen::Matrix<double> Layer_t;
typedef typename uEigen::Vector<double> Linspace_t;
typedef typename uEigen::Vector<double> RavelGroup_t;

// The actual class
class TetradNN
{
  protected:
	int nlayer;
	Config_t config;
	WeightGroup_t w;
	int ravelsize;
	
	RavelGroup_t ravel();
	void unravel(RavelGroup_t);
	double cost(const md_t&, const Group<md_t>&);
	void microtrain(const md_t&, const Group<md_t>&);
	
	md_t makeTetradMatrix(const Layer_t& e);
	md_t makeMetricMatrix(const Layer_t& e);
	Layer_t MakeInputLayer(const md_t&, const Group<md_t>&);
	
  public:
	md_t Minkowski = md_t::Zero(4, 4);
	TetradNN(const Config_t& _config);
	Layer_t forward(const Layer_t& x);
	void train();
};

#define __TETRADNN_HH
#endif