#include "tetradnn.hpp"

// Graeme: Start reading around line 121

TetradNN::TetradNN(const Config_t& _config)
{
	this->config = _config;
	this->nlayer = this->config.size() - 1;
	
	this->w.resize(this->nlayer);
	this->ravelsize = 0;
	
	for (int i = 0; i < this->nlayer; i++)
	{
		this->w[i] = Weight_t::Random(this->config[i], this->config[i + 1]);
		this->ravelsize += this->config[i] * this->config[i + 1];
	}
	
	for (int j = 0; j < 4; j++) this->Minkowski(j, j) = 1;
	this->Minkowski(0, 0) = -1;
}

RavelGroup_t TetradNN::ravel()
{
	RavelGroup_t wravel(this->ravelsize);
	
	int totalOffset = 0;
	int wi_rows = 0;
	int wi_cols = 0;
	
	for (int i = 0; i < this->nlayer; i++)
	{
		wi_rows = this->w[i].rows();
		wi_cols = this->w[i].cols();
		
		for (int j = 0; j < wi_rows; j++)
		for (int k = 0; k < wi_cols; k++)
			wravel[totalOffset + j * wi_cols + k] = this->w[i](j, k);
		
		totalOffset += wi_rows * wi_cols;
	}
	
	return wravel;
}

void TetradNN::unravel(RavelGroup_t wravel)
{
	int totalOffset = 0;
	int wi_rows = 0;
	int wi_cols = 0;
	
	for (int i = 0; i < this->nlayer; i++)
	{
		wi_rows = this->w[i].rows();
		wi_cols = this->w[i].cols();
		
		for (int j = 0; j < wi_rows; j++)
		for (int k = 0; k < wi_cols; k++)
			this->w[i](j, k) = wravel[totalOffset + j * wi_cols + k];
		
		totalOffset += wi_rows * wi_cols;
	}
}

Layer_t TetradNN::forward(const Layer_t& x)
{
	Layer_t a = x;
	Layer_t z;
	
	for (int i = 0; i < this->nlayer; i++)
	{
		z = Dot(a, this->w[i]);
		a = Tanh(z);
	}
	
	return a;
}

md_t TetradNN::makeTetradMatrix(const Layer_t& e)
{
	md_t Tetrad = md_t::Zero(4, 4);
	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++) {
		Tetrad(i, j) = e(0, 4 * i + j);
	}
	
	return Tetrad;
}

md_t TetradNN::makeMetricMatrix(const Layer_t& e)
{
	assert(e.rows() >= 1 && e.cols() >= 1);
	md_t Tetrad = this->makeTetradMatrix(e);
	md_t Metric = md_t::Zero(4, 4);
	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	for (int k = 0; k < 4; k++)
	for (int l = 0; l < 4; l++)
	{
		Metric(i, j) += Tetrad(i, k) * Tetrad(j, l) * this->Minkowski(k, l);
	}
	
	return Metric;
}

Layer_t TetradNN::MakeInputLayer(const md_t& fields, const Group<md_t>& dfields)
{
	Layer_t InputLayer = Layer_t::Zero(fields.cols(), 20);
	
	for (int j = 0; j < fields.cols(); j++)
	{
		for (int i = 0; i < 4; i++)
			InputLayer(j, i) = fields(i, j);
	
		for (int i = 0; i < 16; i++)
			InputLayer(j, i + 4) = dfields[j](floor(i / 4), i % 4);
	}
	
	return InputLayer;
}

// Whew boy, this one's tough
// This cost function is really huge and is responsable for about 10 million matrix products every time it's called
double TetradNN::cost(const md_t& fields, const Group<md_t>& dfields)
{
	// This is pretty quick, actually, even though the input and results of this bit are really big matrices
	Layer_t tetrad = this->forward(this->MakeInputLayer(fields, dfields));
	int ne = tetrad.cols();
	
	double p = 1e-4;
	double J = 0;
	
	// Now I have to loop over every example, don't see how I could optimize this
	for (int i = 0; i < ne; i++)
	{
		md_t Tetrad = this->makeTetradMatrix(tetrad.row(i));
		md_t Metric = this->makeMetricMatrix(tetrad.row(i));
		
		// First we compute the numerical derivatives of the tetrads with respect to x (using the chain rule)
		Group<double> perturb1 = Group<double>::Zero(fields.rows());
		Group<md_t> numgrad1(fields.rows());
	
		for (int j = 0; j < fields.rows(); j++)
		{
			numgrad1[j] = md_t::Zero(4, 4);
			perturb1[j] = p;
		
			Layer_t e1 = this->forward(this->MakeInputLayer(fields.col(i) + perturb1, dfields.row(i)));
			Layer_t e2 = this->forward(this->MakeInputLayer(fields.col(i) - perturb1, dfields.row(i)));
		
			md_t loss1 = this->makeTetradMatrix(e1);
			md_t loss2 = this->makeTetradMatrix(e2);
		
			numgrad1[j] = (1.0 / (2 * p)) * (loss1 - loss2);
		
			perturb1[j] = 0;
		}
		
		numgrad1 = Dot(numgrad1, dfields[i]);
		
		//printf("Finished half of this cost loop\n");
		
		// Now we compute the numerical derivatives of the two Lagrangian-y terms with respect to the metric
		md_t perturb2 = md_t::Zero(4, 4);
		md_t numgrad2 = md_t::Zero(4, 4);
		md_t numgrad3 = md_t::Zero(4, 4);
		double c3 = 0;
		
		for (int j = 0; j < 4; j++)
		for (int k = 0; k < 4; k++)
		{
			perturb2(j, k) = p;
			
			double loss1 = Lagrangian(fields.col(i), dfields[i], Tetrad, numgrad1, Metric + perturb2, this->Minkowski);
			double loss2 = Lagrangian(fields.col(i), dfields[i], Tetrad, numgrad1, Metric - perturb2, this->Minkowski);
			double loss3 = loss1 * sqrt(-((Metric + perturb2).determinant()));
			double loss4 = loss1 * sqrt(-((Metric - perturb2).determinant()));
			numgrad2(j, k) = (loss1 - loss2) / (2 * p);
			numgrad3(j, k) = (loss3 - loss4) / (2 * p);
			
			// This bit is part of the actual cost computation that I found while I was visiting McGill
			double c1 = numgrad3(j, k);
			double c2 = Metric.determinant() * (numgrad2(j, k) - 0.5 * Metric(j, k) * Lagrangian(fields.col(i), dfields[i], Tetrad, numgrad1, Metric, this->Minkowski));
			c3 += pow(c2 - c1, 2);
			
			perturb2(j, k) = 0;
		}
		
		J += 0.5 * c3;
	}
	
	J *= (1.0 / ne);
	//printf("\r\t\t\t\rCost is [%f]", J);
	return J;
}

// Here's where a lot of the long-takage happens
void TetradNN::microtrain(const md_t& fields, const Group<md_t>& dfields)
{
	//std::cout << "Microtaining!\n";
	RavelGroup_t params = this->ravel();
	double p = 1e-4;
	RavelGroup_t perturb = RavelGroup_t::Zero(params.size());
	RavelGroup_t numgrad = RavelGroup_t::Zero(params.size());
	
	// You can see that I have to loop over every weight (there are 1775 of them in the test code)
	for (int i = 0; i < params.size(); i++)
	{
		// Numerical derivative bit
		perturb[i] = p;
		this->unravel(params + perturb);
		double loss1 = this->cost(fields, dfields);
		this->unravel(params - perturb);
		double loss2 = this->cost(fields, dfields);
		numgrad[i] = (loss1 - loss2) / (2 * p);
	}
	
	this->unravel(params - numgrad);
}

// The training starts here
void TetradNN::train()
{
	// We first generate a bunch of example inputs (with a 1000 sized linspace)
	// TODO: Make the fields complex valued because I totally forgot that
	int ne = 1000;
	int fsize = 4 * ne;
	int dfsize = 16 * ne;
	Group<double> lin = Group<double>::LinSpaced(ne, 0, 10);
	md_t fields(4, fsize);
	Group<md_t> dfields(dfsize);
	for (int i = 0; i < 4; i++)
	for (int j = 0; j < ne; j++)
		fields(i, 4 * j + i) = lin[j];
	
	for (int i = 0; i < dfsize; i++)
		dfields[i] = md_t(4, 4);
	
	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 4; j++)
	for (int k = 0; k < ne; k++)
		dfields[16 * k + 4 * i + j](i, j) = lin[k];
		
	// Actually do the training
	std::cout << "Solving Einstein's field equations...\n";
	for (int i = 0; i < 2; i++)
	{
		this->microtrain(fields, dfields);
	}
	std::cout << "\nSolved!\n";
}