#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>
#include <limits>
#include <unistd.h>
#include <iostream>

#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

#include "tensorwrapper.h"

static const double pi = acos(-1.0);
static const double npi = sqrt(0.5 * pi);

// Some functions to make random variables a little more random in the interest of proper evolution
static const unsigned int urand_seed = 697;
static const unsigned int n_urand_steps = 16;

unsigned int urand()
{
	unsigned int crv = urand_seed;
	for (unsigned int i = 0; i < n_urand_steps; ++i)
	{
		srand(crv);
		crv = rand();
	}
	
	return crv;
}

double urand_bound(double a, double b)
{
	return ((double)(urand()) / RAND_MAX) * b + a;
}

// The number of symbols to use for i/o
static const unsigned int n_subvals = 5;

// Acts as a sort of augmented neuron, is represented by a bunch of vectors rather than having single values attributed
struct Nodeon
{
	// Sets m_activation's elements to the sigmoid of the associated m_inp elements
	void lf(double t)
	{
		// Print some values for debugging
		printf("%p\n", &m_inp);
		printf("%p\n", m_inp.data()); // This fails on second nodeon of evn
		for (unsigned int i = 0; i < n_subvals; ++i) {
			double e = erf(npi * m_inp.data()[i]);
			m_activation.data()[i] = e;
			if (m_activation.data()[i] > 0.5) {
				m_llft.data()[i] = t;
				(*this).do_something(i);
			}
		}
	}
	
	// Function letting different child structs do their thing
	void do_something(unsigned int c) { }
	
	Nodeon() { }
	~Nodeon() { printf("Nodeon::~Nodeon()\n"); }
	
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	// Input vector
	Eigen::Tensor<double, 1> m_inp = Eigen::Tensor<double, 1>(n_subvals); // This tensor in the second nodeon of evn has address 0x0
	// Activation vector
	Eigen::Tensor<double, 1> m_activation = Eigen::Tensor<double, 1>(n_subvals);
	// Last logged "firing" time of each associated element of m_activation
	Eigen::Tensor<double, 1> m_llft = Eigen::Tensor<double, 1>(n_subvals);
	
	// Bool indicating whether this is allowed to be removed by mutations
	static const bool removable = true;
};

struct InputNodeon : public Nodeon
{
	// Input values from a file into the nodeon
	void do_something(unsigned int c)
	{
		unsigned int i;
		fscanf(*m_input_file, "%d", &i);
		if (c == 0) {
			m_activation.data()[i] = 1.0;
		} else {
			m_activation.data()[c] = 0.0;
		}
	}
	
	InputNodeon() { }
	
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	InputNodeon(FILE** input_file)
	{
		m_input_file = input_file;
	}
	
	FILE** m_input_file;
	
	// Bool indicating whether this is allowed to be removed by mutations
	static const bool removable = false;
};

struct OutputNodeon : public Nodeon
{
	// Output to a file
	void do_something(unsigned int c)
	{
		fprintf(*m_output_file, "%d\n", c);
	}
	
	OutputNodeon() { }
	
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	OutputNodeon(FILE** output_file)
	{
		m_output_file = output_file;
	}
	
	FILE** m_output_file;
	
	// Bool indicating whether this is allowed to be removed by mutations
	static const bool removable = false;
};

// Values for the connectons
static const double w_A_plus = 0.1;
static const double w_A_minus = 0.15;
static const double w_T_plus = 0.02;
static const double w_T_minus = 0.11;
static const double m_c_tau = 0.2;

// Matrix form of a synapse using RL-STDP for plastic learning
struct Connecton
{
	// Transmit values from source nodeon to dest nodeon
	void tm()
	{
		// All this culminates in a matrix product
		std::array<Eigen::IndexPair<int>, 1> ctr1 = {Eigen::IndexPair<int>(1, 0)};
		m_dst->m_inp += tco(m_src->m_activation, m_weight, ctr1); // tco = Tensor Contraction Operation (from Eigen)
	}
	
	// STDP
	void stdp(double dop)
	{
		// Tensor addition op to subtract the two last logged firing times
		std::array<Eigen::IndexPair<int>, 0> ctr1 = {};
		Eigen::Tensor<double, 2> dt = tao(m_dst->m_llft, -m_src->m_llft, ctr1); // tao = Tensor Addition Operation (from Eigen)
		
		// Compute the eligibility
		for (unsigned int i = 0; i < m_eligibility.size(); ++i) {
			if (m_llft_dst.coeffRef(i / m_eligibility.dimensions()[1]) != m_dst->m_llft.coeffRef(i / m_eligibility.dimensions()[1])) {
				if (dt.coeffRef(i) >= 0)
					m_eligibility.coeffRef(i) += w_A_plus * exp(-dt.coeffRef(i) / w_T_plus);
				else if (dt.coeffRef(i) < 0)
					m_eligibility.coeffRef(i) -= w_A_minus * exp(dt.coeffRef(i) / w_T_minus);
			}
		}
		
		// Finish up the eligibility
		m_eligibility -= m_eligibility / m_c_tau;
		// Modify the weight
		m_weight += m_eligibility * dop;
	}
	
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	Connecton(Nodeon* src, Nodeon* dst)
	{
		m_src = src;
		m_dst = dst;
		
		// Initialize the weight matrix with random values
		for (unsigned int i = 0; i < m_weight.size(); ++i)
			m_weight.coeffRef(i) = urand_bound(-1.0, 1.0);
	}
	
	// Source nodeon
	Nodeon* m_src;
	// Dest nodeon
	Nodeon* m_dst;
	// Dest's last logged firing time
	Eigen::Tensor<double, 1> m_llft_dst = Eigen::Tensor<double, 1>(n_subvals);
	
	// Weight matrix
	Eigen::Tensor<double, 2> m_weight = Eigen::Tensor<double, 2>(n_subvals, n_subvals);
	// Eligibility matrix
	Eigen::Tensor<double, 2> m_eligibility = Eigen::Tensor<double, 2>(n_subvals, n_subvals);
};

// Mutation probabilities
static const double mutaprob_plus_nodeon = 0.2;
static const double mutaprob_plus_connecton = 0.4;
static const double mutaprob_minus_nodeon = 0.05;
static const double mutaprob_minus_connecton = 0.1;

// Some values to let the network do its thing
static const unsigned int n_samples = 1;
static const unsigned int EvNumAllowedCycles = 64;
static const unsigned int NetNumAllowedCycles = 10;

struct EvolutionaryNetwork
{
	// Run the network for <time> iterations
	void net_run(double time)
	{
		double s_time = m_network_time;
		for (m_network_time; m_network_time < s_time + time; m_network_time += 0.1) {
			for (unsigned int i = 0; i < m_nodeons.size(); ++i) {
				m_nodeons.at(i)->lf(m_network_time);
			}
			for (unsigned int i = 0; i < m_connectons.size(); ++i) {
				m_connectons.at(i)->tm();
			}
		}
	}
	
	// Network cost function
	double net_cost()
	{
		double misaligned = 0;
		for (unsigned int i = 0; i < n_samples; ++i) {
			*m_output_file = fopen(m_fname, "w+");
			const char* ifname = ("in" + std::to_string(i) + ".sif").c_str();
			*m_input_file = fopen(ifname, "r");
			const char* ofname = ("out" + std::to_string(i) + ".sof").c_str();
			FILE* out = fopen(ofname, "r");
			
			net_run(NetNumAllowedCycles);
			
			unsigned int gen_number;
			unsigned int tru_number;
			while (!feof(out) && !feof(*m_output_file)) {
				fscanf(*m_output_file, "%d", &gen_number);
				fscanf(out, "%d", tru_number);
				if (gen_number != tru_number)
					misaligned += 1.0;
			}
			
			fclose(*m_input_file);
			fclose(out);
			fclose(*m_output_file);
			const char* cmd = ("rm "+std::string(m_fname)).c_str();
			system(cmd);
		}
		
		return misaligned;
	}
	
	// Network fitness function
	double net_fitness()
	{
		double c_cost = net_cost();
		double ret = c_cost / m_last_cost - 1;
		m_net_last_cost = c_cost;
		return ret;
	}
	
	// Run some evolution steps (basically a way of including stdp)
	void ev_run(unsigned int cycles)
	{
		FILE* ofile = *m_output_file;
		for (unsigned int t = 0; t < cycles - 1; ++t) {
			double dop = net_fitness();
			for (unsigned int i = 0; i < m_connectons.size(); ++i) {
				m_connectons.at(i)->stdp(dop);
			}
		}
		*m_output_file = ofile;
		net_run(NetNumAllowedCycles);
	}
	
	// Evolutionary stage cost function
	double ev_cost()
	{
		double misaligned = 0;
		for (unsigned int i = 0; i < n_samples; ++i) {
			*m_output_file = fopen(m_fname, "w+");
			const char* ifname = ("in" + std::to_string(i) + ".sif").c_str();
			*m_input_file = fopen(ifname, "r");
			const char* ofname = ("out" + std::to_string(i) + ".sof").c_str();
			FILE* out = fopen(ofname, "r");
			
			ev_run(EvNumAllowedCycles);
			
			unsigned int gen_number;
			unsigned int tru_number;
			while (!feof(out) && !feof(*m_output_file)) {
				fscanf(*m_output_file, "%d", &gen_number);
				fscanf(out, "%d", tru_number);
				if (gen_number != tru_number)
					misaligned += 1.0;
			}
			
			fclose(*m_input_file);
			fclose(out);
			fclose(*m_output_file);
			const char* cmd = ("rm "+std::string(m_fname)).c_str();
			system(cmd);
		}
		
		return misaligned;
	}
	
	// Evolutionary stage fitness function
	double ev_fitness()
	{
		double c_cost = ev_cost();
		double ret = c_cost / m_last_cost - 1;
		m_ev_last_cost = c_cost;
		return ret;
	}
	
	// Add a nodeon
	void add_nodeon()
	{
		m_nodeons.push_back(new Nodeon());
	}
	
	// Remove a nodeon
	void remove_nodeon(unsigned int nodeon_id)
	{
		// Only do this if we're allowed to remove the nodeon
		if (m_nodeons.at(nodeon_id)->removable) {
			for (unsigned int i = 0; i < m_connectons.size(); ++i) {
				if (m_connectons.at(i)->m_src == m_nodeons.at(nodeon_id) || \
						m_connectons.at(i)->m_dst == m_nodeons.at(nodeon_id))
				{
					remove_connecton(i);
				}
			}
		
			m_nodeons.at(nodeon_id) = *m_nodeons.end();
			m_nodeons.pop_back();
		}
	}
	
	// Add a connecton
	void add_connecton(unsigned int nodeon_1, unsigned int nodeon_2)
	{
		m_connectons.push_back(new Connecton(m_nodeons.at(nodeon_1), m_nodeons.at(nodeon_2)));
	}
	
	// Remove a connecton
	void remove_connecton(unsigned int connecton_id)
	{
		m_connectons.at(connecton_id) = *m_connectons.end();
		m_connectons.pop_back();
	}
	
	// Mutate function
	void mutate()
	{
		double val;
		
		// Randomly add a random nodeon
		val = urand_bound(0.0, 1.0);
		if (val > mutaprob_plus_nodeon)
			add_nodeon();
		// Randomly remove a random nodeon
		val = urand_bound(0.0, 1.0);
		if (val > mutaprob_minus_nodeon)
			remove_nodeon(urand_bound(0.0, m_nodeons.size() - 1));
		
		// Randomly add a random connecton
		val = urand_bound(0.0, 1.0);
		if (val > mutaprob_plus_connecton)
			add_connecton(urand_bound(0.0, m_nodeons.size() - 1), \
										urand_bound(0.0, m_nodeons.size() - 1));
		// Randomly remove a random connecton
		val = urand_bound(0.0, 1.0);
		if (val > mutaprob_minus_connecton)
			remove_connecton(urand_bound(0.0, m_connectons.size() - 1));
	}
	
	EvolutionaryNetwork(unsigned int ev_number) :
		m_ev_number(ev_number)
	{
		m_fname = (std::to_string(m_ev_number) + ".txt").c_str();
		// Add input and output nodeons right away
		m_nodeons.push_back(new InputNodeon(m_input_file));
		m_nodeons.push_back(new OutputNodeon(m_output_file));
		
		// Mutate some times to get started
		mutate();
		mutate();
		mutate();
	}
	
	// Nodeon array
	std::vector<Nodeon*> m_nodeons;
	// Connecton array
	std::vector<Connecton*> m_connectons;

	// Internal file name
	const char* m_fname;
	// Files
	FILE** m_input_file;
	FILE** m_output_file;

	// For stdp and fitness functions
	double m_net_last_cost = std::numeric_limits<double>::infinity();
	double m_ev_last_cost = std::numeric_limits<double>::infinity();
	double m_network_time = 0;
	unsigned int m_ev_number;
};

// Sort based on known fitnesses
template <typename Derived, typename Dimensions>
Derived quick_sort(Derived& a, Dimensions& c)
{
	Derived greater;
	Derived equal;
	Derived less;
	
	Dimensions gtc;
	Dimensions ltc;
	
	double c0 = c.at(0);
	equal.push_back(a.at(0));
	
	for (unsigned int i = 1; i < a.size(); ++i)
	{
		double c1 = c.at(i);
		if (c1 == c0) {
			equal.push_back(a.at(i));
		} else if (c1 > c0) {
			greater.push_back(a.at(i));
			gtc.push_back(c1);
		} else if (c1 < c0) {
			less.push_back(a.at(i));
			ltc.push_back(c1);
		}
	}
	
	Derived full;
	Derived gts = quick_sort(greater, gtc);
	Derived lts = quick_sort(less, ltc);
	
	for (unsigned int i = 0; i < lts.size(); ++i)
		full.push_back(lts.at(i));
	for (unsigned int i = 0; i < equal.size(); ++i)
		full.push_back(equal.at(i));
	for (unsigned int i = 0; i < gts.size(); ++i)
		full.push_back(gts.at(i));
	
	return full;
}

// Sort and get fitnesses
template <typename Derived>
Derived sort(Derived& a)
{
	typedef std::vector<double> Dimensions;
	
	Derived greater;
	Derived equal;
	Derived less;
	
	Dimensions gtc;
	Dimensions ltc;
	
	double c0 = a.at(0)->ev_fitness();
	equal.push_back(a.at(0));
	
	for (unsigned int i = 1; i < a.size(); ++i)
	{
		double c1 = a.at(i)->ev_fitness();
		if (c1 == c0) {
			equal.push_back(a.at(i));
		} else if (c1 > c0) {
			greater.push_back(a.at(i));
			gtc.push_back(c1);
		} else if (c1 < c0) {
			less.push_back(a.at(i));
			ltc.push_back(c1);
		}
	}
	
	Derived full;
	Derived gts = quick_sort(greater, gtc);
	Derived lts = quick_sort(less, ltc);
	
	for (unsigned int i = 0; i < lts.size(); ++i)
		full.push_back(lts.at(i));
	for (unsigned int i = 0; i < equal.size(); ++i)
		full.push_back(equal.at(i));
	for (unsigned int i = 0; i < gts.size(); ++i)
		full.push_back(gts.at(i));
	
	return full;
}

int main(void)
{
	static const unsigned int NumEVNs = 1000;
	std::vector<EvolutionaryNetwork*> evn;
	for (unsigned int i = 0; i < NumEVNs; ++i)
		evn.push_back(new EvolutionaryNetwork(i));
	
	printf("Starting sort\n");
	evn = sort(evn);
}
