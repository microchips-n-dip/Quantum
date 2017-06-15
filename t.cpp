#include "tetradnn.hpp"

int main(void)
{
	Config_t config(5);
	config << 20, 27, 27, 27, 16;
	TetradNN tetrad = TetradNN(config);
	tetrad.train();
}