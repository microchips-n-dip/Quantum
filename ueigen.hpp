#ifndef __UEIGEN_H

#include <Eigen/Dense>

namespace uEigen
{
	template <typename Type> using Matrix = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
	template <typename Type> using Vector = Eigen::Matrix<Type, Eigen::Dynamic, 1>;
};

#define __UEIGEN_H
#endif