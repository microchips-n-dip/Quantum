#ifndef __TENSOR_HH

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <vector>
#include <string>
#include <complex>

template <typename _DerivedType>
class TensorBase
{
  protected:
	typedef _DerivedType Scalar;
  
	size_t TensorRank;
	size_t fullRawArraySize;
	
	bool isUnconfigured = true;
	
	char* TempIndexForms;
	char* LastTempIndexForms;
	char* Indices;
	
	size_t* IndexSizes;
	size_t* IndexMultipliers;
	Scalar* rawDataIndexComplete;
	
	virtual void TransformIndices() = 0;
	
	std::vector<unsigned int> getNDIndices(unsigned int rawLocation)
	{
		std::vector<unsigned int> NDI = std::vector<unsigned int>(TensorRank);
		unsigned int ArrayIndex1D = rawLocation;
		
		for (unsigned int i = 0; i < TensorRank; i++)
		{
			unsigned int localIndex = floor(ArrayIndex1D / IndexMultipliers[i]);
			NDI[i] = localIndex;
			ArrayIndex1D -= localIndex * IndexMultipliers[i];
		}
		
		return NDI;
	}
	
	unsigned int getRawLocation(std::initializer_list<unsigned int> args)
	{
		unsigned int* localStride = IndexMultipliers;
		unsigned int* localDims = IndexSizes;
		unsigned int ind = 0;
	
		std::initializer_list<unsigned int>::iterator li;
		for (li = args.begin(); li < args.end(); ++li, ++localStride, ++localDims)
		{
			ind += *li * (*localStride);
		}
	
		return ind;
	}
	
	unsigned int getRawLocation(std::vector<unsigned int> args)
	{
		unsigned int* localStride = IndexMultipliers;
		unsigned int* localDims = IndexSizes;
		unsigned int ind = 0;
	
		std::vector<unsigned int>::iterator li;
		for (li = args.begin(); li < args.end(); ++li, ++localStride, ++localDims)
		{
			ind += *li * (*localStride);
		}
	
		return ind;
	}
	
  public:
	TensorBase()
	{
		isUnconfigured = true;
	}
  
	void setIndices(std::string _Indices)
	{
		strcpy(LastTempIndexForms, TempIndexForms);
		
		char LastFoundIndexChar = '^';
	
		for (unsigned int i = 0, j = 0; i < _Indices.size(); i++)
		{
			if (_Indices[i] == '^' && LastFoundIndexChar == '_')
				LastFoundIndexChar = '^';
			else if (_Indices[i] == '_' && LastFoundIndexChar == '^')
				LastFoundIndexChar = '_';
			else if (j < TensorRank)
			{
				TempIndexForms[j] = LastFoundIndexChar;
				Indices[j] = _Indices[i];
				j++;
			}
		}
	
		TransformIndices();
	}
	
	void setIndices(std::string& _Indices)
	{
		strcpy(LastTempIndexForms, TempIndexForms);
		
		char LastFoundIndexChar = '^';
	
		for (unsigned int i = 0, j = 0; i < _Indices.size(); i++)
		{
			if (_Indices[i] == '^' && LastFoundIndexChar == '_')
				LastFoundIndexChar = '^';
			else if (_Indices[i] == '_' && LastFoundIndexChar == '^')
				LastFoundIndexChar = '_';
			else
			{
				TempIndexForms[j] = LastFoundIndexChar;
				Indices[j] = _Indices[i];
				j++;
			}
		}
	
		TransformIndices();
	}
	
	unsigned int getIndexSize(unsigned int index)
	{
		return IndexSizes[index];
	}
	
	/*Tensor<_DerivedType> SelfContract()
	{
		;
	}*/
	
	Scalar& operator()(std::initializer_list<unsigned int> args)
	{
		return rawDataIndexComplete[getRawLocation(args)];
	}
	
	Scalar& operator()(std::vector<unsigned int> args)
	{
		return rawDataIndexComplete[getRawLocation(args)];
	}
	
	template <typename TensorType, typename ProductOpScalar>
	friend TensorType ProductOp(const TensorType& A, const TensorType& B);
};

template <typename _DerivedType>
class MetricTensor : public TensorBase<_DerivedType>
{
  protected:
	typedef _DerivedType Scalar;
  
	void TransformIndices() {;}
	
  public:
	MetricTensor(std::initializer_list<unsigned int> args)
	{
		this->isUnconfigured = false;
		
		this->TensorRank = *args.begin();
		
		this->IndexSizes = new size_t[this->TensorRank];
		this->IndexMultipliers = new size_t[this->TensorRank];
		
		this->TempIndexForms = new char[this->TensorRank];
		this->LastTempIndexForms = new char[this->TensorRank];
		this->Indices = new char[this->TensorRank];
		
		this->fullRawArraySize = 1;
	
		for (unsigned int i = 0; i < this->TensorRank; i++)
		{
			unsigned int val = *(args.begin() + i + 1);
			this->IndexMultipliers[this->TensorRank - (i + 1)] = this->fullRawArraySize;
			this->fullRawArraySize *= val;
			this->IndexSizes[this->TensorRank - (i + 1)] = val;
			
			this->TempIndexForms[i] = '^';
			this->LastTempIndexForms[i] = '^';
			this->Indices[i] = ' ';
		}
		
		this->rawDataIndexComplete = new Scalar[this->fullRawArraySize];
		
		for (unsigned int i = 0; i < this->fullRawArraySize; i++)
		{
			this->rawDataIndexComplete[i] = 0;
		}
	}
	
	MetricTensor(std::vector<unsigned int> args)
	{
		this->isUnconfigured = false;
		
		this->TensorRank = *args.begin();
		
		this->IndexSizes = new size_t[this->TensorRank];
		this->IndexMultipliers = new size_t[this->TensorRank];
		
		this->TempIndexForms = new char[this->TensorRank];
		this->LastTempIndexForms = new char[this->TensorRank];
		this->Indices = new char[this->TensorRank];
		
		this->fullRawArraySize = 1;
	
		for (unsigned int i = 0; i < this->TensorRank; i++)
		{
			unsigned int val = *(args.begin() + i + 1);
			this->IndexMultipliers[this->TensorRank - (i + 1)] = this->fullRawArraySize;
			this->fullRawArraySize *= val;
			this->IndexSizes[this->TensorRank - (i + 1)] = val;
			
			this->TempIndexForms[i] = '^';
			this->LastTempIndexForms[i] = '^';
			this->Indices[i] = ' ';
		}
		
		this->rawDataIndexComplete = new Scalar[this->fullRawArraySize];
		
		for (unsigned int i = 0; i < this->fullRawArraySize; i++)
		{
			this->rawDataIndexComplete[i] = 0;
		}
	}
	
	MetricTensor()
	{
		this->isUnconfigured = true;
	}
  
	~MetricTensor()
	{
		if (this->isUnconfigured == false)
		{
			delete[] this->rawDataIndexComplete;
		
			delete[] this->IndexSizes;
			delete[] this->IndexMultipliers;
		
			delete[] this->TempIndexForms;
			delete[] this->LastTempIndexForms;
			delete[] this->Indices;
		}
		
		else
			printf("Tensor unconfigured, nothing to do on destruction\n");
	}
};

template <typename _DerivedType>
class Tensor : public TensorBase<_DerivedType>
{
  protected:
	typedef _DerivedType Scalar;
  
	MetricTensor<Scalar>* metricTensors;
  
	void TransformIndices()
	{
		size_t fullSize = this->fullRawArraySize;
		
		Scalar* temp = new Scalar[fullSize];
		for (unsigned int i = 0; i < fullSize; i++) temp[i] = this->rawDataIndexComplete[i];
		
		for (unsigned int j = 0; j < this->TensorRank; j++)
		if (this->TempIndexForms[j] != this->LastTempIndexForms[j])
		for (unsigned int i = 0; i < fullSize; i++)
			FlipSingleIndex(j, this->getNDIndices(i), temp);
		
		for (unsigned int i = 0; i < fullSize; i++) this->rawDataIndexComplete[i] = temp[i];
		delete[] temp;
	}
	
	void FlipSingleIndex(unsigned int index, std::vector<unsigned int> args, Scalar*& _data)
	{
		std::vector<unsigned int> omit_args = args;
		Scalar temp = 0;
		
		for (unsigned int i = 0; i < this->IndexSizes[index]; i++)
		{
			omit_args[index] = i;
			temp += metricTensors[index]({args[index], i}) * _data[this->getRawLocation(omit_args)];
		}
		
		printf("Flipped index %d, got %f\n", index, temp);
		
		_data[this->getRawLocation(args)] = temp;
	}
	
	void FlipSingleIndex(unsigned int index, std::vector<unsigned int> args)
	{
		std::vector<unsigned int> omit_args = args;
		Scalar temp = 0;
		
		for (unsigned int i = 0; i < this->IndexSizes[index]; i++)
		{
			omit_args[index] = i;
			temp += metricTensors[index]({args[index], i}) * this->rawDataIndexComplete[this->getRawLocation(omit_args)];
		}
		
		this->rawDataIndexComplete[this->getRawLocation(args)] = temp;
	}
	
	Scalar fetchWithIndexTransform(unsigned int index, std::vector<unsigned int> args)
	{
		Scalar temp = 0;
		std::vector<unsigned int> omit_args = args;
		
		for (unsigned int i = 0; i < this->IndexSizes[index]; i++)
		{
			omit_args[index] = i;
			temp += metricTensors[index]({args[index], i}) * this->rawDataIndexComplete[this->getRawLocation(omit_args)];
		}
		
		return temp;
	}
	
	Scalar fetchWithIndexTransform(unsigned int index, std::vector<unsigned int> args, Scalar*& _data)
	{
		Scalar temp = 0;
		std::vector<unsigned int> omit_args = args;
		
		for (unsigned int i = 0; i < this->IndexSizes[index]; i++)
		{
			omit_args[index] = i;
			temp += metricTensors[index]({args[index], i}) * _data[this->getRawLocation(omit_args)];
		}
		
		return temp;
	}
	
  public:
	Tensor(std::initializer_list<unsigned int> args)
	{
		this->isUnconfigured = false;
		
		this->TensorRank = *args.begin();
		
		metricTensors = new MetricTensor<Scalar>[this->TensorRank];
		
		this->IndexSizes = new size_t[this->TensorRank];
		this->IndexMultipliers = new size_t[this->TensorRank];
		
		this->TempIndexForms = new char[this->TensorRank];
		this->LastTempIndexForms = new char[this->TensorRank];
		this->Indices = new char[this->TensorRank];
		
		this->fullRawArraySize = 1;
	
		for (unsigned int i = 0; i < this->TensorRank; i++)
		{
			unsigned int val = *(args.begin() + i + 1);
			this->IndexMultipliers[this->TensorRank - (i + 1)] = this->fullRawArraySize;
			this->fullRawArraySize *= val;
			this->IndexSizes[this->TensorRank - (i + 1)] = val;
			
			this->TempIndexForms[i] = '^';
			this->LastTempIndexForms[i] = '^';
			this->Indices[i] = ' ';
		}
		
		this->rawDataIndexComplete = new Scalar[this->fullRawArraySize];
		
		for (unsigned int i = 0; i < this->fullRawArraySize; i++)
		{
			this->rawDataIndexComplete[i] = 0;
		}
	}
	
	Tensor(std::vector<unsigned int> args)
	{
		this->isUnconfigured = false;
		
		this->TensorRank = *args.begin();
		
		metricTensors = new MetricTensor<Scalar>[this->TensorRank];
		
		this->IndexSizes = new size_t[this->TensorRank];
		this->IndexMultipliers = new size_t[this->TensorRank];
		
		this->TempIndexForms = new char[this->TensorRank];
		this->LastTempIndexForms = new char[this->TensorRank];
		this->Indices = new char[this->TensorRank];
		
		this->fullRawArraySize = 1;
	
		for (unsigned int i = 0; i < this->TensorRank; i++)
		{
			unsigned int val = *(args.begin() + i + 1);
			this->IndexMultipliers[this->TensorRank - (i + 1)] = this->fullRawArraySize;
			this->fullRawArraySize *= val;
			this->IndexSizes[this->TensorRank - (i + 1)] = val;
			
			this->TempIndexForms[i] = '^';
			this->LastTempIndexForms[i] = '^';
			this->Indices[i] = ' ';
		}
		
		this->rawDataIndexComplete = new Scalar[this->fullRawArraySize];
		
		for (unsigned int i = 0; i < this->fullRawArraySize; i++)
		{
			this->rawDataIndexComplete[i] = 0;
		}
	}
  
	~Tensor()
	{
		if (this->isUnconfigured == false)
		{
			delete[] this->metricTensors;
			delete[] this->rawDataIndexComplete;
		
			delete[] this->IndexSizes;
			delete[] this->IndexMultipliers;
		
			delete[] this->TempIndexForms;
			delete[] this->LastTempIndexForms;
			delete[] this->Indices;
		}
		
		else
			printf("Tensor unconfigured, nothing to do on destruction\n");
	}
  
	void setGlobalIndexMetric(const MetricTensor<Scalar>& _transformer)
	{
		for (unsigned int i = 0; i < this->TensorRank; i++)
			metricTensors[i] = _transformer;
	}

	void setLocalIndexMetric(const MetricTensor<Scalar>& _transformer, size_t specificIndex)
	{
		metricTensors[specificIndex] = _transformer;
	}
};

template <typename Scalar>
void ProductSingleIndex(unsigned int index1, unsigned int index2,
	std::vector<unsigned int> args1, std::vector<unsigned int> args2,
	TensorBase<Scalar>& _data1, TensorBase<Scalar>& _data2, Scalar* _dataC)
{
	std::vector<unsigned int> omit_args1 = args1;
	std::vector<unsigned int> omit_args2 = args2;
	Scalar temp = 0;
	
	for (unsigned int i = 0; i < _data1.getIndexSize(index1); i++)
	for (unsigned int j = 0; j < _data2.getIndexSize(index2); j++)
	{
		omit_args1[index1] = i;
		omit_args2[index2] = j;
		
		temp += _data1(omit_args1) * _data2(omit_args2); 
	}
	
	*_dataC = temp;
}

template <typename TensorType, typename ProductOpScalar>
TensorType ProductOp(const TensorType& A, const TensorType& B)
{
	TensorType tempA = A;
	TensorType tempB = B;
	
	std::vector<unsigned int> Cinit;
	
	for (unsigned int i = 0; i < tempA.TensorRank; i++)
	for (unsigned int j = i + 1; j < tempB.TensorRank; j++)
	if (tempA.Indices[i] == tempB.Indices[j])
	{
		Cinit.erase(Cinit.begin() + i);
		Cinit.erase(Cinit.begin() + tempA.TensorRank + j);
		j = tempB.TensorRank;
	}
	
	unsigned int Crank = Cinit.size();
	Cinit.insert(Cinit.begin(), Crank);
	
	TensorType C = TensorType(Cinit);
	
	for (unsigned int k = 0; k < A.TensorRank; k++)
	for (unsigned int l = 0; l < B.TensorRank; l++)
	for (unsigned int i = 0; i < A.fullRawArraySize; i++)
	for (unsigned int j = 0; j < B.fullRawArraySize; j++)
	for (unsigned int m = 0; m < C.fullRawArraySize; m++)
	if (A.Indices[k] == B.Indices[l])
		ProductSingleIndex<ProductOpScalar>(k, l, tempA.getNDIndices(i), tempB.getNDIndices(j), tempA, tempB, &C(C.getNDIndices(m)));
	else
		C(C.getNDIndices(m)) = tempA(tempA.getNDIndices(i)) * tempB(tempB.getNDIndices(j));
	
	return C;
}

Tensor<double> operator*(const Tensor<double>& A, const Tensor<double>& B)
{ return ProductOp<Tensor<double>, double>(A, B); }

#define __TENSOR_HH
#endif