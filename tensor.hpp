#ifndef __TENSOR_HH

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <vector>
#include <string>

template <typename _DerivedType>
class Tensor
{
  private:
	typedef _DerivedType Scalar;
  
	size_t TensorRank;
	size_t fullRawArraySize;
	
	bool isMetricTensor = false;
	bool isUnconfigured = true;
	
	char* IndexForms;
	char* TempIndexForms;
	char* LastTempIndexForms;
	std::vector<Tensor> transformationTensors;
	
	size_t* IndexSizes;
	size_t* IndexMultipliers;
	Scalar* rawData;
	Scalar* rawDataIndexComplete;
	
	/*
	This is where I'm having trouble
	The function is supposed to raise or lower an index using the metric
	tensor if it detects that the index has been changed
	*/
	
	void TransformIndices()
	{
		size_t fullSize = fullRawArraySize;
		//memcpy(rawDataIndexComplete, rawData, fullSize);
		
		if (isMetric == false)
		{
			for (unsigned int i = 0; i < TensorRank; i++)
		}
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
	
  public:
	Tensor(std::initializer_list<unsigned int> args)
	{
		isUnconfigured = false;
		
		TensorRank = *args.begin();
		
		transformationTensors.resize(TensorRank);
		
		IndexSizes = new size_t[TensorRank];
		IndexMultipliers = new size_t[TensorRank];
		
		IndexForms = new char[TensorRank];
		TempIndexForms = new char[TensorRank];
		LastTempIndexForms = new char[TensorRank];
		
		fullRawArraySize = 1;
	
		for (unsigned int i = 0; i < TensorRank; i++)
		{
			unsigned int val = *(args.begin() + i + 1);
			IndexMultipliers[i] = fullRawArraySize;
			fullRawArraySize *= val;
			IndexSizes[i] = val;
		}
		
		rawData = new Scalar[fullRawArraySize];
		rawDataIndexComplete = new Scalar[fullRawArraySize];
	}
	
	Tensor()
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
			else
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
	
	void setGlobalIndexTransformMatrix(const Tensor& _transformer)
	{
		for (unsigned int i = 0; i < TensorRank; i++)
			transformationTensors[i] = _transformer;
	}

	void setLocalIndexTransformMatrix(const Tensor& _transformer, size_t specificIndex)
	{
		transformationTensors[specificIndex] = _transformer;
	}

	
	void metricize()
	{
		isMetricTensor = true;
	}
	
	Scalar& operator()(std::initializer_list<unsigned int> args)
	{
		return rawDataIndexComplete[getRawLocation(args)];
	}
};

#define __TENSOR_HH
#endif