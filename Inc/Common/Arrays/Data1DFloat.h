// Copyright (c) Microsoft Corporation.   All rights reserved.

#pragma once

#include "Data1D.h"

class Data1DFloat :
	public Data1D
{
public:
	// cons
		Data1DFloat(const size_t);
	
	// cons
		Data1DFloat(const size_t, const float * const);
	
	// cons
		Data1DFloat(const size_t, const GenType);
	
	// cons
		Data1DFloat(const size_t, const GenType, const unsigned int);
	
	// cons
		Data1DFloat(const size_t, const GenType, ds_callback);
	
	// cons
		Data1DFloat(const size_t, const GenType, std::istream &);
	
	float
		operator()(const size_t) const throw (...);
	
	friend std::ostream &
		operator<<(std::ostream &, const Data1DFloat &);
	
protected:
	#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
		ParallelArrays::ParallelArray *
			_ToAcceleratorArray(unsigned int = 32);
	#endif
};
