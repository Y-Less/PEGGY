// Copyright (c) Microsoft Corporation.   All rights reserved.
#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT

#pragma once

#include "Data1D.h"

class Data1DFloat4 :
	public Data1D
{
public:
	// cons
		Data1DFloat4(const size_t);
	
	// cons
		Data1DFloat4(const size_t, const ParallelArrays::Float4 * const);
	
	// cons
		Data1DFloat4(const size_t, const GenType);
	
	// cons
		Data1DFloat4(const size_t, const GenType, const unsigned int);
	
	// cons
		Data1DFloat4(const size_t, const GenType, ds_callback);
	
	// cons
		Data1DFloat4(const size_t, const GenType, std::istream &);
	
	ParallelArrays::Float4
		operator()(const size_t) const throw (...);
	
	friend std::ostream &
		operator<<(std::ostream &, const Data1DFloat4 &);
	
protected:
	ParallelArrays::ParallelArray *
		_ToAcceleratorArray(unsigned int = 32);
};

#endif
