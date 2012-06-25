// Copyright (c) Microsoft Corporation.   All rights reserved.

#pragma once

#include "Data2D.h"

class Data2DInt :
	public Data2D
{
public:
	// cons
		Data2DInt(const size_t, const size_t);
	
	// cons
		Data2DInt(const size_t, const size_t, const __int32 * const);
	
	// cons
		Data2DInt(const size_t, const size_t, const GenType);
	
	// cons
		Data2DInt(const size_t, const size_t, const GenType, const unsigned int);
	
	// cons
		Data2DInt(const size_t, const size_t, const GenType, ds_callback);
	
	// cons
		Data2DInt(const size_t, const size_t, const GenType, std::istream &);
	
	__int32
		operator()(const size_t, const size_t) const throw (...);
	
	friend std::ostream &
		operator<<(std::ostream &, const Data2DInt &);
	
protected:
	#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
		ParallelArrays::ParallelArray *
			_ToAcceleratorArray(unsigned int = 32);
	#endif
};
