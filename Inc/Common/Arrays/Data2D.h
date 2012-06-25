// Copyright (c) Microsoft Corporation.   All rights reserved.

#pragma once

#include "DataStore.h"

class Data2D :
	public DataStore
{
public:
	size_t
		GetHeight() const;
	
	size_t
		GetWidth() const;
	
	size_t
		GetPitch() const;
	
	#if !defined NO_CUDA || !defined NO_CUDA_OPT
		virtual void *
			ToCUDAArray() throw (...);
	#endif
	
protected:
	// cons
		Data2D(const size_t, const size_t, const DataType);
	
	// cons
		Data2D(const size_t, const size_t, const size_t, const void * const);
	
	// cons
		Data2D(const size_t, const size_t, const DataType, const GenType);
	
	// cons
		Data2D(const size_t, const size_t, const DataType, const GenType, const unsigned int);
	
	// cons
		Data2D(const size_t, const size_t, const DataType, const GenType, ds_callback);
	
	// cons
		Data2D(const size_t, const size_t, const DataType, const GenType, std::istream &);
	
private:
	size_t
		m_height,
		m_width,
		m_pitch;
};
