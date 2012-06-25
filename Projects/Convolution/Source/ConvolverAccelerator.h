#if !defined NO_ACCELERATOR

#pragma once

#include <ParallelAccelerator.h>
#include "Convolver.h"

class ConvolverAccelerator :
	public Convolver,
	public ParallelAccelerator
{
public:
	// cons
		ConvolverAccelerator(const ParallelAcceleratorType);
	
protected:
	virtual void
		ConvInit() throw (...);
	
	virtual void
		Execute() throw (...);
	
	ParallelArrays::FloatParallelArray
		m_data;
	
	float *
		m_filter;
};

#endif
