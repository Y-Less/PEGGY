#if !defined NO_CUDA

#pragma once

#include <ParallelCUDADriver.h>
#include "Convolver.h"

class ConvolverCUDA :
	public Convolver,
	public ParallelCUDADriver
{
public:
	// cons
		ConvolverCUDA();
	
protected:
	virtual void
		ConvInit() throw (...);
	
	virtual void
		Execute() throw (...);
	
	float
		* m_data,
		* m_smoothX,
		* m_smoothY;
};

#endif
