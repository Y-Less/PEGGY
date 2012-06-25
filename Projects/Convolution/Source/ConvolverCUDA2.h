#if !defined NO_CUDA_OPT

#pragma once

#include <ParallelCUDADriver.h>
#include "Convolver.h"
#include <cuda_runtime_api.h>

class ConvolverCUDA2 :
	public Convolver,
	public ParallelCUDADriver
{
public:
	// cons
		ConvolverCUDA2();
	
protected:
	virtual void
		ConvInit() throw (...);
	
	virtual void
		Execute() throw (...);
	
private:
	float
		* m_data,
		* m_smoothX,
		* m_smoothY;
};

#endif
