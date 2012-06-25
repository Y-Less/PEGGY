#if !defined NO_CUDA_DRIVER

#pragma once

#include <ParallelCUDADriver.h>
#include "Convolver.h"

class ConvolverCUDADriver :
	public Convolver,
	public ParallelCUDADriver
{
public:
	// cons
		ConvolverCUDADriver();
	
protected:
	virtual void
		ConvInit() throw (...);
	
	virtual void
		Execute() throw (...);
	
	void
		ConvExit() throw (...);
	
	float
		* m_data;
	
	CUdeviceptr
		m_smoothX,
		m_smoothY;
	
	CUfunction
		m_rowKernel,
		m_colKernel;
	
	CUmodule
		m_module;
};

#endif
