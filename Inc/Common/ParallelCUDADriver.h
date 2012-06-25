#if !defined NO_CUDA || !defined NO_CUDA_OPT

#pragma once

#include "ParallelProcessor.h"

//#include <cutil_inline.h>
#include <cuda.h>
//#include <cutil.h>
//#include <cuda_runtime_api.h>

inline void
	TryCUDA(
		CUresult                        error)
	throw (...);

class ParallelCUDADriver :
	public ParallelProcessor
{
public:
	// cons
		ParallelCUDADriver();
	
protected:
	virtual void
		HWInit(const bool) throw (...);
	
	virtual void
		HWClose(const bool) throw (...);
	
	int
		GetWarpSize() const;
	
	int
		GetThreads() const;
	
private:
	
	typedef
		unsigned int
		gflops_t;
	
	gflops_t
		ParallelCUDADriver::
		GetGflops(
			CUdevice                        device);
	
	CUdevice
		GetFastestCard();
	
	int
		m_warpSize,
		m_threads;
	
	CUfunction
		m_kernel;
	
	CUdevice
		m_device;
	
	CUcontext
		m_context;
};

#endif
